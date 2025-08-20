import copy

import math
import torch
from torch.nn.functional import normalize
if torch.cuda.is_available():
    device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cpu")
else:
    device = torch.device("cpu")
import torch.nn as nn
L2norm = nn.functional.normalize

class encoder(nn.Module):
    def __init__(self, dim_layer=None, norm_layer=None, act_layer=None, drop_out=0.0, norm_last_layer=True):
        super(encoder, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm1d
        layers = []
        for i in range(1, len(dim_layer) - 1):
            layers.append(nn.Linear(dim_layer[i - 1], dim_layer[i], bias=False))
            layers.append(norm_layer(dim_layer[i]))
            layers.append(act_layer())
            if drop_out != 0.0 and i != len(dim_layer) - 2:
                layers.append(nn.Dropout(drop_out))

        if norm_last_layer:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=False))
            layers.append(nn.BatchNorm1d(dim_layer[-1], affine=False))
        else:
            layers.append(nn.Linear(dim_layer[-2], dim_layer[-1], bias=True))

        self.ffn = nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)


class projector(nn.Module):
    def __init__(self, dim_in, dim_out=None, hidden_ratio=4.0, act_layer=None):
        super(projector, self).__init__()
        dim_out = dim_out or dim_in
        dim_hidden = int(dim_in * hidden_ratio)
        act_layer = act_layer or nn.ReLU
        self.mlp = nn.Sequential(nn.Linear(dim_in, dim_hidden),
                                 act_layer(),
                                 nn.Linear(dim_hidden, dim_out))

    def forward(self, x):
        x = self.mlp(x)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x_q, x_k, mask_pos=None):
        x_q = L2norm(x_q)
        x_k = L2norm(x_k)
        N = x_q.shape[0]
        if mask_pos is None:
            mask_pos = torch.eye(N).cuda(1)
        similarity = torch.div(torch.matmul(x_q, x_k.T), self.temperature)
        similarity = -torch.log(torch.softmax(similarity, dim=1))
        nll_loss = similarity * mask_pos / mask_pos.sum(dim=1, keepdim=True)
        loss = nll_loss.mean()
        return loss
class CoxModel(nn.Module):
    def __init__(self, input_dim):
        super(CoxModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        hazard_ratio = self.linear(x)
        return hazard_ratio

class SLCGF(torch.nn.Module):
    def __init__(self, n_views, layer_dims, temperature, drop_rate=0.5):
        super(SLCGF, self).__init__()
        self.n_views = n_views
        self.online_encoder = nn.ModuleList([encoder(layer_dims[i], drop_out=drop_rate) for i in range(n_views)])
        self.target_encoder = copy.deepcopy(self.online_encoder)

        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.cross_view_decoder = nn.ModuleList([projector(layer_dims[i][-1], layer_dims[i][-1]) for i in range(n_views)])
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=64, nhead=1,
                                                                  dim_feedforward=128)
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)
        self.cl = ContrastiveLoss(temperature)
        self.COX=CoxModel(64)
        self.feature_dim = [layer_dims[i][-1] for i in range(n_views)]
    def forward(self, data, momentum,tempe,step):
        self._update_target_branch(momentum)
        z = [self.online_encoder[i](data[i]) for i in range(self.n_views)]
        r = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        features_cat, hazard_ratio, z_t = self.extract_feature(data)
        P = [self.kernel_affinity(z_t[i],tempe,step) for i in range(self.n_views)]
        l_inter = (self.cl(r[0], z_t[1], P[1]) + self.cl(r[1], z_t[0], P[0])) / 2
        l_intra = (self.cl(z[0], z_t[0], P[0]) + self.cl(z[1], z_t[1], P[1])) / 2
        loss = l_inter + l_intra
        return loss,hazard_ratio,features_cat

    @torch.no_grad()
    def kernel_affinity(self, z, temp, step):
        G = (2 - 2 * (z @ z.t())).clamp(min=0.)
        G = torch.exp(-G / temp)
        G = G / G.sum(dim=1, keepdim=True)
        G = torch.matrix_power(G, step)
        alpha = 0.5
        G = torch.eye(G.shape[0]).cuda(1) * alpha + G * (1 - alpha)
        return G
    # 更新目标编码器参数
    @torch.no_grad()
    def _update_target_branch(self, momentum):
        for i in range(self.n_views):
            for param_o, param_t in zip(self.online_encoder[i].parameters(), self.target_encoder[i].parameters()):
                param_t.data = param_t.data * momentum + param_o.data * (1 - momentum)
    def extract_feature(self, data):

        z = [self.target_encoder[i](data[i]) for i in range(self.n_views)]

        h = [self.cross_view_decoder[i](z[i]) for i in range(self.n_views)]
        h = [L2norm(h[i]) for i in range(self.n_views)]
        h = torch.cat(h,1)
        features_cat= self.TransformerEncoder(h)
        features_cat = normalize(features_cat, dim=1)
        hazard_ratio = self.COX(features_cat)
        return features_cat, hazard_ratio, z


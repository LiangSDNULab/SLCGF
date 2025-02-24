import os
from typing import Iterable
import numpy as np
import torch
import utils
from utils import MetricLogger
def train_one_epoch(model: torch.nn.Module,
                    data_loader_train: Iterable, data_loader_test: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,tempe,step,
                    args=None):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    data_loader = enumerate(metric_logger.log_every(data_loader_train, header))
    model.train(True)
    log_file_path = os.path.join(args.output_dir, f'{args.dataset}_train_loss.txt')
    with open(log_file_path, 'a') as log_file:
        total_loss_per_epoch = 0
        num_batches_per_epoch = 0
        for data_iter_step, (xs, ystatus, ytime) in data_loader:
            batch_number = ystatus.shape[0]
            mmt = args.momentum
            for i in range(args.n_views):
                xs[i] = xs[i].to(device, non_blocking=True)
            with torch.autocast('cuda', enabled=False):
                l_cl, hazard, features_cat = model(xs, mmt,tempe,step)
            ystatus = ystatus.clone().detach().to(device).float()
            ytime = ytime.clone().to(device).float()
            R_matrix_batch_train = torch.tensor(np.zeros([batch_number, batch_number], dtype=int),
                                                dtype=torch.float).to(device)
            for i in range(batch_number):
                R_matrix_batch_train[i,] = torch.tensor(
                    np.array(list(map(int, (ytime >= ytime[i])))))
            exp_hazard_ratio = torch.reshape(torch.exp(hazard), [batch_number])
            hazard_ratio = torch.reshape(hazard, [batch_number])
            loss_cox = -torch.mean(torch.mul((hazard_ratio - torch.log(torch.sum(torch.mul(exp_hazard_ratio,
                                                                                           R_matrix_batch_train),
                                                                                 dim=1))),
                                             torch.reshape(ystatus, [batch_number])))
            loss = l_cl + loss_cox
            loss_value = loss.item()
            total_loss_per_epoch += loss.item()
            num_batches_per_epoch += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            AUCtrain, C_indextrain = utils.evaluate(hazard, ytime, ystatus)
            metric_logger.update(loss=loss_value)
        average_loss_per_epoch = total_loss_per_epoch / num_batches_per_epoch
        log_file.write(f"Epoch {epoch + 1}, Average Loss: {average_loss_per_epoch:.6f},total Loss:{total_loss_per_epoch:.6f},train cindex:{C_indextrain},trainAUC{AUCtrain}\n")
        log_file.flush()

    AUC, C_index = evaluate(model, data_loader_test, device,args)
    return AUC, C_index

def evaluate(model: torch.nn.Module, data_loader_test: Iterable,
             device: torch.device,
             args=None):
    model.eval()
    extracter = model.extract_feature
    with torch.no_grad():

        for data_iter_step, (xs, ystatus, ytime) in enumerate(data_loader_test):
            batch_size = ystatus.shape[0]
            print('test size',batch_size)
            for i in range(args.n_views):
                xs[i] = xs[i].to(device, non_blocking=True)
            test_ystatus = ystatus.to(dtype=torch.float).to(device)
            test_ytime = ytime.to(dtype=torch.float).to(device)
            features, hazard, _ = extracter(xs)
            hazard = torch.reshape(hazard, [batch_size])
        AUC, C_index = utils.evaluate(hazard, test_ytime, test_ystatus)
    return AUC, C_index

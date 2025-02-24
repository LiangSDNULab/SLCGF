import argparse
import datetime
import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader

from model import SLCGF
import utils
from train_one_epoch import train_one_epoch
from data_loda import CancerDataset
warnings.filterwarnings("ignore")
def get_args_parser():

    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--dataset', default=Dataname)

    # config path
    parser.add_argument('--config_file', type=str, default=None)

    # backbone parameters
    parser.add_argument('--encoder_dim', type=list, nargs='+', default=[])
    parser.add_argument('--embed_dim', type=int, default=0)

    # model parameters
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--start_rectify_epoch', type=int, default=0)
    parser.add_argument('--momentum', type=float, default=0.98)
    parser.add_argument('--drop_rate', type=float, default=0.2)
    parser.add_argument('--n_views', type=int, default=2, help='number of views')

    # training setting
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size per GPU')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--data_norm', type=str, default='standard', choices=['standard', 'min-max', 'l2-norm'])
    parser.add_argument('--train_time', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.002,
                        help='learning rate (absolute lr)')
    parser.add_argument('--missing_rate', type=float, default=0.0)
    parser.add_argument('--device', default='cuda:3'       '',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', type=str, default='result/',
                        help='path where to save, empty for no saving')
    args = parser.parse_args()
    if args.dataset == "lung":
        args.encoder_dim = [[2000, 1000, 1000, 500, 32], [878, 1000, 1000, 500, 32]]

    if args.dataset == "colon":
        args.encoder_dim = [[2000, 1000, 1000, 500, 32], [613, 1000, 1000, 500, 32]]
    if args.dataset == "breast":
        args.encoder_dim = [[2000, 1000, 1000, 500, 32], [ 891, 1000, 1000, 500, 32]]
    if args.dataset == "aml":
        args.encoder_dim = [[2000, 1000, 1000, 500, 32], [558, 1000, 1000, 500, 32]]
    if args.dataset == "ovarian":
        args.encoder_dim = [[2000, 1000, 1000, 500, 32], [616, 1000, 1000, 500, 32]]
    if args.dataset == "liver":
        args.encoder_dim = [[2000, 1000, 1000, 500, 32], [852, 1000, 1000, 500, 32]]
    if args.dataset == "melanoma":
        args.encoder_dim = [[2000, 1000, 1000, 500, 32], [901, 1000, 1000, 500, 32]]
    if args.dataset == "gbm":
        args.encoder_dim = [[2000, 1000, 1000, 500, 32], [534, 1000, 1000, 500, 32]]
    if args.dataset == "kidney":
        args.encoder_dim = [[2000, 1000, 1000, 500, 32], [796, 1000, 1000, 500, 32]]
    if args.dataset == "sarcoma":
        args.encoder_dim = [[2000, 1000, 1000, 500, 32], [838, 1000, 1000, 500, 32]]
    if args.dataset == "filtered":
        args.encoder_dim = [[2000, 1000, 1000, 500, 32], [643, 1000, 1000, 500, 32]]
    return args
def get_cross_validation_dataloaders(dataset, num_folds=5, batch_size=64):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    dataloaders = []
    for train_idx, val_idx in kf.split(dataset):
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        val_batch_size = len(val_subset)
        print(val_batch_size)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,drop_last=False)
        val_loader = DataLoader(val_subset, batch_size=val_batch_size, shuffle=False,drop_last=False)
        dataloaders.append((train_loader, val_loader))
    return dataloaders

def train(temperature,step, args, state_logger):
    device = torch.device(args.device)
    dataset = CancerDataset(args.dataset)
    dataset.data = [torch.tensor(array, dtype=torch.float, device=device) for array in dataset.data]
    n_input = [x.shape[1] for x in dataset.data]
    print('number of input features:', n_input)
    best_AUC_list = []
    best_Cindex_list = []
    dataloaders = get_cross_validation_dataloaders(dataset, num_folds=5, batch_size=64)
    for fold, (train_loader, val_loader) in enumerate(dataloaders):

        model = SLCGF(n_views=args.n_views,
                       layer_dims=args.encoder_dim,
                       temperature=args.temperature,
                       drop_rate=args.drop_rate, )

        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        state_logger.write('Batch size: {}'.format(args.batch_size))
        state_logger.write(optimizer.__repr__())
        print('Data loaded: there are {:} samples.'.format(len(dataset)))
        best_Cindex=0
        for epoch in range(args.epochs):
            AUC_state, Cindex_state= train_one_epoch(
                model, train_loader, val_loader,
                optimizer,
                device, epoch,temperature,step,
                args
            )
            state_logger.write(
                'epoch {} Cindex = {:.4f} AUC={:.4f}'.format(epoch, Cindex_state, AUC_state))
            if args.output_dir:
                current_Cindex = Cindex_state
                if current_Cindex > best_Cindex:
                    best_Cindex = current_Cindex
                    bestCindex_model_state = model.state_dict()
                    torch.save(bestCindex_model_state, args.output_dir + str(fold) + f"checkpoint")
                    best_AUC_state = AUC_state
                    state_logger.write('epoch {} Best Cindex = {:.4f} best_AUC={:.4f}'.format(epoch,best_Cindex,best_AUC_state))
        best_AUC_list.append(best_AUC_state)
        best_Cindex_list.append(best_Cindex)
    return np.mean(best_AUC_list), np.mean(best_Cindex_list), np.std(best_AUC_list), np.std(best_Cindex_list)
def main(args,temperature,step):
    start_time = time.time()
    state_logger = utils.FileLogger(os.path.join(args.output_dir,'train_log.txt'))
    best_AUC_mean, best_Cindex_mean, best_AUC_std, best_Cindex_std= train(temperature,step,args, state_logger)
    state_logger.write('\nBest Result: AUC = {:.4f} ± {:.4f}'.format(best_AUC_mean, best_AUC_std))
    state_logger.write('\nBest Result: Cindex = {:.4f} ± {:.4f}'.format(best_Cindex_mean, best_Cindex_std))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    state_logger.write('\nTraining time {}\n'.format(total_time_str))
if __name__ == '__main__':
    dataset_list = ['ovarian']
    for Dataname in dataset_list:
        for temperature in [0.9]:
            for step in [4]:
                print(Dataname)
                args = get_args_parser()
                print(args.encoder_dim)
                args.embed_dim = args.encoder_dim[0][-1]
                args.output_dir = os.path.join(args.output_dir, Dataname)
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)

                main(args,temperature,step)


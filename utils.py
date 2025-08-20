import random

import numpy as np
import torch
from torch.backends import cudnn

import datetime
import time
from collections import defaultdict, deque

import torch
def fix_random_seeds(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.deterministic = True
        print('\nenable cudnn.deterministic, seed fixed: {}'.format(seed))
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True
class FileLogger:
    def __init__(self, output_file):
        self.output_file = output_file

    def write(self, msg, p=True):
        with open(self.output_file, mode="a", encoding="utf-8") as log_file:
            log_file.writelines(msg + '\n')
        if p:
            print(msg)

def CIndex(pred, ytime_test, ystatus_test):
    N_test = ystatus_test.shape[0]
    ystatus_test = np.squeeze(ystatus_test)
    ytime_test = np.squeeze(ytime_test)
    theta = np.squeeze(pred)
    concord = 0.
    total = 0.
    eav_count = 0
    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                if ytime_test[j] > ytime_test[i]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        concord = concord + 1
                    elif theta[j] == theta[i]:
                        concord = concord + 0.5
                        eav_count = eav_count + 1
                        print(eav_count)
    if total == 0:
        return 0
    return concord / total
def AUC(pred, ytime_test, ystatus_test):
    N_test = ystatus_test.shape[0]
    ystatus_test = np.squeeze(ystatus_test)
    ytime_test = np.squeeze(ytime_test)
    theta = np.squeeze(pred)
    total = 0
    count = 0

    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                if ytime_test[i] < 365 * 1 < ytime_test[j]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        count = count + 1
                    elif theta[j] == theta[i]:
                        count = count + 0.5

    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                if ytime_test[i] < 365 * 5 < ytime_test[j]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        count = count + 1
                    elif theta[j] == theta[i]:
                        count = count + 0.5

    for i in range(N_test):
        if ystatus_test[i] == 1:
            for j in range(N_test):
                if ytime_test[i] < 365 * 10 < ytime_test[j]:
                    total = total + 1
                    if theta[j] < theta[i]:
                        count = count + 1
                    elif theta[j] == theta[i]:
                        count = count + 0.5
    if total == 0:
        return 0

    return count / total
def evaluate( pred,ytime_test, ystatus_test):
    Auc= AUC(pred,ytime_test, ystatus_test)
    C_index= CIndex(pred,ytime_test, ystatus_test)
    return Auc, C_index


class AverageMeter(object):
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.total += value * n
        self.count += n

    @property
    def average(self):
        return self.total / self.count if self.count != 0 else 0.0

    def __str__(self):
        return f"{self.average:.4f}"
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(AverageMeter)
        self.delimiter = delimiter
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        return self.delimiter.join(
            f"{name}: {meter}" for name, meter in self.meters.items()
        )

    def log_every(self, iterable, header=None):
        header = header or ''
        total = len(iterable)
        for i, obj in enumerate(iterable):
            yield obj
            print(f"{header} [{i}/{total}] {self}")

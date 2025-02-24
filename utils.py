
import numpy as np
import datetime
import time
from collections import defaultdict, deque

import torch
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


class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            )
class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
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
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)
    def log_every(self, iterable, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'iter_time: {time}',
            'data_time: {data}'
        ]
        log_msg = self.delimiter.join(log_msg)

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            print(log_msg.format(
                i, len(iterable),
                time=str(iter_time), data=str(data_time),
               ))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))
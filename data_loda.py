import csv
import os.path

import numpy as np
import pandas as pd
from torch.utils.data import ConcatDataset, Dataset

def normalization_v1(input):
    """ Python implementation of NormalizeFea.m """
    # be aware that the input data has to be N * d, rows represent number of samples
    # nSmp = input.shape[0]
    feaNorm = np.maximum(1e-14, pow(input, 2).sum(1))
    feaNorm = np.diag(pow(feaNorm, -0.5))
    return np.matmul(feaNorm, input)

def normalization_v2(input, unitNorm=True):
    """ mean 0 std 1, with unit norm """
    # be aware that the input data has to be N * d, rows represent number of samples
    sampleMean = np.mean(input, axis=1).reshape(input.shape[0], 1) # reshape the vector to a N * 1 matrix
    sampleStd = np.std(input, axis=1).reshape(input.shape[0], 1)

    # take advantage of the broadcasting operation in Numpy
    input = (input - sampleMean) / sampleStd
    sampleNorm = np.linalg.norm(input, axis=1).reshape(input.shape[0], 1)

    # transform to unit norm
    if unitNorm:
        input = input / sampleNorm

    return input


def load_cancer_data(CancerName):
   # filePath = path + '/' + fileName
    RNASeq_dataframe = pd.read_csv('/home/lcheng/LiZhiMin/cancerdata/'+CancerName+'/'+CancerName+'exp.csv')
    RNASeq_names = RNASeq_dataframe.iloc[:, 0].tolist()
    RNASeq_dataframe = RNASeq_dataframe.iloc[:, 1:]  # 去掉第一列
    miRNA_dataframe = pd.read_csv('/home/lcheng/LiZhiMin/cancerdata/'+CancerName+'/'+CancerName+'mirna.csv')
    miRNA_dataframe = miRNA_dataframe.iloc[:, 1:]
    clinical_dataframe = pd.read_csv('/home/lcheng/LiZhiMin/cancerdata/'+CancerName+'/'+CancerName+'_survival.csv')
    RNASeq_feature = np.array(RNASeq_dataframe)

    miRNA_feature = np.array(miRNA_dataframe)
    clinical_feature = np.array(clinical_dataframe)
    ystatus = np.squeeze(clinical_feature[:, 1]).astype(float)
    ytime = np.squeeze(clinical_feature[:, 2]).astype(float)

    data = (normalization_v2(RNASeq_feature),
                  normalization_v2(miRNA_feature))

    return data, ystatus, ytime,RNASeq_names

class CancerDataset(Dataset):
    def __init__(self, CancerName):
        self.data, self.ystatus, self.ytime,self.RNASeq_names = load_cancer_data(CancerName)

    def __len__(self):
        return len(self.ystatus)

    def __getitem__(self, idx):
        data_sample = [view[idx] for view in self.data]
        ystatus_sample = self.ystatus[idx]
        ytime_sample = self.ytime[idx]


        return data_sample, ystatus_sample, ytime_sample



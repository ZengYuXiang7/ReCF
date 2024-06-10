# coding : utf-8
# Author : yuxiang Zeng


import numpy as np
import torch
from utils.dataloader import get_dataloaders


class experiment:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def load_data(args):
        string = args.path + '/' + args.dataset + 'Matrix' + '.txt'
        tensor = np.loadtxt(open(string, 'rb'))
        return tensor

    @staticmethod
    def preprocess_data(data, args):
        data[data == -1] = 0
        return data



# 数据集定义
class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path
        self.data = exper_type.load_data(args)
        self.data = exper_type.preprocess_data(self.data, args)
        self.train_tensor, self.valid_tensor, self.test_tensor, self.max_value = self.get_train_valid_test_dataset(self.data, args)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_tensor, self.valid_tensor, self.test_tensor, exper_type, args)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, args)
        args.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)}')

    def get_dataset(self, train_tensor, valid_tensor, test_tensor, exper_type, args):
        return (
            TensorDataset(train_tensor, exper_type, args),
            TensorDataset(valid_tensor, exper_type, args),
            TensorDataset(test_tensor, exper_type, args)
        )

    def get_train_valid_test_dataset(self, tensor, args):
        quantile = np.percentile(tensor, q=100)
        tensor = tensor / (np.max(tensor))
        trainsize = int(np.prod(tensor.size) * args.density)
        validsize = int((np.prod(tensor.size)) * 0.05)
        # validsize = int(np.prod(tensor.size) * (1 - args.density))
        rowIdx, colIdx = tensor.nonzero()
        p = np.random.permutation(len(rowIdx))
        rowIdx, colIdx = rowIdx[p], colIdx[p]
        trainRowIndex = rowIdx[:trainsize]
        trainColIndex = colIdx[:trainsize]
        traintensor = np.zeros_like(tensor)
        traintensor[trainRowIndex, trainColIndex] = tensor[trainRowIndex, trainColIndex]
        validStart = trainsize
        validRowIndex = rowIdx[validStart:validStart + validsize]
        validColIndex = colIdx[validStart:validStart + validsize]
        validtensor = np.zeros_like(tensor)
        validtensor[validRowIndex, validColIndex] = tensor[validRowIndex, validColIndex]
        testStart = validStart + validsize
        testRowIndex = rowIdx[testStart:]
        testColIndex = colIdx[testStart:]
        testtensor = np.zeros_like(tensor)
        testtensor[testRowIndex, testColIndex] = tensor[testRowIndex, testColIndex]
        return traintensor, validtensor, testtensor, quantile


class TensorDataset(torch.utils.data.Dataset):

    def __init__(self, tensor, exper, args):
        self.tensor = tensor
        self.indices = self.get_pytorch_index(tensor)

    def __getitem__(self, idx):
        output = self.indices[idx, :-1]  # 去掉最后一列
        inputs = tuple(torch.as_tensor(output[i]).long() for i in range(output.shape[0]))
        value = torch.as_tensor(self.indices[idx, -1])  # 最后一列作为真实值
        return inputs, value

    def __len__(self):
        return self.indices.shape[0]

    def get_pytorch_index(self, data):
        userIdx, servIdx = data.nonzero()
        values = data[userIdx, servIdx]
        idx = torch.as_tensor(np.vstack([userIdx, servIdx, values]).T)
        return idx

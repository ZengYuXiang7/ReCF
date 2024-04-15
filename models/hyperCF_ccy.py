# coding : utf-8
# Author : yuxiang Zeng

import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from time import time

class HTCF(torch.nn.Module):
    def __init__(self, train_tensor, user_num, serv_num, args):
        super(HTCF, self).__init__()
        self.args = args
        self.user_num = user_num
        self.serv_num = serv_num

        # 注意力机制的参数
        self.dim = args.dimension
        self.head_num = args.head_num
        self.hyperNum = args.hyperNum
        self.K = nn.Parameter(torch.randn(self.dim, self.dim))

        self.user_embeds = torch.nn.Embedding(user_num, self.dim)
        self.serv_embeds = torch.nn.Embedding(serv_num, self.dim)

        self.VMapping = nn.Parameter(torch.randn(self.dim, self.dim))
        self.fc1 = torch.nn.Linear(self.hyperNum, self.hyperNum)
        self.fc2 = torch.nn.Linear(self.hyperNum, self.hyperNum)

        # 注意力机制的参数
        self.dim = args.dimension
        self.head_num = args.head_num
        self.hyperNum = args.hyperNum
        self.actorchunc = torch.nn.ReLU()  # 激活函数为ReLU

        # 超图部分
        self.uEmbed_ini = torch.nn.Parameter(torch.empty(self.user_num, self.dim))
        self.iEmbed_ini = torch.nn.Parameter(torch.empty(self.serv_num, self.dim))

        # 转换为稀疏张量表示
        self.train_tensor = train_tensor
        idx, data, shape = self.transToLsts(self.train_tensor)
        tpidx, tpdata, tpshape = self.transToLsts(self.transpose(self.train_tensor))
        self.adj = torch.sparse_coo_tensor(idx, data, shape)
        self.tpadj = torch.sparse_coo_tensor(tpidx, tpdata, tpshape)

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2 * args.dimension, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

        self.uHyper = torch.empty(self.hyperNum, self.dim)
        torch.nn.init.xavier_normal_(self.uHyper)
        self.iHyper = torch.empty(self.hyperNum, self.dim)
        torch.nn.init.xavier_normal_(self.iHyper)

        self.init()

    def init(self):
        init.xavier_uniform_(self.uEmbed_ini)
        init.xavier_uniform_(self.iEmbed_ini)

    def transpose(self, mat):
        coomat = sp.coo_matrix(mat)
        return csr_matrix(coomat.transpose())

    def transToLsts(self, mat):
        shape = [mat.shape[0], mat.shape[1]]
        coomat = sp.coo_matrix(mat)
        indices = np.array(list(map(list, zip(coomat.row, coomat.col))), dtype=np.int32)
        data = coomat.data.astype(np.float32)
        rowD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=1) + 1e-8) + 1e-8)))
        colD = np.squeeze(np.array(1 / (np.sqrt(np.sum(mat, axis=0) + 1e-8) + 1e-8)))
        for i in range(len(data)):
            row = indices[i, 0]
            col = indices[i, 1]
            data[i] = data[i] * rowD[row] * colD[col]

        if mat.shape[0] == 0:
            indices = np.array([[0, 0]], dtype=np.int32)
            data = np.array([0.0], np.float32)
        return indices.T, data, shape

    # 多跳节点之间的的信息传播
    def GCN(self, ulat, ilat, adj, tpadj):
        ulats = [ulat]
        ilats = [ilat]
        adj = adj.to(self.args.device)
        tpadj = tpadj.to(self.args.device)
        ulats[-1] = ulats[-1].to(self.args.device)
        ilats[-1] = ilats[-1].to(self.args.device)
        for i in range(self.args.order):
            temulat = torch.mm(adj, ilats[-1])
            temilat = torch.mm(tpadj, ulats[-1])
            ulats.append(temulat)
            ilats.append(temilat)
        return ulats, ilats



    # 准备注意力机制中的key
    def prepareKey(self, nodeEmbed):
        key = torch.matmul(nodeEmbed, self.K)  # Matrix multiplication
        key = key.view(-1, self.head_num, self.dim // self.head_num)  # Reshape
        key = key.transpose(0, 1)  # Transpose to get [head_num, N, dimension // head_num]
        return key


    # key 被用于与最后一层的嵌入（lats[-1]）以及超图嵌入（hyper）相乘，从而进行信息的传播和聚合。
    def propagate(self, lats, key, hyper):
        lstLat = torch.matmul(lats[-1], self.VMapping).view(-1, self.head_num, self.dim // self.head_num)  # 339*2*16
        lstLat = lstLat.permute(1, 2, 0)  # Head * d' * N  2*16*339
        # key.size() 2*339*16
        temlat1 = torch.matmul(lstLat, key)  # Head * d' * d' 2*16*16
        hyper = hyper.view(-1, self.head_num, self.dim // self.head_num)
        hyper = hyper.permute(1, 2, 0)  # Head * d' * E 2*16*128
        hyper = hyper.to(self.args.device)
        temlat1 = torch.matmul(temlat1, hyper).view(self.dim, -1)  # d * E 32*128
        temlat2 = self.fc1(temlat1)
        temlat2 = F.leaky_relu(temlat2) + temlat1  # 应用激活函数并加上残差连接
        temlat3 = self.fc2(temlat2)
        temlat3 = F.relu(temlat3) + temlat2  # 再次应用激活函数和残差连接
        preNewLat = torch.matmul(temlat3.permute(1, 0), self.VMapping).view(-1, self.head_num, self.dim // self.head_num)
        preNewLat = preNewLat.permute(1, 0, 2)  # Head * E * d'
        preNewLat = torch.matmul(hyper, preNewLat)  # Head * d'(lowrank) * d'(embed)
        newLat = torch.matmul(key, preNewLat)  # Head * N * d'
        newLat = newLat.permute(1, 0, 2).reshape(-1, self.dim)
        lats.append(newLat)
        return lats


    def hyper_gnn(self, random_embed, Embed_ini, hyper_embed):
        Hyper_ini = random_embed
        Embed_gcn = torch.sum(torch.stack(hyper_embed[1:], dim=0), dim=0)
        Embed_ini = Embed_ini
        Embed0 = Embed_ini + Embed_gcn
        Key = self.prepareKey(Embed0)
        lats = [Embed0]
        for i in range(self.args.order):
            lats = self.propagate(lats, Key, Hyper_ini)
        lat = torch.sum(torch.stack(lats), dim=0)
        return lat


    def forward(self, userIdx, itemIdx):
        UIndex = torch.arange(339).to(self.args.device)
        user_embeds = self.user_embeds(UIndex)
        SIndex = torch.arange(5825).to(self.args.device)
        serv_embeds = self.serv_embeds(SIndex)

        ulats, ilats = self.GCN(user_embeds, serv_embeds, self.adj, self.tpadj)

        user_embeds_now = self.hyper_gnn(self.uHyper, user_embeds, ulats)[userIdx]
        serv_embeds_now = self.hyper_gnn(self.iHyper, serv_embeds, ilats)[itemIdx]

        estimated = self.layers(torch.cat((user_embeds_now, serv_embeds_now), dim=-1)).sigmoid().reshape(-1)

        return estimated

    def prepare_test_model(self):
        pass

# coding : utf-8
# Author : yuxiang Zeng

import numpy as np
import scipy.sparse as sp
from dgl.nn.pytorch import SAGEConv
from scipy.sparse import csr_matrix
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from time import time
from models.GATCF import create_graph

class GraphSAGEConv(torch.nn.Module):
    def __init__(self, graph, dim, order, args):
        super(GraphSAGEConv, self).__init__()
        self.args = args
        self.order = order
        self.graph = graph
        self.layers = torch.nn.ModuleList([SAGEConv(dim, dim, aggregator_type='gcn') for _ in range(order)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(dim) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ELU() for _ in range(order)])

    def forward(self, emebeds):
        grpah = self.graph.to(self.args.device)
        grpah.ndata['L0'] = emebeds
        feats = grpah.ndata['L0']
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(grpah, feats).squeeze()
            feats = norm(feats)
            feats = act(feats)
            grpah.ndata[f'L{i + 1}'] = feats
        embeds = grpah.ndata[f'L{self.order}']
        return embeds

class HyperCF(torch.nn.Module):
    def __init__(self, user_num, serv_num, args):
        super(HyperCF, self).__init__()
        self.args = args
        self.user_num = user_num
        self.serv_num = serv_num

        # 注意力机制的参数
        self.dim = args.dimension
        self.head_num = args.head_num
        self.hyperNum = args.hyperNum

        # Attention
        self.K = torch.nn.Parameter(torch.randn(self.dim, self.dim))
        self.VMapping = torch.nn.Parameter(torch.randn(self.dim, self.dim))


        # 上下文图
        try:
           self.userg = pickle.load(open('./models/pretrain/userg.pkl', 'rb'))
           self.servg = pickle.load(open('./models/pretrain/servg.pkl', 'rb'))
        except:
            user_lookup, serv_lookup, self.userg, self.servg = create_graph()
            pickle.dump(self.userg, open('./models/pretrain/userg.pkl', 'wb'))
            pickle.dump(self.servg, open('./models/pretrain/servg.pkl', 'wb'))

        self.user_gcn = GraphSAGEConv(self.userg, self.dim, args.order, args)
        self.serv_gcn = GraphSAGEConv(self.servg, self.dim, args.order, args)

        self.user_embeds = torch.nn.Embedding(self.userg.number_of_nodes(), self.dim)
        self.serv_embeds = torch.nn.Embedding(self.servg.number_of_nodes(), self.dim)

        self.fc1 = torch.nn.Linear(self.hyperNum, self.hyperNum)
        self.fc2 = torch.nn.Linear(self.hyperNum, self.hyperNum)

        # 注意力机制的参数
        self.dim = args.dimension
        self.head_num = args.head_num
        self.hyperNum = args.hyperNum
        self.actorchunc = torch.nn.ReLU()  # 激活函数为ReLU

        # 与SHT衔接
        # idx, data, shape = self.transToLsts(self.train_tensor)
        # tpidx, tpdata, tpshape = self.transToLsts(self.transpose(self.train_tensor))
        # self.adj = torch.sparse_coo_tensor(idx, data, shape)
        # self.tpadj = torch.sparse_coo_tensor(tpidx, tpdata, tpshape)

        self.interaction = torch.nn.Sequential(
            torch.nn.Linear(2 * args.dimension, args.dimension),
            torch.nn.LayerNorm(args.dimension),
            torch.nn.ReLU(),
            torch.nn.Linear(args.dimension, args.dimension // 2),
            torch.nn.LayerNorm(args.dimension // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(args.dimension // 2, 1)
        )

        # 超图部分
        self.uHyper = torch.empty(self.hyperNum, self.dim)
        torch.nn.init.xavier_normal_(self.uHyper)
        self.iHyper = torch.empty(self.hyperNum, self.dim)
        torch.nn.init.xavier_normal_(self.iHyper)

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
    def propagate(self, embeds, key, edge_embeds):

        # 第一步： node -> edge
        # query = user/item    key = user/item  value = edge_embeds
        lstLat = torch.matmul(embeds[-1], self.VMapping).view(-1, self.head_num, self.dim // self.head_num)  # 339*2*16
        lstLat = lstLat.permute(1, 2, 0)  # Head * d' * N  2*16*339

        # key.size() 2*339*16
        temlat1 = torch.matmul(lstLat, key)  # Head * d' * d' 2*16*16
        edge_embeds = edge_embeds.view(-1, self.head_num, self.dim // self.head_num)
        edge_embeds = edge_embeds.permute(1, 2, 0)  # Head * d' * E 2*16*128
        edge_embeds = edge_embeds.to(self.args.device)

        temlat1 = torch.matmul(temlat1, edge_embeds).view(self.dim, -1)  # d * E 32*128

        # HGNN = ø(HX + X)  HGNN^2
        temlat2 = self.fc1(temlat1)
        temlat2 = F.leaky_relu(temlat2) + temlat1  # 应用激活函数并加上残差连接
        temlat3 = self.fc2(temlat2)
        temlat3 = F.relu(temlat3) + temlat2  # 再次应用激活函数和残差连接

        # Final edge embedding = temlat3


        # 第二步： edge -> node
        # query = 12   key = user/item   value =
        # key * (final edge_embeds * edge_embeds)
        preNewLat = torch.matmul(temlat3.permute(1, 0), self.VMapping).view(-1, self.head_num, self.dim // self.head_num)
        preNewLat = preNewLat.permute(1, 0, 2)  # Head * E * d'
        preNewLat = torch.matmul(edge_embeds, preNewLat)  # Head * d'(lowrank) * d'(embed)
        newLat = torch.matmul(key, preNewLat)  # Head * N * d'
        newLat = newLat.permute(1, 0, 2).reshape(-1, self.dim)
        embeds.append(newLat)

        return embeds

    def hyper_gnn(self, edge_embeds, embeds, gcn_embeds):

        Embed0 = embeds + gcn_embeds

        # 3.2.1
        Key = self.prepareKey(Embed0)

        # 3.2.2
        lats = [Embed0]
        for i in range(self.args.order):
            lats = self.propagate(lats, Key, edge_embeds)

        # 3.2.3
        lat = torch.sum(torch.stack(lats), dim=0)

        return lat

    def forward(self, userIdx, itemIdx):
        UIndex = torch.arange(self.userg.number_of_nodes()).to(self.args.device)
        user_embeds = self.user_embeds(UIndex)
        SIndex = torch.arange(self.servg.number_of_nodes()).to(self.args.device)
        serv_embeds = self.serv_embeds(SIndex)
        # u_gcn_lats, i_gcn_lats = self.GCN(user_embeds, serv_embeds, self.adj, self.tpadj)

        u_gcn_lats = self.user_gcn(user_embeds)
        i_gcn_lats = self.serv_gcn(serv_embeds)

        user_embeds_now = self.hyper_gnn(self.uHyper, user_embeds, u_gcn_lats)[userIdx]
        serv_embeds_now = self.hyper_gnn(self.iHyper, serv_embeds, i_gcn_lats)[itemIdx]

        estimated = self.interaction(torch.cat((user_embeds_now, serv_embeds_now), dim=-1)).sigmoid().reshape(-1)
        return estimated

    def prepare_test_model(self):
        pass

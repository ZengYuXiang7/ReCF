# coding : utf-8
# Author : yuxiang Zeng
import torch as t
from torch import nn
import torch.nn.functional as F

import torch
import scipy.sparse as sp
import numpy as np
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


def normalizeAdj(mat):
    degree = np.array(mat.sum(axis=-1))
    dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
    dInvSqrt[np.isinf(dInvSqrt)] = 0.0
    dInvSqrtMat = sp.diags(dInvSqrt)
    return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()


def makeTorchAdj(mat):
    # make ui adj
    a = sp.csr_matrix((339, 339))
    b = sp.csr_matrix((5825, 5825))
    mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
    mat = (mat != 0) * 1.0
    # mat = (mat + sp.eye(mat.shape[0])) * 1.0
    mat = normalizeAdj(mat)
    # make torch tensor
    idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = torch.from_numpy(mat.data.astype(np.float32))
    shape = torch.Size(mat.shape)
    return torch.sparse.FloatTensor(idxs, vals, shape)

class DNNInteraction(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DNNInteraction, self).__init__()
        self.input_dim = input_dim
        self.NeuCF = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim // 2),  # FFN
            torch.nn.LayerNorm(hidden_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(hidden_dim // 2, hidden_dim // 2),  # FFN
            torch.nn.LayerNorm(hidden_dim // 2),  # LayerNorm
            torch.nn.ReLU(),  # ReLU
            torch.nn.Linear(hidden_dim // 2, output_dim)  # y
        )

    def forward(self, x):
        outputs = self.NeuCF(x)
        return outputs


class HyperModel(nn.Module):
    def __init__(self, train_tensor, args):
        super(HyperModel, self).__init__()
        self.args = args
        self.uEmbeds = nn.Parameter(init(t.empty(args.user_num, args.dimension)))
        self.iEmbeds = nn.Parameter(init(t.empty(args.serv_num, args.dimension)))
        self.uHyper = nn.Parameter(init(t.empty(32, args.dimension)))
        self.iHyper = nn.Parameter(init(t.empty(32, args.dimension)))
        train_tensor[train_tensor != 0] = 1
        self.adj = makeTorchAdj(sp.coo_matrix(train_tensor)).to(self.args.device)
        self.interaction = DNNInteraction(args.dimension * 2, args.dimension, 1)

    def gcnLayer(self, adj, embeds):
        return t.spmm(adj, embeds)

    def hgnnLayer(self, embeds, hyper):
        return embeds @ (hyper.T @ hyper)  # @ (embeds.T @ embeds)

    def forward(self, userIdx, servIdx):
        embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)

        for i in range(self.args.order):
            embeds = self.gcnLayer(self.adj, embeds)

        # this detach helps eliminate the mutual influence between the local GCN and the global HGNN
        hyperUEmbeds = self.hgnnLayer(embeds[:self.args.user_num].detach(), self.uHyper)[userIdx]
        hyperIEmbeds = self.hgnnLayer(embeds[self.args.user_num:].detach(), self.iHyper)[servIdx]

        estimated = self.interaction(torch.cat((hyperUEmbeds, hyperIEmbeds), dim = 1))
        return estimated


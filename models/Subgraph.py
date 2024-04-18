# coding : utf-8
# Author : yuxiang Zeng
import torch
import dgl
import dgl.nn.pytorch as dglnn

# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import time
import torch
import pickle
import numpy as np
import pandas as pd
from tqdm import *

import dgl as d
from dgl.nn.pytorch import SAGEConv

from models.GraphMF import create_graph


class GraphSAGEConv(torch.nn.Module):
    def __init__(self, graph, dim, order, args):
        super(GraphSAGEConv, self).__init__()
        self.args = args
        self.order = order
        self.graph = graph
        self.embedding = torch.nn.Parameter(torch.Tensor(self.graph.number_of_nodes(), dim))
        torch.nn.init.kaiming_normal_(self.embedding)
        self.graph.ndata['L0'] = self.embedding
        self.layers = torch.nn.ModuleList([SAGEConv(dim, dim, aggregator_type='gcn') for _ in range(order)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(dim) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ELU() for _ in range(order)])

    def forward(self, index):
        # grpah = self.graph.to(self.args.device)
        # 获得子图
        subgraph, _ = dgl.khop_in_subgraph(self.graph, index, 2)
        # subgraph = self.get_subgraph(self.graph, index)
        # print(subgraph)
        # print(subgraph.ndata['L0'].shape)
        # print(subgraph.ndata['L0'][index].shape)
        # print(index.max())
        # print(subgraph.nodes())
        # exit()
        # print(self.graph.nodes())
        # print('-' * 80)
        # 创建原始索引到子图索引的映射
        # print(index)
        index_map = {int(original): i for i, original in enumerate(subgraph.ndata['_ID'].tolist())}
        # print(index_map)

        subgraph_node_id = []
        for i in range(len(index)):
            subgraph_node_id.append(index_map[index[i].item()])
            # print(subgraph_node_id)
        # subgraph_node_id = index_map[index]
        subgraph_node_id = torch.tensor(subgraph_node_id)
        # print(subgraph_node_id)
        # print(len(subgraph.ndata['original_indices']))
        # print(len(subgraph.nodes()))
        # print(subgraph.nodes())
        # print(subgraph.ndata['_ID'])
        # exit()

        # print(subgraph_node_id)
        feats = subgraph.ndata['L0']
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(subgraph, feats).squeeze()
            feats = norm(feats)
            feats = act(feats)
            subgraph.ndata[f'L{i + 1}'] = feats
        # print(index)
        # embeds = subgraph.ndata[f'L{self.order}'][index]
        embeds = subgraph.ndata[f'L{self.order}'][subgraph_node_id]
        return embeds

    def get_subgraph(self, graph, indices):
        k = 2
        # 提取入向k跳子图
        subgraph_in, _ = dgl.khop_in_subgraph(graph, indices, k)

        # 提取出向k跳子图
        subgraph_out, _ = dgl.khop_out_subgraph(graph, indices, k)

        # 合并两个子图
        nodes_union = torch.unique(torch.cat([subgraph_in.nodes(), subgraph_out.nodes()], dim=0))
        subgraph_union = graph.subgraph(nodes_union)

        subgraph_union.ndata['original_indices'] = nodes_union

        return subgraph_union

class SubgraphCF(torch.nn.Module):
    def __init__(self, args):
        self.args = args
        super(SubgraphCF, self).__init__()
        try:
            userg = pickle.load(open('./models/pretrain/userg.pkl', 'rb'))
            servg = pickle.load(open('./models/pretrain/servg.pkl', 'rb'))
        except:
            user_lookup, serv_lookup, userg, servg = create_graph()
            pickle.dump(userg, open('./models/pretrain/userg.pkl', 'wb'))
            pickle.dump(servg, open('./models/pretrain/servg.pkl', 'wb'))
        self.usergraph, self.servgraph = userg, servg
        self.dim = args.dimension
        self.order = args.order
        self.user_embeds = GraphSAGEConv(self.usergraph, args.dimension, args.order, args)
        self.item_embeds = GraphSAGEConv(self.servgraph, args.dimension, args.order, args)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2 * args.dimension, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

    def forward(self, userIdx, itemIdx):

        user_embeds = self.user_embeds(userIdx)
        serv_embeds = self.item_embeds(itemIdx)
        estimated = self.layers(torch.cat((user_embeds, serv_embeds), dim=-1)).sigmoid()
        estimated = estimated.reshape(user_embeds.shape[0])
        return estimated.flatten()

    def prepare_test_model(self):
        pass








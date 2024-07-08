# coding : utf-8
# Author : yuxiang Zeng
import dgl

# -*- coding: utf-8 -*-
# Author : yuxiang Zeng

import torch
import pickle

from dgl.nn.pytorch import SAGEConv
from models.baselines.GraphMF import create_graph


class GraphSAGEConv(torch.nn.Module):
    def __init__(self, graph, rank, order, args):
        super(GraphSAGEConv, self).__init__()
        self.args = args
        self.order = order
        self.graph = graph
        self.embedding = torch.nn.Parameter(torch.Tensor(self.graph.number_of_nodes(), rank))
        torch.nn.init.kaiming_normal_(self.embedding)
        self.graph.ndata['L0'] = self.embedding
        self.layers = torch.nn.ModuleList([SAGEConv(rank, rank, aggregator_type='gcn') for _ in range(order)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(rank) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ELU() for _ in range(order)])

    def forward(self, index):
        # grpah = self.graph.to(self.args.device)
        # 获得子图
        subgraph, all_idx = dgl.khop_in_subgraph(self.graph, index, 2)
        index_map = {int(original): i for i, original in enumerate(subgraph.ndata['_ID'].tolist())}

        subgraph_node_id = []
        for i in range(len(index)):
            subgraph_node_id.append(index_map[index[i].item()])
        subgraph_node_id = torch.tensor(subgraph_node_id)
        feats = subgraph.ndata['L0']
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(subgraph, feats).squeeze()
            feats = norm(feats)
            feats = act(feats)
            subgraph.ndata[f'L{i + 1}'] = feats
        embeds = subgraph.ndata[f'L{self.order}'][subgraph_node_id]
        return embeds



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
        self.rank = args.rank
        self.order = args.order
        self.user_embeds = GraphSAGEConv(self.usergraph, args.rank, args.order, args)
        self.item_embeds = GraphSAGEConv(self.servgraph, args.rank, args.order, args)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2 * self.rank, self.rank),
            torch.nn.LayerNorm(self.rank),
            torch.nn.ReLU(),
            torch.nn.Linear(self.rank, self.rank // 2),
            torch.nn.LayerNorm(self.rank // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.rank // 2, 1)
        )

    def forward(self, userIdx, itemIdx):
        user_embeds = self.user_embeds(userIdx)
        serv_embeds = self.item_embeds(itemIdx)
        estimated = self.layers(torch.cat((user_embeds, serv_embeds), dim=-1)).sigmoid()
        estimated = estimated.reshape(user_embeds.shape[0])
        return estimated.flatten()

    def prepare_test_model(self):
        pass








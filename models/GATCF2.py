# -*- coding: utf-8 -*-
# Author : yuxiang Zeng
import time

import dgl as d
import torch
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class GraphGATConv(torch.nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1, dropout=0.1):
        super(GraphGATConv, self).__init__()
        self.num_heads = num_heads
        self.layer = GATConv(in_dim, out_dim, num_heads=num_heads, feat_drop=dropout)
        self.norm = torch.nn.LayerNorm(out_dim * num_heads)
        self.act = torch.nn.ELU()

    def forward(self, graph, features):
        g, feats = graph, features
        feats = self.layer(g, feats).view(feats.size(0), -1)
        feats = self.norm(feats)
        feats = self.act(feats)
        feats = feats.view(feats.size(0), self.num_heads, -1)
        feats = torch.mean(feats, dim = 1)
        return feats

# 线性与图结构Add融合
class GATCF2(torch.nn.Module):
    def __init__(self, args):
        super(GATCF2, self).__init__()
        self.args = args
        try:
            userg = pickle.load(open('./models/pretrain/userg.pkl', 'rb'))
            servg = pickle.load(open('./models/pretrain/servg.pkl', 'rb'))
        except:
            user_lookup, serv_lookup, userg, servg = create_graph()
            pickle.dump(userg, open('./models/pretrain/userg.pkl', 'wb'))
            pickle.dump(servg, open('./models/pretrain/servg.pkl', 'wb'))
        self.usergraph, self.servgraph = userg.to(self.args.device), servg.to(self.args.device)
        self.dim = args.dimension
        self.user_embeds = torch.nn.Embedding(self.usergraph.number_of_nodes(), self.dim)
        torch.nn.init.kaiming_normal_(self.user_embeds.weight)
        self.serv_embeds = torch.nn.Embedding(self.servgraph.number_of_nodes(), self.dim)
        torch.nn.init.kaiming_normal_(self.serv_embeds.weight)

        self.user_attention = GraphGATConv(args.dimension, args.dimension, args.head_num, 0.10)
        self.serv_attention = GraphGATConv(args.dimension, args.dimension, args.head_num, 0.10)
        if self.args.agg == 'cat':
            input_dim = 4 * args.dimension
        else:
            input_dim = 2 * args.dimension

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LayerNorm(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )

        self.cache = {}

    def forward(self, userIdx, servIdx):
        Index = torch.arange(self.usergraph.number_of_nodes()).to(self.args.device)
        user_embeds = self.user_embeds(Index)
        Index = torch.arange(self.servgraph.number_of_nodes()).to(self.args.device)
        serv_embeds = self.serv_embeds(Index)
        if self.args.agg == 'add':
            user_embeds = self.user_attention(self.usergraph, user_embeds)[userIdx] + user_embeds[userIdx]
            serv_embeds = self.serv_attention(self.servgraph, serv_embeds)[servIdx] + serv_embeds[servIdx]
            estimated = self.layers(torch.cat((user_embeds, serv_embeds), dim=-1)).sigmoid().reshape(-1)
        elif self.args.agg == 'mean':
            user_embeds = torch.cat([self.user_attention(self.usergraph, user_embeds)[userIdx].unsqueeze(0), user_embeds[userIdx].unsqueeze(0)]).mean(0)
            serv_embeds = torch.cat([self.serv_attention(self.servgraph, serv_embeds)[servIdx].unsqueeze(0), serv_embeds[servIdx].unsqueeze(0)]).mean(0)
            # print(user_embeds.shape, serv_embeds.shape)
            estimated = self.layers(torch.cat((user_embeds, serv_embeds), dim=-1)).sigmoid().reshape(-1)
        elif self.args.agg == 'cat':
            user_embeds = torch.cat([self.user_attention(self.usergraph, user_embeds)[userIdx], user_embeds[userIdx]], dim=1)
            serv_embeds = torch.cat([self.serv_attention(self.servgraph, serv_embeds)[servIdx], serv_embeds[servIdx]], dim=1)
            # print(user_embeds.shape, serv_embeds.shape)
            estimated = self.layers(torch.cat((user_embeds, serv_embeds), dim=-1)).sigmoid().reshape(-1)

        return estimated




# Graph
def create_graph():
    userg = d.graph([])
    servg = d.graph([])
    user_lookup = FeatureLookup()
    serv_lookup = FeatureLookup()
    ufile = pd.read_csv('./datasets/userlist_table.csv')
    ufile = pd.DataFrame(ufile)
    ulines = ufile.to_numpy()
    ulines = ulines

    sfile = pd.read_csv('./datasets/wslist_table.csv')
    sfile = pd.DataFrame(sfile)
    slines = sfile.to_numpy()
    slines = slines

    for i in range(339):
        user_lookup.register('User', i)
    for j in range(5825):
        serv_lookup.register('Serv', j)

    for ure in ulines[:, 2]:
        user_lookup.register('URE', ure)
    for uas in ulines[:, 4]:
        user_lookup.register('UAS', uas)

    for sre in slines[:, 4]:
        serv_lookup.register('SRE', sre)
    for spr in slines[:, 2]:
        serv_lookup.register('SPR', spr)
    for sas in slines[:, 6]:
        serv_lookup.register('SAS', sas)

    userg.add_nodes(len(user_lookup))
    servg.add_nodes(len(serv_lookup))

    for line in ulines:
        uid = line[0]
        ure = user_lookup.query_id(line[2])
        if not userg.has_edges_between(uid, ure):
            userg.add_edges(uid, ure)

        uas = user_lookup.query_id(line[4])
        if not userg.has_edges_between(uid, uas):
            userg.add_edges(uid, uas)

    for line in slines:
        sid = line[0]
        sre = serv_lookup.query_id(line[4])
        if not servg.has_edges_between(sid, sre):
            servg.add_edges(sid, sre)

        sas = serv_lookup.query_id(line[6])
        if not servg.has_edges_between(sid, sas):
            servg.add_edges(sid, sas)

        spr = serv_lookup.query_id(line[2])
        if not servg.has_edges_between(sid, spr):
            servg.add_edges(sid, spr)

    userg = d.add_self_loop(userg)
    userg = d.to_bidirected(userg)
    servg = d.add_self_loop(servg)
    servg = d.to_bidirected(servg)
    return user_lookup, serv_lookup, userg, servg


class FeatureLookup:
    def __init__(self):
        self.__inner_id_counter = 0
        self.__inner_bag = {}
        self.__category = set()
        self.__category_bags = {}
        self.__inverse_map = {}

    def register(self, category, value):
        self.__category.add(category)
        if category not in self.__category_bags:
            self.__category_bags[category] = {}
        if value not in self.__inner_bag:
            self.__inner_bag[value] = self.__inner_id_counter
            self.__inverse_map[self.__inner_id_counter] = value
            if value not in self.__category_bags[category]:
                self.__category_bags[category][value] = self.__inner_id_counter
            self.__inner_id_counter += 1

    def query_id(self, value):
        return self.__inner_bag[value]

    def query_value(self, id):
        return self.__inverse_map[id]

    def __len__(self):
        return len(self.__inner_bag)



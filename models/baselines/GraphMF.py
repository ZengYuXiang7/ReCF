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
        grpah = self.graph.to(self.args.device)
        feats = grpah.ndata['L0']
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(grpah, feats).squeeze()
            feats = norm(feats)
            feats = act(feats)
            grpah.ndata[f'L{i + 1}'] = feats
        embeds = grpah.ndata[f'L{self.order}'][index]
        return embeds


class GraphMF(torch.nn.Module):
    def __init__(self, args):
        self.args = args
        super(GraphMF, self).__init__()
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




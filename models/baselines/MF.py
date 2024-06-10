# coding : utf-8
# Author : yuxiang Zeng

import torch
import numpy as np
import pandas as pd
import pickle as pk


class PureMF(torch.nn.Module):
    def __init__(self, args):
        super(PureMF, self).__init__()
        self.embed_user_GMF = torch.nn.Embedding(339, args.dimension)
        self.embed_item_GMF = torch.nn.Embedding(5825, args.dimension)
        self.predict_layer = torch.nn.Linear(args.dimension, 1)

    def forward(self, UserIdx, itemIdx):
        user_embed = self.embed_user_GMF(UserIdx)
        item_embed = self.embed_item_GMF(itemIdx)
        gmf_output = user_embed * item_embed
        prediction = gmf_output.sum(dim=-1)
        return prediction.flatten()

    def prepare_test_model(self):
        pass
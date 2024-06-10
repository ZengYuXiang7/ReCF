import faiss
import numpy as np
import torch


class ErrorCompensation:

    def __init__(self, args, default_lamda=0.5):
        self.args = args
        self.default_lamda = default_lamda
        self.clear()

    def append(self, hidden, target, predict):
        # hidden as tensors
        hidden = hidden.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        predict = predict.detach().cpu().numpy()

        self.hiddens.append(hidden)
        self.targets.append(target)
        self.predicts.append(predict.reshape(target.shape))

    def set_ready(self):

        # Filter out error that is small
        error = np.abs(np.concatenate(self.predicts) - np.concatenate(self.targets))
        percentile = np.percentile(error, 0)
        maskIdx = error > percentile

        self.ready_hiddens = np.vstack(self.hiddens)[maskIdx]
        self.ready_targets = np.concatenate(self.targets)[maskIdx]
        self.index.train(self.ready_hiddens)
        self.index.add(self.ready_hiddens)

    def clear(self):
        self.index = faiss.IndexLSH(self.args.dimension // 2, 64)
        self.hiddens = []
        self.targets = []
        self.predicts = []
        self.ready_hiddens = None
        self.ready_targets = None

    def correction(self, hidden):
        hidden = hidden.detach().cpu().numpy()
        dists, I = self.index.search(hidden, self.args.topk)
        compensation = np.zeros(len(hidden), dtype=np.float32)
        lmdas = np.zeros_like(compensation, dtype=np.float32)
        for i in range(len(lmdas)):
            lmdas[i] = self.default_lamda if dists[i][0] > 7 else self.default_lamda
            # quantile = 0.5 if dists[i][0] > 7 else 0.5

            temp = []
            for j in range(self.args.topk):
                temp.append(self.ready_targets[I[i][j]])
            compensation[i] = np.mean(temp)

            # compensation[i] = self.ready_targets[I[i][0]]

        return torch.from_numpy(compensation).to(self.args.device), torch.from_numpy(lmdas).to(self.args.device)

# coding : utf-8
# Author : yuxiang Zeng
import collections
import time

import numpy as np
import torch

from tqdm import *

from data import experiment, DataModule
from models.baselines.CSMF import CSMF
from models.baselines.GATCF import GATCF
from models.baselines.GATCF2 import GATCF2
from models.baselines.GraphMF import GraphMF
from models.baselines.MF import PureMF
from models.baselines.NeuCF import NeuCF
from models.baselines.Subgraph import SubgraphCF
from models.baselines.hyperCF_ccy import HTCF
from models.memory import ErrorCompensation
from utils.config import get_config
from utils.dataloader import get_dataloaders
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.plotter import MetricsPlotter
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import set_settings, set_seed, makedir

global log

torch.set_default_dtype(torch.float32)


class Model(torch.nn.Module):
    def __init__(self, train_graph, args):
        super(Model, self).__init__()
        self.args = args
        self.user_num = args.user_num
        self.serv_num = args.serv_num
        if self.args.model == 'GATCF':
            self.model = GATCF(args)
        else:
            raise NotImplementedError

        self.error_memory = ErrorCompensation(args)

    def forward(self, inputs):
        userIdx, servIdx = inputs
        hidden, estimated = self.model(userIdx, servIdx)
        return hidden, estimated.flatten()

    def setup_optimizer(self, args):
        self.to(args.device)
        self.loss_function = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, threshold=0.01)

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in tqdm(dataModule.train_loader):
            inputs, value = train_Batch
            inputs = inputs[0].to(self.args.device), inputs[1].to(self.args.device)
            value = value.to(self.args.device)
            hidden, pred = self.forward(inputs)
            if self.args.classification:
                loss = self.loss_function(pred, value.long())
            else:
                loss = self.loss_function(pred, value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)

        for train_Batch in (dataModule.train_loader):
            inputs, value = train_Batch
            inputs = inputs[0].to(self.args.device), inputs[1].to(self.args.device)
            value = value.to(self.args.device)
            hidden, pred = self.forward(inputs)
            # Error Compensation
            self.error_memory.append(hidden, value, pred)

        return loss, t2 - t1

    def valid_one_epoch(self, dataModule):
        val_loss = 0.
        reals = []
        preds = []
        wiecs = []
        for valid_Batch in tqdm(dataModule.valid_loader):
            inputs, value = valid_Batch
            inputs = inputs[0].to(self.args.device), inputs[1].to(self.args.device)
            value = value.to(self.args.device)
            hidden, pred = self.forward(inputs)
            val_loss += self.loss_function(pred, value)

            # Error Compensation
            queries = hidden
            compensates, lmdas = self.error_memory.correction(queries)
            corrected = lmdas * compensates + (1-lmdas) * pred
            wiecs.append(corrected)

            if self.args.classification:
                pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
            preds.append(pred)
            reals.append(value)
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        wiecs = torch.cat(wiecs, dim=0)
        self.scheduler.step(val_loss)
        noec_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value, self.args)
        wiec_error = ErrorMetrics(reals * dataModule.max_value, wiecs * dataModule.max_value, self.args)
        return noec_error, wiec_error

    def test_one_epoch(self, dataModule):
        preds = []
        reals = []
        wiecs = []
        for test_Batch in (dataModule.test_loader):
            inputs, value = test_Batch
            inputs = inputs[0].to(self.args.device), inputs[1].to(self.args.device)
            value = value.to(self.args.device)
            hidden, pred = self.forward(inputs)

            # Error Compensation
            queries = hidden
            compensates, lmdas = self.error_memory.correction(queries)
            corrected = lmdas * compensates + (1 - lmdas) * pred
            wiecs.append(corrected)

            if self.args.classification:
                pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
            preds.append(pred)
            reals.append(value)
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        wiecs = torch.cat(wiecs, dim=0)
        noec_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value, self.args)
        wiec_error = ErrorMetrics(reals * dataModule.max_value, wiecs * dataModule.max_value, self.args)
        return noec_error, wiec_error



def RunOnce(args, runId, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(datamodule, args)
    monitor = EarlyStopping(args)

    # Setup training tool
    model.setup_optimizer(args)
    train_time = []
    for epoch in range(args.epochs):
        model.error_memory.clear()
        epoch_loss, time_cost = model.train_one_epoch(datamodule)
        model.error_memory.set_ready()
        noec_error, wiec_error = model.valid_one_epoch(datamodule)
        monitor.track_one_epoch(epoch, model, wiec_error)
        train_time.append(time_cost)
        log.show_epoch_error(runId, epoch, monitor, epoch_loss, noec_error, train_time)
        log.show_epoch_error(runId, epoch, monitor, epoch_loss, wiec_error, train_time)
        plotter.append_epochs(wiec_error)
        if monitor.early_stop:
            break
    model.load_state_dict(monitor.best_model)
    sum_time = sum(train_time[: monitor.best_epoch])
    noec_error, wiec_error = model.test_one_epoch(datamodule)
    log.show_test_error(runId, monitor, wiec_error, sum_time)

    # Save the best model parameters
    makedir('./checkpoints')
    model_path = f'./checkpoints/{args.model}_{args.seed}.pt'
    torch.save(monitor.best_model, model_path)
    # log.only_print(f'Model parameters saved to {model_path}')
    return wiec_error


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(args.rounds):
        plotter.reset_round()
        results = RunOnce(args, runId, log)
        plotter.append_round()
        for key in results:
            metrics[key].append(results[key])
    log('*' * 20 + 'Experiment Results:' + '*' * 20)

    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')

    if args.record:
        log.save_result(metrics)
        plotter.record_metric(metrics)

    log('*' * 20 + 'Experiment Success' + '*' * 20)

    return metrics


if __name__ == '__main__':
    args = get_config()
    set_settings(args)

    # logger plotter
    exper_detail = f"Dataset : {args.dataset.upper()}, Model : {args.model}, Density : {args.density:.2f}"
    log_filename = f'd{args.density}_r{args.dimension}'
    log = Logger(log_filename, exper_detail, args)
    plotter = MetricsPlotter(log_filename, args)
    args.log = log
    log(str(args.__dict__))

    # Run Experiment
    RunExperiments(log, args)


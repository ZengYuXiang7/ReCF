# coding : utf-8
# Author : yuxiang Zeng
from dataclasses import dataclass


@dataclass
class LoggerConfig:
    logger: str = 'None'


@dataclass
class ExperimentConfig:
    seed: int = 0
    rounds: int = 1
    epochs: int = 100
    patience: int = 30

    verbose: int = 1
    device: str = 'mps'
    debug: bool = False
    experiment: bool = False
    program_test: bool = False
    record: bool = True



@dataclass
class BaseModelConfig:
    model: str = 'MF'
    rank: int = 40


@dataclass
class DatasetInfo:
    path: str = './datasets/'
    dataset: str = 'rt'
    train_size: int = 100
    density: float = 0.10
    user_num: int = 339
    serv_num: int = 5825


@dataclass
class TrainingConfig:
    bs: int = 256
    lr: float = 0.001
    decay: float = 0.0001
    loss_func: str = 'L1Loss'
    optim: str = 'AdamW'


@dataclass
class OtherConfig:
    classification: bool = False
    visualize: bool = True

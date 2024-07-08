# coding : utf-8
# Author : Yuxiang Zeng

from default_config import *
from dataclasses import dataclass



@dataclass
class MFConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mf'
    rank: int = 30



@dataclass
class NeuCFConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'neucf'
    rank: int = 30
    num_layers: int = 2

@dataclass
class CSMFConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'csmf'
    rank: int = 30



@dataclass
class GraphMFConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'graphmf'
    rank: int = 30
    order: int = 2


@dataclass
class GATCFConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'gatcf'
    rank: int = 30
    head_num: int = 2



from typing import Type, Dict, Any, List, Optional

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

from ...util import TrainingConfig
from .dataset import SoloData


class TrainingCombination:
    def __init__(self,
                 data_size: List[int],
                 add_direction: bool,
                 model_class: Type[nn.Module],
                 model_params: Dict[str, Any],
                 model_desc: str,
                 loss_fn_class: Type[nn.Module],
                 loss_fn_params: Dict[str, Any],
                 loss_fn_desc: str,
                 optimizer_class: Type[optim.Optimizer],
                 optimizer_params: Dict[str, Any],
                 optimizer_desc: str,
                 epochs: int,
                 batch_size: int,
                 scheduler_class: Optional[Type[LRScheduler]] = None,
                 scheduler_params: Optional[Dict[str, Any]] = None,
                 scheduler_desc: str = "",
                 other: str = ""):
        self.data_size = data_size
        self.add_direction = add_direction
        self.model_class = model_class
        self.model_params = model_params
        self.model_desc = model_desc
        self.loss_fn_class = loss_fn_class
        self.loss_fn_params = loss_fn_params
        self.loss_fn_desc = loss_fn_desc
        self.optimizer_class = optimizer_class
        self.optimizer_params = optimizer_params
        self.optimizer_desc = optimizer_desc
        self.scheduler_class = scheduler_class
        self.scheduler_params = scheduler_params
        self.scheduler_desc = scheduler_desc
        self.epochs = epochs
        self.batch_size = batch_size
        self.other = other

    def init_data(self, data: np.ndarray, split: float = 0.9):
        raw_train = data[:int(len(data) * split)]
        raw_test = data[int(len(data) * split):]

        train_dataset = SoloData(raw_train, self.data_size, self.add_direction)
        test_dataset = SoloData(raw_test, self.data_size, self.add_direction)

        return train_dataset, test_dataset

    def initialize_model(self) -> nn.Module:
        return self.model_class(**self.model_params)

    def initialize_loss_fn(self) -> nn.Module:
        return self.loss_fn_class(**self.loss_fn_params)

    def initialize_optimizer(self, model: nn.Module) -> optim.Optimizer:
        return self.optimizer_class(model.parameters(), **self.optimizer_params)

    def initialize_scheduler(self, optimizer: optim.Optimizer) -> Optional[LRScheduler]:
        if self.scheduler_class is not None:
            return self.scheduler_class(optimizer, **self.scheduler_params)
        return None

    def get_config(self) -> TrainingConfig:
        tc = TrainingConfig(
            data=f"{"add_dir" if self.add_direction else "dir"}/{self.data_size[0]}/{self.data_size[1]}",
            model=self.model_desc,
            loss_fn=self.loss_fn_desc,
            optimizer=self.optimizer_desc,
            scheduler=self.scheduler_desc,
            other="",
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.optimizer_params['lr'])

        return tc

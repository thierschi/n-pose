from typing import List

import numpy as np
from torch.utils.data import DataLoader

from .dataset import SoloData
from .metrics import ATE, AOE
from .trainer import Trainer
from .training_combination import TrainingCombination
from ...util import Logger, TrainingResult, EpochLogEntry


class CombinationTrainer:
    """
    Trainer that trains multiple models with different configurations and logs the results.
    """

    def __init__(self, configs: List[TrainingCombination], logger: Logger, data: np.ndarray, eval_data: np.ndarray):
        self.configs = configs
        self.logger = logger
        self.data = data
        self.eval_data = eval_data
        self.train_loss = 0.0
        self.train_metrics = [0.0, 0.0]
        self.value_count = 0

    def train_all(self):
        """
        Train all models with the given configurations.
        """
        for config in self.configs:
            # Initialize the model, loss function, optimizer, scheduler, and data
            model = config.initialize_model()
            loss_fn = config.initialize_loss_fn()
            optimizer = config.initialize_optimizer(model)
            scheduler = config.initialize_scheduler(optimizer)
            train_data, test_data = config.init_data(self.data)

            # Create data loaders
            train_data_loader = DataLoader(train_data, batch_size=config.batch_size)
            test_data_loader = DataLoader(test_data, batch_size=config.batch_size)

            # Initialize metrics
            metrics = [ATE(), AOE(config.add_direction)]
            self.train_loss = 0.0
            self.train_metrics = [0.0, 0.0]
            self.value_count = 0

            # Init training
            trainer = Trainer(model, train_data_loader, test_data_loader, metrics, self.log_epoch_callback)
            trainer.use_best_device()
            self.logger.start_training(config.get_config())

            # Train the model and save it
            trainer.train(config.epochs, loss_fn, optimizer, scheduler)
            trainer.save(f"{self.logger.get_folder_path()}/models/model_{self.logger.get_training_id()}.pth")

            # Evaluate the model
            eval_data = SoloData(self.eval_data, config.data_size, config.add_direction)
            eval_data_loader = DataLoader(eval_data, batch_size=config.batch_size)
            eval_loss, eval_metrics = trainer.evaluate_with_loader(eval_data_loader, loss_fn)

            train_loss = self.train_loss / self.value_count
            train_metrics = [self.train_metrics[0] / self.value_count, self.train_metrics[1] / self.value_count]

            # Log results
            result = TrainingResult(
                train_loss=train_loss,
                val_loss=eval_loss,
                train_ATE=train_metrics[0],
                train_AOE=train_metrics[1],
                val_ATE=eval_metrics[0],
                val_AOE=eval_metrics[1]
            )
            self.logger.complete_training(result)

    def log_epoch_callback(self, log_values):
        """
        Function that serves as a callback for the Trainer class to log the results of each epoch.
        """
        epoch, train_loss, val_loss, train_ate, train_aoe, val_ate, val_aoe = log_values
        entry = EpochLogEntry(epoch, train_loss, val_loss, train_ate, train_aoe, val_ate, val_aoe)
        self.logger.log_epoch(entry)
        self.train_loss += train_loss
        self.train_metrics[0] += train_ate
        self.train_metrics[1] += train_aoe
        self.value_count += 1

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    data: str
    model: str
    loss_fn: str
    optimizer: str
    scheduler: str
    other: str
    batch_size: int
    epochs: int
    learning_rate: float

    def to_csv_row(self):
        return ', '.join(map(str, vars(self).values()))

    @staticmethod
    def get_header():
        return ', '.join(TrainingConfig.__annotations__.keys())


@dataclass
class TrainingResult:
    train_loss: float
    val_loss: float
    train_ATE: float
    train_AOE: float
    val_ATE: float
    val_AOE: float

    def to_csv_row(self):
        return ', '.join(map(str, vars(self).values()))

    @staticmethod
    def get_header():
        return ', '.join(TrainingResult.__annotations__.keys())


@dataclass
class EpochLogEntry:
    epoch: int
    train_loss: float
    val_loss: float
    train_ATE: float
    train_AOE: float
    val_ATE: float
    val_AOE: float

    def to_csv_row(self):
        return ', '.join(map(str, vars(self).values()))

    @staticmethod
    def get_header():
        return ', '.join(EpochLogEntry.__annotations__.keys())


class Logger:
    _META_FILE_NAME = "training_meta.csv"
    _LOG_FILE_NAME = "training.csv"

    _folder_path: str
    _training_counter: int
    _current_training_id: Optional[int]
    _current_training_config: Optional[TrainingConfig]

    def __init__(self, folder_path: str):
        self._folder_path = folder_path
        self._training_counter = 0
        self._current_training_config = None

        self._init_folder()
        self._init_trainings_meta_file()
        self._init_training_log_file()

    def _init_folder(self):
        if not os.path.exists(self._folder_path):
            os.makedirs(self._folder_path)
        else:
            i = 1
            while os.path.exists(self._folder_path + '_' + str(i)):
                i += 1
            self._folder_path = self._folder_path + '_' + str(i)
            os.makedirs(self._folder_path)

    def _init_trainings_meta_file(self):
        header = "id, "
        header += TrainingConfig.get_header()
        header += ", "
        header += TrainingResult.get_header()

        with open(f"{self._folder_path}/{self._META_FILE_NAME}", "w") as f:
            f.write(header + "\n")

    def _init_training_log_file(self):
        header = "train_id, "
        header += EpochLogEntry.get_header()

        with open(f"{self._folder_path}/{self._LOG_FILE_NAME}", "w") as f:
            f.write(header + "\n")

    def start_training(self, config: TrainingConfig):
        assert self._current_training_config is None

        self._current_training_config = config
        self._current_training_id = self._training_counter
        self._training_counter += 1

    def complete_training(self, result: TrainingResult):
        assert self._current_training_config is not None

        with open(f"{self._folder_path}/{self._META_FILE_NAME}", "a") as f:
            f.write(str(self._current_training_id) + ", ")
            f.write(self._current_training_config.to_csv_row() + ", ")
            f.write(result.to_csv_row() + "\n")

        self._current_training_config = None

    def log_epoch(self, entry: EpochLogEntry):
        assert self._current_training_config is not None

        with open(f"{self._folder_path}/{self._LOG_FILE_NAME}", "a") as f:
            f.write(str(self._current_training_id) + ", ")
            f.write(entry.to_csv_row() + "\n")

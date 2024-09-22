import datetime
import os

import numpy as np

from src.models.n_pose.combination_trainer import CombinationTrainer
from src.models.n_pose.combinations import get_combinations
from src.util import Logger

# Change directory to the root of the project
path = os.getcwd()
while not os.path.exists(os.path.join(path, 'src')):
    path = os.path.join(path, "..")
os.chdir(path)

# Change these paths to the correct paths for your data
DATA_PATH_TRAIN_SET = 'data/solo_161_vec.csv'
DATA_PATH_VAL_SET = 'data/solo_159_vec.csv'
LOG_PATH = 'logs/n_pose'

# Get all combinations of models to train
combinations = get_combinations()

# Init Logger with current date as folder path
logger = Logger(f"{LOG_PATH}/{datetime.datetime.now().strftime('%Y-%m-%d')}-np", True)

# Load data to np arrays
data = np.loadtxt(DATA_PATH_TRAIN_SET, delimiter=",", skiprows=1)
eval_data = np.loadtxt(DATA_PATH_VAL_SET, delimiter=",", skiprows=1)

# Train all models
trainer = CombinationTrainer(combinations, logger, data, eval_data)
trainer.train_all()

import datetime
import os

import numpy as np

from src.models.n_pose.combination_trainer import CombinationTrainer
from src.models.n_pose.combinations import get_combinations
from src.util import Logger

path = os.getcwd()
while not os.path.exists(os.path.join(path, 'src')):
    path = os.path.join(path, "..")
os.chdir(path)

combinations = get_combinations()

# Today's date as yyyy-mm-dd
logger = Logger(f"logs/{datetime.datetime.now().strftime('%Y-%m-%d')}-np", True)

data = np.loadtxt("data/solo_161_vec.csv", delimiter=",", skiprows=1)
eval_data = np.loadtxt("data/solo_159_vec.csv", delimiter=",", skiprows=1)

trainer = CombinationTrainer(combinations, logger, data, eval_data)
trainer.train_all()

import os

from src.data.n_pose_conversion import unity_to_n_pose, TransformerBasedVectorConverter
from src.data.unity_data import UnityData

"""
This script can be used for converting the data to csv
"""

# Change directory to the root of the project
path = os.getcwd()
while not os.path.exists(os.path.join(path, 'src')):
    path = os.path.join(path, "..")
os.chdir(path)

# Change these paths to the correct paths for your data
DATA_PATH_TRAIN_SET = 'data/solo_161'
DATA_PATH_VAL_SET = 'data/solo_159'

# Convert val set
data = UnityData(DATA_PATH_VAL_SET)
path = unity_to_n_pose(data, f"{DATA_PATH_VAL_SET}_vec.csv", TransformerBasedVectorConverter)
print(f'Converted data to {path}')

# Convert train set
data = UnityData(DATA_PATH_TRAIN_SET)
path = unity_to_n_pose(data, f"{DATA_PATH_TRAIN_SET}_vec.csv", TransformerBasedVectorConverter)
print(f'Converted data to {path}')

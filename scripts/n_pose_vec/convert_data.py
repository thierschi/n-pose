import os

from src.data.n_pose_conversion import unity_to_n_pose, TransformerBasedVectorConverter
from src.data.unity_data import UnityData

path = os.getcwd()
while not os.path.exists(os.path.join(path, 'src')):
    path = os.path.join(path, "..")
os.chdir(path)

data = UnityData("data/solo_159")
path = unity_to_n_pose(data, "data/solo_159_vec.csv", TransformerBasedVectorConverter)
print(path)

data = UnityData("data/solo_161")
path = unity_to_n_pose(data, "data/solo_161_vec.csv", TransformerBasedVectorConverter)

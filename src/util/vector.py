import numpy as np


def normalize_vector(v):
    v = np.array(v)
    return v / np.linalg.norm(v)

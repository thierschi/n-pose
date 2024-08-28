import numpy as np


def normalize_vector(v, length=1):
    v = np.array(v)
    return v / np.linalg.norm(v) * length

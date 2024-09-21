import numpy as np


def normalize_vector(v, length=1) -> np.ndarray:
    """
    Normalize a vector to a given length
    :param v: The vector to normalize
    :param length: Length of the normalized vector
    :return: The normalized vector
    """
    v = np.array(v)
    return v / np.linalg.norm(v) * length

import numpy as np


def generate_random_split(n: int, val: float, test: float) -> np.ndarray:
    """
    Generate a random split of a given length
    :param n: Amount of samples
    :param val: Percentage of validation samples
    :param test: Percentage of test samples
    :return: A list coding the split as "train", "val" or "test"
    """
    n_val = int(n * val)
    n_test = int(n * test)
    n_train = n - n_val - n_test

    split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    np.random.shuffle(split)

    return split

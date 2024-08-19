import numpy as np


def generate_random_split(n: int, val: float, test: float):
    n_val = int(n * val)
    n_test = int(n * test)
    n_train = n - n_val - n_test

    split = np.array(["train"] * n_train + ["val"] * n_val + ["test"] * n_test)
    np.random.shuffle(split)

    return split

from math import floor

import torch.nn as nn

from .custom_fcn import CustomFCN


class SmallFCN(CustomFCN):
    def __init__(self, input_size: int, activation_fn=nn.ReLU, activation_params=None, inter_layer_module=None,
                 inter_layer_params=None, inter_layer_indices=None):
        layer_sizes = [input_size, input_size * 3, floor(input_size * 1.5), 6]
        super(SmallFCN, self).__init__(layer_sizes, activation_fn, activation_params, inter_layer_module,
                                       inter_layer_params, inter_layer_indices)

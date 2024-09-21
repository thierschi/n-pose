from typing import List

import torch.nn as nn


class CustomFCN(nn.Module):
    """
    Fully connected neural network with custom activation functions
    and interlayer modules as well as layer sizes.
    """

    def __init__(self, layer_sizes: List[int], activation_fn=nn.ReLU, activation_params=None, inter_layer_module=None,
                 inter_layer_params=None, inter_layer_indices=None):
        """
        Constructor for the CustomFCN class.
        :param layer_sizes: Array of integers representing the size of each layer.
        :param activation_fn: The activation function to use between layers.
        :param activation_params: The parameters to pass to the activation function.
        :param inter_layer_module: The interlayer module to use between layers. (e.g. dropout)
        :param inter_layer_params: The parameters to pass to the interlayer module.
        :param inter_layer_indices: The indices of the layers where the interlayer module should be applied.
        """
        super(CustomFCN, self).__init__()

        # Set default params for functions
        if activation_params is None:
            activation_params = {}
        if inter_layer_params is None:
            inter_layer_params = {}
        if inter_layer_indices is None:
            inter_layer_indices = range(len(layer_sizes) - 1)

        layers = []

        # Build model
        for i in range(1, len(layer_sizes)):
            l_in = layer_sizes[i - 1]
            l_out = layer_sizes[i]

            layers.append(nn.Linear(l_in, l_out))

            if i >= len(layer_sizes) - 1:  # No activation or dropout after the last layer
                continue

            layers.append(activation_fn(**activation_params))

            if inter_layer_module and (i - 1) in inter_layer_indices:
                layers.append(inter_layer_module(**inter_layer_params))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

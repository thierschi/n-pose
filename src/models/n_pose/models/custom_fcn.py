import torch.nn as nn


class CustomFCN(nn.Module):
    def __init__(self, layer_sizes, activation_fn=nn.ReLU, activation_params=None, inter_layer_module=None,
                 inter_layer_params=None, inter_layer_indices=None):
        super(CustomFCN, self).__init__()
        if activation_params is None:
            activation_params = {}
        if inter_layer_params is None:
            inter_layer_params = {}
        if inter_layer_indices is None:
            inter_layer_indices = range(len(layer_sizes) - 1)

        layers = []

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

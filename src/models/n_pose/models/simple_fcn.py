import torch.nn as nn


class SimpleFCN(nn.Module):
    def __init__(self, input_size, activation_fn=nn.ReLU, **activation_params):
        super(SimpleFCN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            activation_fn(**activation_params),
            nn.Linear(input_size // 2, 256),
            activation_fn(**activation_params),
            nn.Linear(256, 128),
            activation_fn(**activation_params),
            nn.Linear(128, 6)  # Output layer
        )

    def forward(self, x):
        return self.model(x)

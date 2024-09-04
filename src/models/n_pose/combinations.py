import torch
from torch import nn

from .custom_loss import RMSELoss, CombinationLoss, CosineSimilarityLoss, EuclideanLoss
from .models import CustomFCN
from .training_combination import TrainingCombination

BATCH_SIZE = 256
DATA_SIZE = [10, 100]
EPOCHS = [50, 100, 250]


def get_losses():
    losses = []
    losses.append((nn.MSELoss, {}, 'MSELoss'))
    # losses.append((nn.L1Loss, {}, 'L1Loss'))
    losses.append((RMSELoss, {}, 'RMSELoss'))

    losses.append(
        (CombinationLoss, {'l1': nn.MSELoss, 'l2': CosineSimilarityLoss}, 'CombinationLoss-MSE-Cos'))
    losses.append(
        (CombinationLoss, {'l1': EuclideanLoss, 'l2': CosineSimilarityLoss}, 'CombinationLoss-Eucl-Cos'))

    return losses


def get_activation_functions():
    fns = []
    fns.append((nn.ReLU, {}, 'ReLU'))
    fns.append((nn.ELU, {}, 'ELU'))

    return fns


def get_models(input_size):
    act_fns = get_activation_functions()

    layer_designs = []
    layer_designs.append(([input_size, input_size * 3, input_size * 3, input_size * 3, input_size * 3, 6], 'block-4hl'))
    layer_designs.append(([input_size, input_size * 3, input_size * 3, 6], 'block-2hl'))
    layer_designs.append(([input_size, input_size * 3, input_size * 2, input_size, 64, 6], 'slope-4hl'))
    layer_designs.append(([input_size, input_size * 3, (input_size * 3) // 2, 6], 'slope-2hl'))
    layer_designs.append(([input_size, input_size * 3, 6], '1hl'))
    layer_designs.append(([input_size, input_size // 2, input_size // 4, 6], 'slope-2hl-no-incr'))

    models = []

    for design in layer_designs:
        for fn in act_fns:
            models.append((CustomFCN, {'layer_sizes': design[0], 'activation_fn': fn[0], 'activation_params': fn[1]},
                           f"{design[1]}-{fn[2]}"))
            models.append((CustomFCN, {'layer_sizes': design[0], 'activation_fn': fn[0], 'activation_params': fn[1],
                                       'inter_layer_module': nn.Dropout, 'inter_layer_params': {'p': 0.5}},
                           f"{design[1]}-{fn[2]}-Dropout-0.5"))

    return models


def get_optimizers():
    optimizers = []
    optimizer_classes = [torch.optim.Adam, torch.optim.SGD]

    for optimizer_class in optimizer_classes:
        optimizers.append((optimizer_class, {'lr': 0.001}, f"{optimizer_class.__name__}-0.001"))

    return optimizers


def get_combinations():
    combinations = []

    for epoch in EPOCHS:
        losses = get_losses()
        models = get_models((DATA_SIZE[0] + DATA_SIZE[1]) * 2)
        optimizers = get_optimizers()

        for loss in losses:
            for model in models:
                for optimizer in optimizers:
                    combinations.append(
                        TrainingCombination(DATA_SIZE, False, model[0], model[1], model[2], loss[0], loss[1], loss[2],
                                            optimizer[0], optimizer[1], optimizer[2], epoch, BATCH_SIZE))
    return combinations

import torch
from torch import nn

from .custom_loss import RMSELoss, CombinationLoss, CosineSimilarityLoss, EuclideanLoss
from .models import *
from .training_combination import TrainingCombination


def get_losses():
    losses = []
    losses.append((nn.MSELoss, {}, 'MSELoss'))
    # losses.append((nn.L1Loss, {}, 'L1Loss'))
    losses.append((RMSELoss, {}, 'RMSELoss'))

    # losses.append(
    #     (CombinationLoss, {'l1': nn.MSELoss, 'l2': CosineSimilarityLoss}, 'CombinationLoss-MSE-Cos'))
    losses.append(
        (CombinationLoss, {'l1': EuclideanLoss, 'l2': CosineSimilarityLoss, 'w1': 2}, 'CombinationLoss-L1-Cos-2-1'))
    # losses.append(
    #     (CombinationLoss, {'l1': EuclideanLoss, 'l2': CosineSimilarityLoss, 'w2': 2}, 'CombinationLoss-L1-Cos1-2'))

    return losses


def get_activation_functions():
    fns = []
    fns.append((nn.ReLU, {}, 'ReLU'))
    # fns.append((nn.LeakyReLU, {'negative_slope': 0.01}, 'LeakyReLU-0.01'))

    return fns


def get_models(input_size):
    act_fns = get_activation_functions()

    models = []
    model_classes = [SmallFCN, SmallerFCN, DeepFCN]

    for fn in act_fns:
        for model_class in model_classes:
            models.append((model_class, {'input_size': input_size, 'activation_fn': fn[0], 'activation_params': fn[1]},
                           f"{model_class.__name__}-{fn[2]}"))
            models.append((model_class, {'input_size': input_size, 'activation_fn': fn[0], 'activation_params': fn[1],
                                         'inter_layer_module': nn.Dropout, 'inter_layer_params': {'p': 0.5},
                                         'inter_layer_indices': [0]},
                           f"{model_class.__name__}-{fn[2]}-Dropout-0.5"))

    return models


def get_optimizers():
    optimizers = []
    optimizer_classes = [torch.optim.Adam, torch.optim.SGD]
    learning_rates = [0.001, 0.0001]

    for optimizer_class in optimizer_classes:
        for lr in learning_rates:
            optimizers.append((optimizer_class, {'lr': lr}, f"{optimizer_class.__name__}-{lr}"))

    return optimizers


def get_schedulers():
    schedulers = []

    schedulers.append((torch.optim.lr_scheduler.StepLR, {'step_size': 10, 'gamma': 0.1}, 'StepLR-10-0.1'))
    schedulers.append((torch.optim.lr_scheduler.StepLR, {'step_size': 20, 'gamma': 0.1}, 'StepLR-20-0.1'))
    schedulers.append((torch.optim.lr_scheduler.StepLR, {'step_size': 5, 'gamma': 0.1}, 'StepLR-5-0.1'))

    return schedulers


def get_combinations():
    epochs = [50, 100, 250]
    batch_sizes = [32]
    data_sizes = [[10, 100]]
    combinations = []

    for epoch in epochs:
        for batch_size in batch_sizes:
            for data_size in data_sizes:
                losses = get_losses()
                models = get_models((data_size[0] + data_size[1]) * 2)
                optimizers = get_optimizers()

                for loss in losses:
                    for model in models:
                        for optimizer in optimizers:
                            combinations.append(
                                TrainingCombination(data_size, False, model[0], model[1], model[2], loss[0],
                                                    loss[1],
                                                    loss[2], optimizer[0], optimizer[1], optimizer[2], epoch,
                                                    batch_size))
    return combinations

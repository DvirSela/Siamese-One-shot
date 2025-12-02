from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.optim as optim


def get_loss_function() -> nn.Module:
    """
    Returns the loss function for the sanity check: Binary Cross Entropy Loss.
    Returns:
        nn.Module: Loss function
    """
    return nn.BCELoss()


def get_optimizer(model, lr=0.001, momentum=None, weight_decay=None) -> optim.Optimizer:
    """
    Returns the optimizer for the sanity check: Adam optimizer with no momentum or weight decay.
    Args:
        model: The model to optimize.
        lr: Learning rate.
        momentum: Initial momentum (paper starts at 0.5)
        weight_decay: This is the lambda parameter from the paper's loss equation.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer

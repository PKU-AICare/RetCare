import torch
import torch.nn.functional as F

from .multitask_loss import get_multitask_loss
from .time_aware_loss import get_time_aware_loss


def get_loss(y_pred, y_true, task, time_aware=False):
    if task == "outcome":
        loss = F.binary_cross_entropy(y_pred, y_true[:, 0]) if len(y_true.shape) > 1 else F.binary_cross_entropy(y_pred, y_true)
    elif task == "los":
        loss = F.mse_loss(y_pred, y_true[:, 1])
    elif task == "multitask":
        loss = get_multitask_loss(y_pred[:,0], y_pred[:,1], y_true[:,0], y_true[:,1])

    # If use time aware loss:
    if task == "outcome" and time_aware:
        loss = get_time_aware_loss(y_pred, y_true[:, 0], y_true[:, 1])

    return loss

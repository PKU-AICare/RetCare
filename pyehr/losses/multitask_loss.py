import torch
from torch import nn


class MultitaskLoss(nn.Module):
    def __init__(self, task_num=2):
        super(MultitaskLoss, self).__init__()
        self.task_num = task_num
        self.alpha = nn.Parameter(torch.ones((task_num)))
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, outcome_pred, los_pred, outcome, los):
        loss0 = self.bce(outcome_pred, outcome)
        loss1 = self.mse(los_pred, los)
        return loss0 * self.alpha[0] + loss1 * self.alpha[1]

def get_multitask_loss(outcome_pred, los_pred, outcome, los):
    mtl = MultitaskLoss(task_num=2)
    return mtl(outcome_pred, los_pred, outcome, los)
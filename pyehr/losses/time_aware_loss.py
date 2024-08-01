import torch
from torch import nn


class TimeAwareLoss(nn.Module):
    def __init__(self, decay_rate=0.1, reward_factor=0.1):
        super(TimeAwareLoss, self).__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.decay_rate = decay_rate
        self.reward_factor = reward_factor

    def forward(self, outcome_pred, outcome_true, los_true):
        los_weights = torch.exp(-self.decay_rate * los_true)  # Exponential decay
        loss_unreduced = self.bce(outcome_pred, outcome_true)

        reward_term = (los_true * torch.abs(outcome_true - outcome_pred)).mean()  # Reward term
        loss = (loss_unreduced * los_weights).mean()-self.reward_factor * reward_term  # Weighted loss
        
        return torch.clamp(loss, min=0)

def get_time_aware_loss(outcome_pred, outcome_true, los_true):
    time_aware_loss = TimeAwareLoss()
    return time_aware_loss(outcome_pred, outcome_true, los_true)

if __name__ == "__main__":
    outcome_pred = torch.tensor([0.1])
    outcome_true = torch.tensor([1.])
    los_true = torch.tensor([-4.0])
    print(get_time_aware_loss(outcome_pred, outcome_true, los_true))

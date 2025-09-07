# src/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Custom Focal Loss implementation, based on cross-entropy.
    This is suitable for multi-class classification problems.
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate the standard cross-entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Get the probability of the correct class
        pt = torch.exp(-ce_loss)
        # Calculate the focal loss
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
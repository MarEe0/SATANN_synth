"""Detection loss functions and classes."""
from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.ops as tvops

class IoULoss(nn.Module):
    def __init__(self):
        """Intersection over union loss"""
        super(IoULoss, self).__init__()
        self.smoothl1loss = torch.nn.SmoothL1Loss(reduction="mean", beta=1)

    def forward(self, output, target):
        """Output and target must be of shape (B,C,4)"""
        # Getting the boxes for each class
        ious = torch.empty((output.size(0), output.size(1)))
        for _class in range(output.size(1)):
            all_ious = tvops.box_iou(output[:,_class], target[:,_class])
            ious[:,_class] = all_ious.diag()

        
        return torch.mean(ious) + self.smoothl1loss(output,target)


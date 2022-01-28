"""Functions for computing metrics."""
import numpy as np
import torch
from skimage.measure import label

def dice_score(outputs, labels, _class):
    """Computes the Dice score of the given class"""
    axes = tuple(range(1,len(labels.shape)))
    tp = torch.sum((outputs == _class).float() * (labels == _class).float(), dim=axes)
    fp = torch.sum((outputs == _class).float() * (labels != _class).float(), dim=axes)
    fn = torch.sum((outputs != _class).float() * (labels == _class).float(), dim=axes)

    dice = (2 * tp) / (2 * tp + fp + fn)
    
    return dice

def count_connected_components(outputs, _class):
    """Counts how many connected components the output has for a given class"""
    masks = (outputs == _class).detach().cpu().numpy()
    counts = [label(mask, return_num=True)[1] for mask in masks]
    return counts


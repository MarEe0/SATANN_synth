"""Functions for computing metrics."""
import torch
from skimage.measure import label
import torchvision.ops as tvops

def dice_score(outputs, labels, _class):
    """Computes the Dice score of the given class"""
    axes = tuple(range(1,len(labels.shape)))
    tp = torch.sum((outputs == _class).float() * (labels == _class).float(), dim=axes)
    fp = torch.sum((outputs == _class).float() * (labels != _class).float(), dim=axes)
    fn = torch.sum((outputs != _class).float() * (labels == _class).float(), dim=axes)

    dice = (2 * tp) / (2 * tp + fp + fn)
    
    return dice

def precision(outputs, labels, _class):
    """Computes the precision of the given class"""
    axes = tuple(range(1,len(labels.shape)))
    tp = torch.sum((outputs == _class).float() * (labels == _class).float(), dim=axes)
    fp = torch.sum((outputs == _class).float() * (labels != _class).float(), dim=axes)

    precision = tp / (tp + fp)

    # replacing nans with 0
    precision = torch.nan_to_num(precision, nan=0)

    return precision

def recall(outputs, labels, _class):
    """Computes the recall of the given class"""
    axes = tuple(range(1,len(labels.shape)))
    tp = torch.sum((outputs == _class).float() * (labels == _class).float(), dim=axes)
    fn = torch.sum((outputs != _class).float() * (labels == _class).float(), dim=axes)

    recall = tp / (tp + fn)

    # replacing nans with 0
    recall = torch.nan_to_num(recall, nan=0)
    
    return recall

def count_connected_components(outputs, _class):
    """Counts how many connected components the output has for a given class"""
    masks = (outputs == _class).detach().cpu().numpy()
    counts = [label(mask, return_num=True)[1] for mask in masks]
    return counts

def jaccard(outputs, targets, _class):
    """Computes the IoU of pairs of bounding boxes of the given class"""
    # Converting to xyxy format
    outputs = tvops.box_convert(outputs[:, _class], "cxcywh", "xyxy")
    targets = tvops.box_convert(targets[:, _class], "cxcywh", "xyxy")

    jaccards = torch.empty(outputs.size(0))
    for i, (output, target) in enumerate(zip(outputs, targets)):
        jaccards[i] = tvops.box_iou(output[None,:], target[None,:])[0]

    return jaccards
    

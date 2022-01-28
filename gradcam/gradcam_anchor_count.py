# """Script for applying GradCAM to trained models."""
from collections import deque
import copy

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision as tv
from torch.nn.functional import softmax

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from unet import UNet
from datasets.clostob.clostob_dataset import CloStObDataset
from utils import targetToTensor, mkdir

# Needed for GradCAM with semantic segmentation
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()


if __name__ == "__main__":
    dataset_size = 400

    path_base = "/home/mriva/Recherche/PhD/SATANN/synthetic/results/dataset_{}".format(dataset_size)
    experiment_labels = ["T_easy_noise", "T_hard_noise", "T_veryhard_noise"]
    model_seeds = range(2)
    dataset_seeds = range(2)
    #alphas = [0, 0.5]
    alphas = [0]
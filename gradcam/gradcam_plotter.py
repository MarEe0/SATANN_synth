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

experiment = "T_hard_noise_m1_d0_a0"  # If not using veryhard, fix the test dataset generator

# Needed for GradCAM with semantic segmentation
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()

# Getting CUDA
device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset size
dataset_size = 400

# Loading the model specified
model_path = "/home/mriva/Recherche/PhD/SATANN/synthetic/results/dataset_{}/{}/best_model.pth".format(dataset_size, experiment)
model = UNet(input_channels=1, output_channels=4).to(device=device)
model.load_state_dict(torch.load(model_path))
model.eval()

# Getting the test images
test_set_size = 30

# Preparing the foreground
fg_label = "T"
fg_classes = [0, 1, 8]
base_fg_positions = [(0.65, 0.3), (0.65, 0.7), (0.35, 0.7)]
position_translation=0.2
position_noise=0.1

# Also setting the image dimensions in advance
image_dimensions = [160, 160]

transform = tv.transforms.Compose(                                  # For the images:
    [tv.transforms.ToTensor(),                                      # Convert to torch.Tensor CxHxW type
        tv.transforms.Normalize((255/2,), (255/2,), inplace=True)])    # Normalize from [0,255] to [-1,1] range
target_transform = tv.transforms.Compose(                           # For the labelmaps:
    [targetToTensor()])                                             # Convert to torch.Tensor type

# Preparing the rotated part of the test set
rotated_fg_positions = deque(base_fg_positions)
rotated_fg_positions.rotate(1)
# Preparing the swap part of the test set
swap_fg_positions = copy.copy(base_fg_positions)
swap_fg_positions[1], swap_fg_positions[2] = swap_fg_positions[2], swap_fg_positions[1]
# Preparing the distant part of the test set
dist_fg_position = copy.copy(base_fg_positions)
dist_fg_position[0] = (dist_fg_position[0][0], dist_fg_position[0][1]-0.1)
dist_fg_position[1] = (dist_fg_position[1][0], dist_fg_position[1][1]+0.1)
dist_fg_position[2] = (dist_fg_position[2][0], dist_fg_position[2][1]+0.1)

test_dataset = torch.utils.data.ConcatDataset([
                        CloStObDataset(base_dataset_name="fashion",
                                        image_dimensions=image_dimensions,
                                        size=test_set_size,
                                        fg_classes=fg_classes,
                                        fg_positions=fg_positions,
                                        position_translation=position_translation,
                                        position_noise=position_noise,
                                        bg_classes=[0], # Background class from config
                                        bg_amount=3,
                                        flattened=False,
                                        lazy_load=False,
                                        transform=transform,
                                        target_transform=target_transform,
                                        start_seed=dataset_size)
                        for fg_positions in [base_fg_positions, rotated_fg_positions, swap_fg_positions, dist_fg_position]
                    ])

# Running model on test dataset
for i, item_pair in enumerate(test_dataset):
    print("Processing image {}".format(i))
    input_tensor = item_pair["image"].unsqueeze(0).to(device)
    target = item_pair["labelmap"]

    output = model(input_tensor)

    output_softmax = softmax(output, dim=1)  # Softmax output along class dimension
    output_argmax = output_softmax.argmax(dim=1)  # Argmax output along class dimension
    output_argmax = output_argmax.detach().cpu().numpy()

    rgb_image = ((np.repeat(input_tensor.detach().cpu().numpy().squeeze()[...,None],3,axis=2) + 1) / 2).astype(np.float32)

    # Preparing the figure
    plt.figure(figsize=[36, 6])
    # First subplot: base image
    plt.subplot(1, 6, 1)
    plt.imshow(rgb_image)
    plt.title("Image")
    plt.axis("off")

    # Plotting the labelmap/target
    plt.subplot(1, 6, 2)
    plt.imshow(target, cmap="tab10", vmax=9)
    plt.title("Target")
    plt.axis("off")

    # Plotting the output
    plt.subplot(1, 6, 3)
    plt.imshow(output_argmax[0], cmap="tab10", vmax=9)
    plt.title("Output")
    plt.axis("off")

    # Getting all classes
    for target_category in [1,2,3]:
        # Doing GradCAM magic
        target_layers = [model.dconv_down4, model.dconv_up3, model.dconv_up2, model.dconv_up1, model.conv_last]
        #target_layers = [model.dconv_up1, model.conv_last]
        target_mask = np.float32(output_argmax == target_category)
        targets = [SemanticSegmentationTarget(target_category, target_mask)]

        with GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
            
            cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)

            # Plotting the CAM
            plt.subplot(1,6,3+target_category)
            plt.imshow(cam_image)
            plt.title("CAM for Class {}".format(target_category))
            plt.axis("off")

    plt.tight_layout(pad=1.5)

    # Saving the image
    save_folder = "/home/mriva/Recherche/PhD/SATANN/synthetic/gradcam/gradcam_images/{}".format(experiment)
    mkdir(save_folder)
    
    plt.savefig(os.path.join(save_folder, "cam_test{}.png".format(i)))
    plt.clf()
    plt.close()
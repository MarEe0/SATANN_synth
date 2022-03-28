"""Script for training networks with different structure sizes.

Author
------
 * Mateus Riva (mateus.riva@telecom-paris.fr)
"""
import time
import os
import json
import copy
import sys
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import softmax, cross_entropy
import torchvision as tv

sys.path.append("/home/mriva/Recherche/PhD/SATANN/SATANN_synth")
from train import train_model
from unet import UNet
from utils import targetToTensor, mkdir, plot_output, multi_logical_or
from metrics import dice_score, count_connected_components
from datasets.clostob.clostob_dataset import CloStObDataset
from spatial_loss import SpatialPriorErrorSegmentation

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.segmentation import mark_boundaries
def show_sample_with_receptive_field(sample):
    """Function for plotting a singular sample with the receptive fields overlayed"""
    image = (sample["image"][0].numpy() + 1.0)/2.0
    image = mark_boundaries(image, sample["labelmap"]==1, color=plt.get_cmap("tab10")(1)[:3], mode="thick", background_label=0)
    plt.imshow(image)  # Plotting the image

    # Getting center of OI structure
    min_struct_center_x = min(torch.nonzero(sample["labelmap"], as_tuple=True)[0])
    max_struct_center_x = max(torch.nonzero(sample["labelmap"], as_tuple=True)[0])
    min_struct_center_y = min(torch.nonzero(sample["labelmap"], as_tuple=True)[1])
    max_struct_center_y = max(torch.nonzero(sample["labelmap"], as_tuple=True)[1])
    struct_center = (min_struct_center_y + torch.div((max_struct_center_y-min_struct_center_y),2,rounding_mode="floor"),
                    min_struct_center_x + torch.div((max_struct_center_x-min_struct_center_x),2,rounding_mode="floor"))
    
    # Getting center of shirt
    min_oi_center_x = min(torch.nonzero(sample["labelmap"]==1, as_tuple=True)[0])
    max_oi_center_x = max(torch.nonzero(sample["labelmap"]==1, as_tuple=True)[0])
    min_oi_center_y = min(torch.nonzero(sample["labelmap"]==1, as_tuple=True)[1])
    max_oi_center_y = max(torch.nonzero(sample["labelmap"]==1, as_tuple=True)[1])
    oi_center = (min_oi_center_y + torch.div((max_oi_center_y-min_oi_center_y),2,rounding_mode="floor"),
                    min_oi_center_x + torch.div((max_oi_center_x-min_oi_center_x),2,rounding_mode="floor"))

    # Drawing receptive fields
    bottleneck_rf_size = (61,61)
    output_rf_size = (101,101)

    struct_bottleneck_rf = patches.Rectangle(np.array(struct_center) - (np.array(bottleneck_rf_size)//2), *bottleneck_rf_size,
                                            linewidth=1, edgecolor=(1.0,0.0,0.0), facecolor="none")
    plt.gca().add_patch(struct_bottleneck_rf)
    struct_output_rf = patches.Rectangle(np.array(struct_center) - (np.array(output_rf_size)//2), *output_rf_size,
                                            linewidth=1, edgecolor=(0.0,0.0,1.0), facecolor="none")
    plt.gca().add_patch(struct_output_rf)
    
    oi_bottleneck_rf = patches.Rectangle(np.array(oi_center) - (np.array(bottleneck_rf_size)//2), *bottleneck_rf_size,
                                            linewidth=1, edgecolor=(1.0,0.0,1.0), ls="--", facecolor="none")
    plt.gca().add_patch(oi_bottleneck_rf)
    oi_output_rf = patches.Rectangle(np.array(oi_center) - (np.array(output_rf_size)//2), *output_rf_size,
                                            linewidth=1, edgecolor=(0.0,1.0,1.0), ls="--", facecolor="none")
    plt.gca().add_patch(oi_output_rf)

    plt.axis("off")


class limitedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """Applies torch.nn.CrossEntropyLoss() to a specific subset of classes only."""
    def __init__(self, crit_classes = None, weight = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super(limitedCrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.crit_classes = crit_classes
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input, target):
        if self.crit_classes is not None:
            input = input[:,self.crit_classes]
            target_mask = multi_logical_or([target == _class for _class in crit_classes])
            target = torch.where(target_mask, target, 0)
        return cross_entropy(input, target, weight=self.weight,
                             ignore_index=self.ignore_index, reduction=self.reduction,
                             label_smoothing=self.label_smoothing)


def run_experiment(model_seed, dataset_split_seed, dataset, test_dataset, relational_criterion, alpha, crit_classes=None, deterministic=False, max_val_set_size=3000, experiment_label=None):
    results_path = "results/results_size"

    # Default training label: timestamp of start of training
    if experiment_label is None:
        experiment_label = time.strftime("%Y%m%d-%H%M%S")
    print("Starting segmentation experiment {} with model_seed={}, dataset_split_seed={}, alpha={}".format(experiment_label, model_seed, dataset_split_seed, alpha))

    # Fixing torch random generator for the dataset splitter
    dataset_split_rng=torch.Generator().manual_seed(dataset_split_seed)

    # Fixing deterministic factors if asked
    if deterministic:
        print("WARNING: Training is set to deterministic and may have lower performance.")
        torch.backends.cudnn.benchmark = True
        #torch.use_deterministic_algorithms(True)

    learning_rate = 0.001   # Optimizer's learning rate
    momentum = 0.9          # SGD's momentum
    betas = (0.9, 0.999)    # Adam's betas
    eps = 1e-08             # Adam's epsilons
    weight_decay = 0        # Adam's weight decay

    batch_size = 4          # Batch size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No CUDA available, running on CPU.")

    # Initializing data loaders
    # Splitting dataset into train, val and test subsets
    dataset_size = len(dataset)
    # Limiting validation set size
    train_set_size, val_set_size = (dataset_size*7)//10, (dataset_size*3)//10
    if val_set_size > max_val_set_size:
        val_set_size = max_val_set_size
    test_set_size = dataset_size - (train_set_size + val_set_size)  # to discard
    train_set, val_set, _ = random_split(dataset, (train_set_size, val_set_size, test_set_size), generator=dataset_split_rng)
    test_set = test_dataset
    # Preparing dataloaders
    data_loaders = {"train": DataLoader(train_set, batch_size=batch_size, num_workers=2),
                    "val": DataLoader(val_set, batch_size=batch_size, num_workers=2),
                    "test": DataLoader(test_set, batch_size=batch_size, num_workers=2)}


    # Initializing model
    torch.manual_seed(model_seed)  # Fixing random weight initialization
    model = UNet(input_channels=1, output_channels=len(crit_classes))
    model = model.to(device=device)

    # Preparing optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # Preparing loss
    if crit_classes is None:
        criterion = torch.nn.CrossEntropyLoss()
    else:  # Only a few classes to be used in the criterion
        criterion = limitedCrossEntropyLoss(crit_classes)

    # Training
    model = train_model(model, optimizer, scheduler, criterion, relational_criterion, "labelmap", alpha, data_loaders,
                        max_epochs=100, metrics=["dice","cc"], clip_max_norm=1, training_label=experiment_label,
                        results_path=results_path)


if __name__ == "__main__":
    # Running experiments
    for dataset_size in [10000]:
        # Preparing the fixed foreground information
        fg_label = "T"
        fg_classes = [0, 1, 8]
        #base_fg_positions = [(0.65, 0.3), (0.65, 0.7), (0.35, 0.7)]
        #position_translation=0.5
        position_noise=0
        #bg_bboxes = (0.4, 0.0, 0.9, 0.5)

        # Also setting the image dimensions in advance
        image_dimensions = [256,256]
        object_dimensions = [28,28]  # This does not control object size, but is used as reference
        
        # Preparing the limited cross entropy targets
        crit_classes = [0,1]  # BG and first class (shirts)

        
        # Preparing dataset transforms:
        transform = tv.transforms.Compose(                                  # For the images:
            [tv.transforms.ToTensor(),                                      # Convert to torch.Tensor CxHxW type
            tv.transforms.Normalize((255/2,), (255/2,), inplace=True)])    # Normalize from [0,255] to [-1,1] range
        target_transform = tv.transforms.Compose(                           # For the labelmaps:
            [targetToTensor()])                                             # Convert to torch.Tensor type

        # Experiment configurations
        model_seeds = range(1)
        dataset_split_seeds = range(1)
        #alphas=[0, 0.2, 0.5, 0.7]
        alphas = [0.0]

        experimental_configs = [{"label": fg_label + "_strict_noise_incbf", "bg_classes": [0], "bg_amount": 3, "structure_size": [24,32,40]},
                                {"label": fg_label + "_strict_noise_inbf", "bg_classes": [0], "bg_amount": 3, "structure_size": [33,44,55]},
                                {"label": fg_label + "_strict_noise_outbf", "bg_classes": [0], "bg_amount": 3, "structure_size": [60,80,100]},
                                {"label": fg_label + "_strict_noise_outof", "bg_classes": [0], "bg_amount": 3, "structure_size": [84,112,140]}]
        
        # Running experiments
        for experimental_config in experimental_configs:
            # Computing the base fg position for the experimental config - HARDCODED ON TRIANGLE
            fractional_structure_size = np.repeat(np.array(experimental_config["structure_size"])[...,None],2,axis=-1) / image_dimensions
            bottom_left_position = (0.5+(fractional_structure_size[0][0]/2), 0.5-(fractional_structure_size[1][1]/2))
            bottom_right_position = (0.5+(fractional_structure_size[0][0]/2), 0.5+(fractional_structure_size[1][1]/2))
            top_right_position = (0.5-(fractional_structure_size[0][0]/2), 0.5+(fractional_structure_size[1][1]/2))
            base_fg_positions = [bottom_left_position, bottom_right_position, top_right_position]

            # Computing the maximal translational noise
            object_size_buffer = np.array(object_dimensions)/image_dimensions
            left_clearance = min(pos[1] for pos in base_fg_positions) - object_size_buffer[1]
            right_clearance = (1-max(pos[1] for pos in base_fg_positions)) - object_size_buffer[1]
            top_clearance = min(pos[0] for pos in base_fg_positions) - object_size_buffer[0]
            bottom_clearance = (1-max(pos[0] for pos in base_fg_positions)) - object_size_buffer[0]
            position_translation = min(left_clearance, right_clearance, top_clearance, bottom_clearance)*2

            # Computing the bg bboxes
            bg_bboxes = [base_fg_positions[0][0]-(position_translation/2), base_fg_positions[0][1]-(position_translation/2),
                         base_fg_positions[0][0]+(position_translation/2), base_fg_positions[0][1]+(position_translation/2)]

            # Computing the graph relations
            graph_relations =   [[1,2,base_fg_positions[1-1][0]-base_fg_positions[2-1][0],base_fg_positions[1-1][1]-base_fg_positions[2-1][1]],
                                [1,3,base_fg_positions[1-1][0]-base_fg_positions[3-1][0],base_fg_positions[1-1][1]-base_fg_positions[3-1][1]],
                                [2,3,base_fg_positions[2-1][0]-base_fg_positions[3-1][0],base_fg_positions[2-1][1]-base_fg_positions[3-1][1]]]
            relational_criterion = SpatialPriorErrorSegmentation(graph_relations, image_dimensions=image_dimensions,
                                                             num_classes=len(fg_classes), crit_classes=crit_classes)

            for model_seed in model_seeds:
                for dataset_split_seed in dataset_split_seeds:
                    for alpha in alphas:
                        # Label of experiment:
                        experiment_label = "{}_m{}_d{}_a{}".format(experimental_config["label"], model_seed, dataset_split_seed, alpha)

                        # Preparing train dataset
                        train_dataset = CloStObDataset(base_dataset_name="fashion",
                                                    image_dimensions=image_dimensions,
                                                    size=dataset_size,
                                                    fg_classes=fg_classes,
                                                    fg_positions=base_fg_positions,
                                                    position_translation=position_translation,
                                                    position_noise=position_noise,
                                                    bg_classes=experimental_config["bg_classes"], # Background class from config
                                                    bg_amount=experimental_config["bg_amount"],
                                                    bg_bboxes=bg_bboxes,
                                                    fine_segment=True,
                                                    flattened=False,
                                                    lazy_load=True,
                                                    transform=transform,
                                                    target_transform=target_transform)
            

                        for example_idx in range(4):
                            show_sample_with_receptive_field(train_dataset[example_idx])
                            plt.savefig("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/structure_size/examples/{}_{}.png".format(experiment_label, example_idx),bbox_inches="tight")
                            plt.clf()
                        # Run experiment
                        run_experiment(model_seed=model_seed, dataset_split_seed=dataset_split_seed,
                                    dataset=train_dataset, test_dataset=None,
                                    relational_criterion=relational_criterion, crit_classes=crit_classes, alpha=alpha,
                                    deterministic=True, experiment_label=os.path.join("dataset_{}".format(dataset_size),experiment_label))


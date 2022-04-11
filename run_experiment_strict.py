"""Scripts for running synthetic SATANN experiments using segmentation.

Author
------
 * Mateus Riva (mateus.riva@telecom-paris.fr)
"""
import time
import os
import json
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import cross_entropy
import torchvision as tv

from train import train_model
from unet import UNet
from utils import targetToTensor, multi_logical_or
from datasets.clostob.clostob_dataset import CloStObDataset
from spatial_loss import SpatialPriorErrorSegmentation


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


def run_experiment(model_seed, dataset_split_seed, dataset, relational_criterion, alpha, crit_classes=None, deterministic=False, max_val_set_size=3000, experiment_label=None):
    results_path = "results/results_strict"

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
    # Preparing dataloaders
    data_loaders = {"train": DataLoader(train_set, batch_size=batch_size, num_workers=2),
                    "val": DataLoader(val_set, batch_size=batch_size, num_workers=2)}


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
    # Testing experiments
    #dataset_size = 400
    for dataset_size in [10000]:
        # Preparing the foreground
        fg_label = "D"  # Change here for other configurations. It's ugly, I know.

        if fg_label == "T":  # Right triangle
            fg_classes = [0, 1, 8]
            base_fg_positions = [(0.65, 0.3), (0.65, 0.7), (0.35, 0.7)]
            position_translation=0.5
            position_noise=0
            bg_bboxes = (0.4, 0.0, 0.9, 0.5)
        elif fg_label == "D":
            fg_classes = [0, 1, 8, 9]
            base_fg_positions = [(0.5, 0.3), (0.7, 0.5), (0.5, 0.7), (0.3, 0.5)]
            position_translation=0.5
            position_noise=0
            bg_bboxes = (0.25, 0.0, 0.75, 0.55)
        elif fg_label == "H":
            fg_classes = [0, 1]
            base_fg_positions = [(0.5, 0.3), (0.5, 0.7)]
            position_translation=0.5
            position_noise=0
            bg_bboxes = (0.25, 0.0, 0.75, 0.55)
        else: raise ValueError("fg_label {} not recognized".format(fg_label))

        # Also setting the image dimensions in advance
        image_dimensions = [160,160]
        
        # Preparing the limited cross entropy targets
        crit_classes = [0,1]  # BG and first class (shirts)

        # Preparing the relations
        graph_relations = [[1, 2, 0, -0.4],
                        [1, 3, 0.3, -0.4],
                        [2, 3, 0.3, 0]]   
        relational_criterion = SpatialPriorErrorSegmentation(graph_relations, image_dimensions=image_dimensions,
                                                             num_classes=len(fg_classes), crit_classes=crit_classes)

        # Preparing dataset transforms:
        transform = tv.transforms.Compose(                                  # For the images:
            [tv.transforms.ToTensor(),                                      # Convert to torch.Tensor CxHxW type
            tv.transforms.Normalize((255/2,), (255/2,), inplace=True)])    # Normalize from [0,255] to [-1,1] range
        target_transform = tv.transforms.Compose(                           # For the labelmaps:
            [targetToTensor()])                                             # Convert to torch.Tensor type

        # Experiment configurations
        model_seeds = range(5)
        dataset_split_seeds = range(5)
        #alphas=[0, 0.2, 0.5, 0.7]
        alphas = [0]
        experimental_configs = [{"label": fg_label + "_strict_noise", "bg_classes": [0], "bg_amount": 3}]
        
        # Running experiments
        for experimental_config in experimental_configs:
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
            
                        # Run experiment
                        run_experiment(model_seed=model_seed, dataset_split_seed=dataset_split_seed,
                                    dataset=train_dataset, 
                                    relational_criterion=relational_criterion, crit_classes=crit_classes, alpha=alpha,
                                    deterministic=True, experiment_label=os.path.join("dataset_{}".format(dataset_size),experiment_label))
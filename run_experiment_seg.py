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
from torch.nn.functional import softmax
import torchvision as tv

from train import train_model
from unet import UNet
from utils import targetToTensor, mkdir, plot_output
from metrics import dice_score, count_connected_components

def run_experiment(model_seed, dataset_split_seed, dataset, test_dataset, relational_criterion, alpha, deterministic=False, experiment_label=None):
    results_path = "results/results_seg"

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
    train_set, val_set, _ = random_split(dataset, ((dataset_size*7)//10, (dataset_size*3)//10, 0), generator=dataset_split_rng)
    test_set = test_dataset
    # Preparing dataloaders
    data_loaders = {"train": DataLoader(train_set, batch_size=batch_size, num_workers=2),
                    "val": DataLoader(val_set, batch_size=batch_size, num_workers=2),
                    "test": DataLoader(test_set, batch_size=batch_size, num_workers=2)}


    # Initializing model
    torch.manual_seed(model_seed)  # Fixing random weight initialization
    model = UNet(input_channels=1, output_channels=dataset.number_of_classes)
    model = model.to(device=device)

    # Preparing optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    # Preparing loss
    criterion = torch.nn.CrossEntropyLoss()

    # Training
    model = train_model(model, optimizer, scheduler, criterion, relational_criterion, "labelmap", alpha, data_loaders,
                        max_epochs=100, metrics=["dice","cc"], clip_max_norm=1, training_label=experiment_label,
                        results_path=results_path)

    # Testing
    test_outputs, truths = [], []
    for item_pair in data_loaders["test"]:
        inputs = item_pair["image"].to(device)
        with torch.set_grad_enabled(False):
            test_outputs.append(model(inputs.to(device)).detach().cpu())
        truths.append(item_pair["labelmap"])
    
    # Saving test and metrics
    print("\n Test results:")
    test_path = os.path.join(results_path, experiment_label, "test")
    with torch.no_grad():
        test_outputs = torch.cat(test_outputs, dim=0).to(device)
        truths = torch.cat(truths, dim=0).to(device)
        all_test_images = torch.cat([item_pair["image"] for item_pair in data_loaders["test"]], dim=0)
        num_classes = test_outputs.shape[1]  # Get number of classes

        outputs_softmax = softmax(test_outputs, dim=1)  # Softmax outputs along class dimension
        outputs_argmax = outputs_softmax.argmax(dim=1)  # Argmax outputs along class dimension

        # Compute dices
        outputs_dices = [dice_score(outputs_argmax, truths, _class) for _class in range(num_classes)]

        # Compute relational scores
        outputs_relational_scores = relational_criterion.compute_metric(outputs_softmax)

        # Compute connected components per class
        outputs_connected_components = [count_connected_components(outputs_argmax, _class) for _class in range(num_classes)]

        # Print foreground dices
        mean_output_dices = torch.mean(torch.stack(outputs_dices), dim=1)
        print("Mean Dices: ", end="")
        for mean_output_dice in mean_output_dices[1:]:
            print("{:.4f}, ".format(mean_output_dice.item()), end="")
        print("")

        # Save validation metrics
        mkdir(test_path)
        test_metrics = { "all" : 
            [
                {
                    _class : {
                        "Dice" : outputs_dices[_class][test_index].item(),
                        "Relational Loss" : outputs_relational_scores[test_index].item(),
                        "Connected Components" : outputs_connected_components[_class][test_index]
                    } for _class in range(num_classes)
                } for test_index in range(test_outputs.shape[0])
            ],
            "mean": {
                _class : {
                    "Dice" : torch.mean(outputs_dices[_class]).item(),
                    "Relational Loss" : torch.mean(outputs_relational_scores).item(),
                    "Connected Components" : np.mean(outputs_connected_components[_class])
                } for _class in range(num_classes)
            }
        }
        with open(os.path.join(test_path, "summary.json"), 'w') as f:
            json.dump(test_metrics, f, sort_keys=True, indent=4)


        # Save test
        for i, (test_image, test_targets, test_outputs) in enumerate(zip(all_test_images, 
                                                                        truths, 
                                                                        outputs_softmax)):
            plot_output(test_image, test_targets, test_outputs, os.path.join(test_path, "test{}.png".format(i)))



from datasets.clostob.clostob_dataset import CloStObDataset
from spatial_loss import GraphSpatialLoss
from collections import deque

if __name__ == "__main__":
    # Testing experiments
    dataset_size = 20
    test_set_size = 30

    # Preparing the foreground
    fg_label = "T"
    fg_classes = [0, 1, 8]
    base_fg_positions = [(0.65, 0.3), (0.65, 0.7), (0.35, 0.7)]
    position_translation=0.2
    position_noise=0.1

    # Also setting the image dimensions in advance
    image_dimensions = [256, 256]
    
    # Preparing the relations
    graph_relations = [[1, 2, 0, -0.4],
                       [1, 3, 0.3, -0.4],
                       [2, 3, 0.3, 0]]   
    relational_criterion = GraphSpatialLoss(graph_relations, image_dimensions=image_dimensions, num_classes=len(fg_classes))

    # Preparing dataset transforms:
    transform = tv.transforms.Compose(                                  # For the images:
        [tv.transforms.ToTensor(),                                      # Convert to torch.Tensor CxHxW type
         tv.transforms.Normalize((255/2,), (255/2,), inplace=True)])    # Normalize from [0,255] to [-1,1] range
    target_transform = tv.transforms.Compose(                           # For the labelmaps:
        [targetToTensor()])                                             # Convert to torch.Tensor type

    # Experiment configurations
    model_seeds = range(2)
    dataset_split_seeds = range(2)
    #alphas=[0, 0.2, 0.5, 0.7]
    alphas = [0, 0.5]
    experimental_configs = [{"label": fg_label + "_easy_noise", "bg_classes": [7], "bg_amount": 3},
                            {"label": fg_label + "_hard_noise", "bg_classes": [0], "bg_amount": 3},
                            {"label": fg_label + "_veryhard_noise", "bg_classes": [0,1,8], "bg_amount": 6}]
    
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
                                                flattened=False,
                                                lazy_load=False,
                                                transform=transform,
                                                target_transform=target_transform)
                    
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
                    # Preparing the size part of the test set
                    # TODO
                    # Preparing the affine-transform of the test set
                    # TODO
                    # Preparing test dataset
                    test_dataset = torch.utils.data.ConcatDataset([
                        CloStObDataset(base_dataset_name="fashion",
                                        image_dimensions=image_dimensions,
                                        size=test_set_size,
                                        fg_classes=fg_classes,
                                        fg_positions=fg_positions,
                                        position_translation=position_translation,
                                        position_noise=position_noise,
                                        bg_classes=experimental_config["bg_classes"], # Background class from config
                                        bg_amount=experimental_config["bg_amount"],
                                        flattened=False,
                                        lazy_load=False,
                                        transform=transform,
                                        target_transform=target_transform,
                                        start_seed=dataset_size)
                        for fg_positions in [base_fg_positions, rotated_fg_positions, swap_fg_positions, dist_fg_position]
                    ])
        
                    # Run experiment
                    run_experiment(model_seed=model_seed, dataset_split_seed=dataset_split_seed,
                                dataset=train_dataset, test_dataset=test_dataset,
                                relational_criterion=relational_criterion, alpha=alpha,
                                deterministic=True, experiment_label=os.path.join("dataset_{}".format(dataset_size),experiment_label))
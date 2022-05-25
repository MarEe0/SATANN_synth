"""Trains a U-Net for a given database and loss function.

Authors
-------
 * Mateus Riva (mateus.riva@telecom-paris.fr)
"""
import os
from copy import deepcopy
import re
import time
import json
from itertools import chain

import tqdm
import numpy as np
import torch
from torch.nn.functional import softmax

from metrics import dice_score, count_connected_components, jaccard, precision, recall
from utils import mkdir, plot_output, plot_output_det

def train_model(model, optimizer, scheduler, criterion, relational_criterions, relational_loss_criterion_idx, target_key, alpha, data_loaders, metrics=None, max_epochs=100, loss_strength=1, clip_max_norm=0, training_label=None, results_path=None, vals_to_plot=5):
    """Trains a neural network model until specified criteria are met.

    This function is a generic PyTorch NN training loop.

    Parameters
    ----------
    model : `torch.nn.Module`
        Network model to train.
    optimizer : `torch.optim.Optimizer`
        Optimizer to be used for training.
    scheduler : `torch.optim.Optimizer`
        Scheduler to be used for training.
    criterion : torch.nn.Loss
        Loss function class to be used for training.
    relational_criterions : list of `spatial_loss` objects
        The relational loss objects (needed even for the baseline for computing metrics).
    relational_loss_criterion_idx : `int` or list of `int`
        Which of the `relational_criterions` to use to compute relational loss.
    target_key : `str`, one of `"labelmap"`, `"bboxes"`
        The key to the wanted target in the data dictionary. `"labelmap"` is for segmentation,
        while `"bboxes"` is for detection.
    alpha : `float`
        The weight of the relational loss. The weight of the criterion is `(1-alpha)`.
    data_loaders : `dict`
        Dictionary containing the 'train' and 'val' `DataLoader`s, keyed to those names.
    metrics : `list`
        List containing the metric functions to be computed. Available values: "dice", "cc", "iou"
    max_epochs : `int`
        Maximum number of training epochs.
    clip_max_norm : `int`
        If greater than zero, the norm under which gradients will be clipped.
    training_label : `str` or _None_
        Label of this training, for saving results. If None, get current timestamp.
    vals_to_plot : `int`
        How many validation images to save per epoch.

    Returns
    -------
    trained_model
        The trained network model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If relational_criterion_loss_idx is an int, make it a single item list
    if type(relational_loss_criterion_idx) == int:
        relational_loss_criterion_idx = [relational_loss_criterion_idx]

    # Default training label: timestamp of start of training
    if training_label is None:
        training_label = time.strftime("%Y%m%d-%H%M%S")
    # Default storage path: current folder
    if results_path is None:
        results_path = "."
    
    # Variables for storing the best model and loss
    best_loss = float("inf")
    best_model_weights = deepcopy(model.state_dict())

    # Path for storing model and results
    model_training_path = os.path.join(results_path, training_label)
    validation_path = os.path.join(model_training_path, "val")
    mkdir(model_training_path)
    mkdir(validation_path)

    # Computing number of classes in the model's output
    num_classes = model.output_channels
    # Computing number of validation examples
    validation_count = len(data_loaders["val"].dataset)
    # Computing number of relational criterions
    num_relational = len(relational_criterions)

    for epoch in range(max_epochs):
        print("\nAt epoch {}".format(epoch))

        # Dict for storing loss per phase
        phase_losses = {}

        # Lists for storing validation metrics
        images_to_plot, targets_to_plot, outputs_to_plot = [], [], []                         # Visual results
        if "dice" in metrics:                                                                 # Segmentation metrics
            outputs_dices = [[] for _ in range(num_classes)]     
            outputs_precisions = [[] for _ in range(num_classes)]     
            outputs_recalls = [[] for _ in range(num_classes)]                
        if "cc" in metrics: outputs_connected_components = [[] for _ in range(num_classes)]   # Segmentation metrics
        if "iou" in metrics: outputs_ious = [[] for _ in range(num_classes)]                  # Detection metrics
        outputs_relational_scores = [[] for _ in range(num_relational)]                       # Relational metrics

        for phase in ["train","val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            if phase == "val":
                model.eval()  # Set model to evaluation mode
            
            # Storing losses to compute average epoch loss
            running_loss = 0
            running_crit_loss = 0
            running_rel_loss = 0
            running_batches = 0

            # Iterating over all items
            items_pbar = tqdm.tqdm(data_loaders[phase], total=len(data_loaders[phase]),
                                   desc="Epoch {} - {}".format(epoch, phase),
                                   postfix={"loss": float("inf")})
            for item_pair in items_pbar:
                images = item_pair["image"].to(device)
                targets = item_pair[target_key].to(device)

                # Zeroing gradients for a new minibatch
                optimizer.zero_grad()

                # Disabling gradients if we are in val phase
                with torch.set_grad_enabled(phase == "train"):
                    # Forward
                    outputs = model(images)
                    outputs_softmax = softmax(outputs, dim=1)       # softmax is used for relational loss, metric
                    if phase == "val":
                        outputs_argmax = outputs_softmax.argmax(dim=1)  # argmax is used for metrics
                    # Losses
                    if alpha < 1:
                        crit_loss = criterion(outputs, targets)  # Most criterions (like cross entropy) expect raw outputs
                    else:
                        crit_loss = torch.tensor(0)
                    if alpha > 0:
                        # Relational losses expect softmaxed outputs
                        rel_loss = torch.sum([relational_criterions[crit_idx](outputs_softmax, targets) for crit_idx in relational_loss_criterion_idx])
                    else:
                        rel_loss = torch.tensor(0)
                    loss = ((1-alpha)*crit_loss + (alpha)*rel_loss) * loss_strength
                    # Backward (only in training phase)
                    if phase == "train":
                        loss.backward()

                        if clip_max_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)

                        optimizer.step()
                    # Stepping scheduler (only in validation phase)
                    #if phase == "val":
                    #    scheduler.step(loss)
                    
                    # Accumulate running loss and batches
                    running_loss += loss.item()
                    running_crit_loss += crit_loss.item()
                    running_rel_loss += rel_loss.item()
                    running_batches += 1
                    phase_losses[phase] = {
                        "all": running_loss/running_batches,
                        "crit": running_crit_loss/running_batches,
                        "rel": running_rel_loss/running_batches
                    }
                    # Updating progress bar
                    items_pbar.set_postfix({"loss": phase_losses[phase]})
                    
                    # Compute minibatch validation metrics
                    if phase == "val":
                        # Storing images to plot
                        if len(images_to_plot) < vals_to_plot:
                            for image, target, output in zip(images, targets, outputs_softmax):
                                images_to_plot.append(image.cpu())
                                targets_to_plot.append(target.cpu())
                                outputs_to_plot.append(output.cpu())
                                if len(images_to_plot) >= vals_to_plot: break
                                
                        # Segmentation metrics
                        if "dice" in metrics:
                            for _class in range(num_classes):
                                outputs_dices[_class] = outputs_dices[_class] + [dice_score(outputs_argmax, targets, _class)]
                                outputs_precisions[_class] = outputs_precisions[_class] + [precision(outputs_argmax, targets, _class)]
                                outputs_recalls[_class] = outputs_recalls[_class] + [recall(outputs_argmax, targets, _class)]
                        if "cc" in metrics:
                            for _class in range(num_classes):
                                outputs_connected_components[_class] = outputs_connected_components[_class] + [count_connected_components(outputs_argmax, _class)]
                         
                        # Detection metrics
                        if "iou" in metrics:
                            for _class in range(num_classes):
                                outputs_ious[_class] = outputs_ious[_class] + [jaccard(outputs, targets, _class)]
                        
                        # Relational metrics
                        if "dice" in metrics or "cc" in metrics:
                            for rel_crit_idx in range(num_relational):
                                outputs_relational_scores[rel_crit_idx] = outputs_relational_scores[rel_crit_idx] + [relational_criterions[rel_crit_idx].compute_metric(outputs_softmax, targets)]
                        else:
                            for rel_crit_idx in range(num_relational):
                                outputs_relational_scores[rel_crit_idx] = outputs_relational_scores[rel_crit_idx] + [relational_criterions[rel_crit_idx].compute_metric(outputs_softmax)]

        # Epoch is done, save checkpoint
        torch.save(model.state_dict(), os.path.join(model_training_path, "last_model.pth"))
            
        # Epoch is done, verify validation loss to save best model
        if phase_losses["val"]["all"] < best_loss:
            best_loss = phase_losses["val"]["all"]
            best_model_weights = deepcopy(model.state_dict())
            print("* New best model *")
            torch.save(model.state_dict(), os.path.join(model_training_path, "best_model.pth"))

        # Epoch is done, compute validation metrics
        with torch.no_grad():
            # Collapse metric lists (from N/B x B to N)
            if "dice" in metrics: 
                outputs_dices = [torch.cat(outputs_dices[_class]) for _class in range(num_classes)]
                outputs_precisions = [torch.cat(outputs_precisions[_class]) for _class in range(num_classes)]
                outputs_recalls = [torch.cat(outputs_recalls[_class]) for _class in range(num_classes)]
            if "cc" in metrics: 
                outputs_connected_components = [list(chain.from_iterable(outputs_connected_components[_class])) for _class in range(num_classes)]
            if "iou" in metrics:
                outputs_ious = [torch.cat(outputs_ious[_class]) for _class in range(num_classes)]
            outputs_relational_scores = [torch.cat(outputs_relational_scores[crit_idx]) for crit_idx in range(num_relational)]

            # Printing report
            if "dice" in metrics:
                # Compute dices
                # Print foreground dices
                mean_output_dices = torch.mean(torch.stack(outputs_dices), dim=1)
                mean_output_precisions = torch.mean(torch.stack(outputs_precisions), dim=1)
                mean_output_recalls = torch.mean(torch.stack(outputs_recalls), dim=1)
                print("Mean foreground Dices: ", end="")
                for mean_output_dice in mean_output_dices[1:]:
                    print("{:.4f}, ".format(mean_output_dice.item()), end="")
                print("")
            if "iou" in metrics:
                # Compute IoUs
                # Print foreground IoUs
                mean_output_ious = torch.mean(torch.stack(outputs_ious), dim=1)
                print("Mean foreground IoUs: ", end="")
                for mean_output_iou in mean_output_ious:
                    print("{:.4f}, ".format(mean_output_iou.item()), end="")
                print("")

            # Save validation metrics
            epoch_validation_path = os.path.join(validation_path, "epoch_{}".format(epoch))
            mkdir(epoch_validation_path)
            validation_metrics = { 
                "train_loss_all" : phase_losses["train"]["all"],
                "train_loss_crit" : phase_losses["train"]["crit"],
                "train_loss_rel" : phase_losses["train"]["rel"],
                "val_loss_all" : phase_losses["val"]["all"],
                "val_loss_crit" : phase_losses["val"]["crit"],
                "val_loss_rel" : phase_losses["val"]["rel"],
                "all" : 
                [
                    {
                        _class : {
                            "Relational Losses" : [outputs_relational_score[val_index].item() for outputs_relational_score in outputs_relational_scores],
                        } for _class in range(num_classes)
                    } for val_index in range(validation_count)
                ],
                "mean": {
                    _class : {
                        "Relational Losses" : [torch.mean(outputs_relational_score).item() for outputs_relational_score in outputs_relational_scores],
                    } for _class in range(num_classes)
                }
            }
            # If dice is one of the validation metrics:
            if "dice" in metrics:
                for _class in range(num_classes):
                    validation_metrics["mean"][_class]["Dice"] = torch.mean(outputs_dices[_class]).item()
                    validation_metrics["mean"][_class]["Precision"] = torch.mean(outputs_precisions[_class]).item()
                    validation_metrics["mean"][_class]["Recall"] = torch.mean(outputs_recalls[_class]).item()
                    for val_index in range(validation_count):
                        validation_metrics["all"][val_index][_class]["Dice"] = outputs_dices[_class][val_index].item()
                        validation_metrics["all"][val_index][_class]["Precision"] = outputs_precisions[_class][val_index].item()
                        validation_metrics["all"][val_index][_class]["Recall"] = outputs_recalls[_class][val_index].item()
            # If cc is one of the validation metrics:
            if "cc" in metrics:
                for _class in range(num_classes):
                    validation_metrics["mean"][_class]["Connected Components"] = np.mean(outputs_connected_components[_class])
                    for val_index in range(validation_count):
                        validation_metrics["all"][val_index][_class]["Connected Components"] = outputs_connected_components[_class][val_index]
            # If iou is one of the validation metrics:
            if "iou" in metrics:
                for _class in range(num_classes):
                    validation_metrics["mean"][_class]["Jaccard"] = torch.mean(outputs_ious[_class]).item()
                    for val_index in range(validation_count):
                        validation_metrics["all"][val_index][_class]["Jaccard"] = outputs_ious[_class][val_index].item()
            

            with open(os.path.join(epoch_validation_path, "summary.json"), 'w') as f:
                json.dump(validation_metrics, f, sort_keys=True, indent=4)


            # Save validation images
            if "dice" in metrics or "cc" in metrics:
                for i, (val_image, val_targets, val_outputs) in enumerate(zip(images_to_plot, 
                                                                            targets_to_plot, 
                                                                            outputs_to_plot)):
                    plot_output(val_image, val_targets, val_outputs, os.path.join(epoch_validation_path, "val{}.png".format(i)))
            if "iou" in metrics:
                for i, (val_image, val_targets, val_outputs) in enumerate(zip(images_to_plot, 
                                                                            targets_to_plot, 
                                                                            outputs_to_plot)):
                    plot_output_det(val_image, val_targets, val_outputs, os.path.join(epoch_validation_path, "val{}.png".format(i)))

    # Load best weights
    model.load_state_dict(best_model_weights)
    return model
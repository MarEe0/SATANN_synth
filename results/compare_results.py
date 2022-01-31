import os
import json

import matplotlib.pyplot as plt
import numpy as np


def nested_dict_values(d):
    for v in d.values():
        if isinstance(v, dict):
            yield from nested_dict_values(v)
        else:
            yield v


def get_val_rel_losses_dice(path_base, experiment_label):
    path_results = os.path.join(path_base, experiment_label)
    # Plotting losses and mean foreground dice over epochs
    epoch_count = len(next(os.walk(os.path.join(path_results, "val")))[1])
    val_losses, val_crit_losses, val_rel_losses = [], [], []
    dices = []
    relational_losses = []
    for epoch in range(epoch_count):
        with open(os.path.join(path_results, "val", "epoch_{}".format(epoch), "summary.json")) as f:
            epoch_results = json.load(f)
            val_losses.append(epoch_results["val_loss_all"])
            val_crit_losses.append(epoch_results["val_loss_crit"])
            val_rel_losses.append(epoch_results["val_loss_rel"])
            relational_losses.append(
                epoch_results["mean"]["0"]["Relational Loss"])
            dices.append([epoch_results["mean"][_class]["Dice"]
                         for _class in list(epoch_results["mean"].keys())[1:]])
    return val_losses, relational_losses, dices


if __name__ == "__main__":
    path_base = "/home/mriva/Recherche/PhD/SATANN/synthetic/results/dataset_20"
    experiment_labels = ["T_easy_noise", "T_hard_noise", "T_veryhard_noise"]
    model_seeds = range(2)
    dataset_seeds = range(2)
    #alphas = [0, 0.5]
    alphas = [0]

    val_losses = {experiment_label: {model_seed: {dataset_seed: {alpha: {} for alpha in alphas}
                                                  for dataset_seed in dataset_seeds} for model_seed in model_seeds} for experiment_label in experiment_labels}
    rel_losses = {experiment_label: {model_seed: {dataset_seed: {alpha: {} for alpha in alphas}
                                                  for dataset_seed in dataset_seeds} for model_seed in model_seeds} for experiment_label in experiment_labels}
    dices = {experiment_label: {model_seed: {dataset_seed: {alpha: {} for alpha in alphas}
                                             for dataset_seed in dataset_seeds} for model_seed in model_seeds} for experiment_label in experiment_labels}

    # Loading the losses
    for experiment_label in experiment_labels:
        for model_seed in model_seeds:
            for dataset_seed in dataset_seeds:
                for alpha in alphas:
                    val_losses[experiment_label][model_seed][dataset_seed][alpha], rel_losses[experiment_label][model_seed][dataset_seed][alpha], dices[experiment_label][model_seed][dataset_seed][alpha] = get_val_rel_losses_dice(
                        path_base, "{}_m{}_d{}_a{}".format(experiment_label, model_seed, dataset_seed, alpha))

    # Getting the minimal vals and corresponding relational losses
    min_val = {experiment_label: {model_seed: {dataset_seed: {alpha: {} for alpha in alphas}
                                               for dataset_seed in dataset_seeds} for model_seed in model_seeds} for experiment_label in experiment_labels}
    rel_for_min_val = {experiment_label: {model_seed: {dataset_seed: {alpha: {} for alpha in alphas}
                                                       for dataset_seed in dataset_seeds} for model_seed in model_seeds} for experiment_label in experiment_labels}
    dice_for_min_val = {experiment_label: {model_seed: {dataset_seed: {alpha: {} for alpha in alphas}
                                                        for dataset_seed in dataset_seeds} for model_seed in model_seeds} for experiment_label in experiment_labels}

    rels_for_min_val = {experiment_label: []
                        for experiment_label in experiment_labels}
    for experiment_label in experiment_labels:
        for alpha in alphas:
            for dataset_seed in dataset_seeds:
                for model_seed in model_seeds:
                    min_val_epoch = np.argmin(
                        val_losses[experiment_label][model_seed][dataset_seed][alpha])
                    min_val[experiment_label][model_seed][dataset_seed][alpha] = val_losses[
                        experiment_label][model_seed][dataset_seed][alpha][min_val_epoch]
                    rel_for_min_val[experiment_label][model_seed][dataset_seed][alpha] = rel_losses[
                        experiment_label][model_seed][dataset_seed][alpha][min_val_epoch]
                    dice_for_min_val[experiment_label][model_seed][dataset_seed][alpha] = dices[
                        experiment_label][model_seed][dataset_seed][alpha][min_val_epoch]
                    rels_for_min_val[experiment_label].append(
                        rel_for_min_val[experiment_label][model_seed][dataset_seed][alpha])

    # Calculating epochs to mean rel loss
    epochs_to_mean_rel = {experiment_label: {model_seed: {dataset_seed: {alpha: {} for alpha in alphas}
                                                          for dataset_seed in dataset_seeds} for model_seed in model_seeds} for experiment_label in experiment_labels}

    for experiment_label in experiment_labels:
        # Get mean/std of hardest experiment
        mean_rel, std_rel = np.mean(
            rels_for_min_val[experiment_labels[-1]]), np.std(rels_for_min_val[experiment_labels[-1]])
        for alpha in alphas:
            for dataset_seed in dataset_seeds:
                for model_seed in model_seeds:
                    min_val_epoch = np.argmin(
                        val_losses[experiment_label][model_seed][dataset_seed][alpha])
                    close_rel = abs(
                        rel_losses[experiment_label][model_seed][dataset_seed][alpha] - mean_rel) <= 2*std_rel
                    if len(np.nonzero(close_rel)[0]) > 0:
                        epochs_to_mean_rel[experiment_label][model_seed][dataset_seed][alpha] = np.nonzero(close_rel)[0][0]

    # Printing this MESS
    print("experiment_id, min_val, rel_for_min_val, epochs_to_mean_rel, dice_class1_for_min_val, dice_class2_for_min_val, dice_class3_for_min_val")  # Header
    for experiment_label in experiment_labels:
        for model_seed in model_seeds:
            for dataset_seed in dataset_seeds:
                for alpha in alphas:
                    cur_experiment_identifier = "{}_m{}_d{}_a{:<3}".format(
                        experiment_label, model_seed, dataset_seed, alpha)
                    cur_min_val = "{:.3}".format(
                        min_val[experiment_label][model_seed][dataset_seed][alpha])
                    cur_rel_for_min_val = "{:.3}".format(
                        rel_for_min_val[experiment_label][model_seed][dataset_seed][alpha])
                    cur_epochs_to_mean_rel = "{}".format(
                        epochs_to_mean_rel[experiment_label][model_seed][dataset_seed][alpha])
                    cur_dices_for_min_val = ", ".join(["{:.3}".format(
                        dice_for_min_val[experiment_label][model_seed][dataset_seed][alpha][_class]) for _class in [0, 1, 2]])
                    print(", ".join([cur_experiment_identifier, cur_min_val,
                          cur_rel_for_min_val, cur_epochs_to_mean_rel, cur_dices_for_min_val]))
        # Printing the mean per noise type
        cur_experiment_identifier = experiment_label + "_mean"
        cur_min_val = "{:.3}".format(np.mean([list(nested_dict_values(d))
                                              for d in min_val[experiment_label].values()]))
        cur_rel_for_min_val = "{:.3}".format(np.mean([list(nested_dict_values(
            d)) for d in rel_for_min_val[experiment_label].values()]))
        cur_epochs_to_mean_rel = "{:.3}".format(np.mean([list(nested_dict_values(
            d)) for d in epochs_to_mean_rel[experiment_label].values()]))
        cur_dices_for_min_val = ", ".join(["{:.3}".format(np.mean(np.array(list(nested_dict_values(
            dice_for_min_val[experiment_label])))[:, _class])) for _class in [0, 1, 2]])

        print(", ".join([cur_experiment_identifier, cur_min_val,
                         cur_rel_for_min_val, cur_epochs_to_mean_rel, cur_dices_for_min_val]))

        # Printing the std per noise type
        cur_experiment_identifier = experiment_label + "_std"
        cur_min_val = "{:.3}".format(np.std([list(nested_dict_values(d))
                                             for d in min_val[experiment_label].values()]))
        cur_rel_for_min_val = "{:.3}".format(np.std([list(nested_dict_values(
            d)) for d in rel_for_min_val[experiment_label].values()]))
        cur_epochs_to_mean_rel = "{:.3}".format(np.std([list(nested_dict_values(
            d)) for d in epochs_to_mean_rel[experiment_label].values()]))
        cur_dices_for_min_val = ", ".join(["{:.3}".format(np.std(np.array(list(nested_dict_values(
            dice_for_min_val[experiment_label])))[:, _class])) for _class in [0, 1, 2]])
        print(", ".join([cur_experiment_identifier, cur_min_val,
                         cur_rel_for_min_val, cur_epochs_to_mean_rel, cur_dices_for_min_val]))

        # Breaking a line
        print("")

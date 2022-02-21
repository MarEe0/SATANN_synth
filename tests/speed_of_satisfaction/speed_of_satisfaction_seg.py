import os
import json

import matplotlib.pyplot as plt
import numpy as np


def nested_dict_values(d):
    for v in d.values():
        if v == {}:
            continue #skip empty values
        if isinstance(v, dict):
            yield from nested_dict_values(v)
        else:
            yield v

def flatten(L):
    for item in L:
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def label_to_name(label):
    if "veryhard" in label:
        return "{}-V.H.".format(label[0].upper())
    if "hard" in label:
        return "{}-Hard".format(label[0].upper())
    if "easy" in label:
        return "{}-Easy".format(label[0].upper())

def get_val_rel_losses_dice(path_base, experiment_label):
    path_results = os.path.join(path_base, experiment_label)
    # Plotting losses and mean foreground dice over epochs
    epoch_count = len(next(os.walk(os.path.join(path_results, "val")))[1])
    # Raising misstored cases
    if epoch_count == 0:
        raise Exception("{} failed to produce epochs".format(path_results))
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
    dataset_sizes = [100,1000,10000]

    # Acquire these from the proof_of_convergence
    # Format: list of size D, containing list of size labels
    convergence_lists = [[
                            [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                            [False, True, True, True, False, False, False, True, True, True, False, False, False, False, True, True, False, False, False, False, False, False, True, False, False]
                        ],
                        [
                            [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, False, True, True, True, True, True, True, True, False, True, True, True, False, True, False, False, True, False, False, True]
                        ],
                        [
                            [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
                            [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, True]
                        ]]

    for dataset_size, dataset_convergence_lists in zip(dataset_sizes, convergence_lists):
        path_base = "/media/mriva/LaCie/SATANN/synthetic_fine_segmentation_results/results_seg/dataset_{}".format(dataset_size)
        experiment_labels = ["T_easy_noise", "T_hard_noise"]#, "T_veryhard_noise"]
        model_seeds = range(5)
        dataset_seeds = range(5)
        #alphas = [0, 0.5]
        alphas = [0]
        crit_classes = [0]

        val_losses = {experiment_label: {model_seed: {dataset_seed: {alpha: {} for alpha in alphas}
                                                    for dataset_seed in dataset_seeds} for model_seed in model_seeds} for experiment_label in experiment_labels}
        rel_losses = {experiment_label: {model_seed: {dataset_seed: {alpha: {} for alpha in alphas}
                                                    for dataset_seed in dataset_seeds} for model_seed in model_seeds} for experiment_label in experiment_labels}
        dices = {experiment_label: {model_seed: {dataset_seed: {alpha: {} for alpha in alphas}
                                                for dataset_seed in dataset_seeds} for model_seed in model_seeds} for experiment_label in experiment_labels}

        # Loading the losses
        for experiment_label, convergence_list in zip(experiment_labels, dataset_convergence_lists):
            init_idx = 0  # syncs with convergence list
            for model_seed in model_seeds:
                for dataset_seed in dataset_seeds:
                    for alpha in alphas:
                        if convergence_list[init_idx]:  # Add model data, since it converged
                            try:
                                pack = get_val_rel_losses_dice(path_base, "{}_m{}_d{}_a{}".format(experiment_label, model_seed, dataset_seed, alpha))
                            except:  # Empty epochs bug
                                continue  # Will never set as converged
                            if pack is not None:
                                val_loss, rel_loss, dice = pack 
                                val_losses[experiment_label][model_seed][dataset_seed][alpha], rel_losses[experiment_label][model_seed][dataset_seed][alpha], dices[experiment_label][model_seed][dataset_seed][alpha] = val_loss, rel_loss, dice
                            else: 
                                print("ERROR!", dataset_size, experiment_label, model_seed, dataset_seed, alpha)
                        init_idx += 1

        # Getting the minimal vals and corresponding relational losses
        min_val = {experiment_label: {model_seed: {dataset_seed: {alpha: {} for alpha in alphas}
                                                for dataset_seed in dataset_seeds} for model_seed in model_seeds} for experiment_label in experiment_labels}
        epochs_to_min_val = {experiment_label: {model_seed: {dataset_seed: {alpha: {} for alpha in alphas}
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
                        # Skipping over non-convergeds
                        if val_losses[experiment_label][model_seed][dataset_seed][alpha] == {}: continue

                        min_val_epoch = np.argmin(val_losses[experiment_label][model_seed][dataset_seed][alpha])

                        epochs_to_min_val[experiment_label][model_seed][dataset_seed][alpha] = min_val_epoch

                        min_val[experiment_label][model_seed][dataset_seed][alpha] = val_losses[experiment_label][model_seed][dataset_seed][alpha][min_val_epoch]
                        
                        rel_for_min_val[experiment_label][model_seed][dataset_seed][alpha] = rel_losses[experiment_label][model_seed][dataset_seed][alpha][min_val_epoch]
                        
                        dice_for_min_val[experiment_label][model_seed][dataset_seed][alpha] = dices[experiment_label][model_seed][dataset_seed][alpha][min_val_epoch]
                        
                        rels_for_min_val[experiment_label].append(rel_for_min_val[experiment_label][model_seed][dataset_seed][alpha])

        # Calculating epochs to mean rel loss
        epochs_to_mean_rel = {experiment_label: {model_seed: {dataset_seed: {alpha: {} for alpha in alphas}
                                                            for dataset_seed in dataset_seeds} for model_seed in model_seeds} for experiment_label in experiment_labels}

        for experiment_label in experiment_labels:
            # Get mean/std of easiest experiment
            #mean_rel, std_rel = np.mean(rels_for_min_val[experiment_labels[0]]), np.std(rels_for_min_val[experiment_labels[0]])
            # Values of means and stds for T-Easy on different D:
            #   D=100: 0.11930124893784523, 0.14421368655142672
            #   D=1000: 0.02888376288115978, 0.02442151730516227
            #   D=10000: 0.014036799781024456, 0.0029953978918440703
            rel_satisfaction = np.float64(0.1)
            for alpha in alphas:
                for dataset_seed in dataset_seeds:
                    for model_seed in model_seeds:
                        # Skipping over non-convergeds
                        if rel_losses[experiment_label][model_seed][dataset_seed][alpha] == {}: continue

                        close_rel = rel_losses[experiment_label][model_seed][dataset_seed][alpha] <= rel_satisfaction
                        if len(np.nonzero(close_rel)[0]) > 0:
                            epochs_to_mean_rel[experiment_label][model_seed][dataset_seed][alpha] = np.nonzero(close_rel)[0][0]
                        else:  # never converged
                            epochs_to_mean_rel[experiment_label][model_seed][dataset_seed][alpha] = 100

        # Printing this MESS
        #print("experiment_id, min_val, rel_for_min_val, epochs_to_mean_rel, dice_class1_for_min_val, dice_class2_for_min_val, dice_class3_for_min_val")  # Header
        for experiment_label in experiment_labels:
            # for model_seed in model_seeds:
            #     for dataset_seed in dataset_seeds:
            #         for alpha in alphas:
            #             cur_experiment_identifier = "{}_m{}_d{}_a{:<3}".format(
            #                 experiment_label, model_seed, dataset_seed, alpha)
            #             # Skipping over non-convergences
            #             if min_val[experiment_label][model_seed][dataset_seed][alpha] == {}:
            #                 #print(", ".join([cur_experiment_identifier, "non_converged"])
            #                 continue
                        
            #             cur_min_val = "{:.3}".format(
            #                 min_val[experiment_label][model_seed][dataset_seed][alpha])
            #             cur_rel_for_min_val = "{:.3}".format(
            #                 rel_for_min_val[experiment_label][model_seed][dataset_seed][alpha])
            #             cur_epochs_to_mean_rel = "{}".format(
            #                 epochs_to_mean_rel[experiment_label][model_seed][dataset_seed][alpha])
            #             cur_dices_for_min_val = ", ".join(["{:.3}".format(
            #                 dice_for_min_val[experiment_label][model_seed][dataset_seed][alpha][_class]) for _class in crit_classes])
            #             #print(", ".join([cur_experiment_identifier, cur_min_val,
            #             #      cur_rel_for_min_val, cur_epochs_to_mean_rel, cur_dices_for_min_val]))
            
            # Printing the mean per noise type
            cur_experiment_identifier = experiment_label + "_mean"

            list_of_min_vals = list(flatten([list(nested_dict_values(d)) for d in min_val[experiment_label].values()]))
            cur_min_val_mean = "{:.3f}".format(np.mean(list_of_min_vals))
            cur_min_val_std = "{:.3f}".format(np.std(list_of_min_vals))

            list_of_epochs_to_min_val = list(flatten([list(nested_dict_values(d)) for d in epochs_to_min_val[experiment_label].values()]))
            cur_epochs_to_min_val_mean = "{:.2f}".format(np.mean(list_of_epochs_to_min_val))
            cur_epochs_to_min_val_std = "{:.2f}".format(np.std(list_of_epochs_to_min_val))

            list_of_rel_for_min_val = list(flatten([list(nested_dict_values(d)) for d in rel_for_min_val[experiment_label].values()]))
            cur_rel_for_min_val_mean = "{:.2f}".format(np.mean(list_of_rel_for_min_val))
            cur_rel_for_min_val_std = "{:.2f}".format(np.std(list_of_rel_for_min_val))

            list_of_epochs_to_mean_rel = list(flatten([list(nested_dict_values(d)) for d in epochs_to_mean_rel[experiment_label].values()]))
            cur_epochs_to_mean_rel_mean = "{:.2f}".format(np.mean(list_of_epochs_to_mean_rel))
            cur_epochs_to_mean_rel_std = "{:.2f}".format(np.std(list_of_epochs_to_mean_rel))

            #cur_dices_for_min_val = ", ".join(["{:.2f}".format(np.mean(np.array(list(nested_dict_values(
            #    dice_for_min_val[experiment_label])))[:, _class])) for _class in crit_classes])

            #cur_dices_for_min_val_std = ", ".join(["{:.2f}".format(np.std(np.array(list(nested_dict_values(
            #    dice_for_min_val[experiment_label])))[:, _class])) for _class in crit_classes])

            # Breaking a line
            print("{} & {} & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$ & ${} \pm {}$ \\\\".format(
                label_to_name(experiment_label), 
                dataset_size,
                cur_rel_for_min_val_mean, cur_rel_for_min_val_std,
                cur_epochs_to_mean_rel_mean, cur_epochs_to_mean_rel_std,
                cur_min_val_mean, cur_min_val_std,
                cur_epochs_to_min_val_mean, cur_epochs_to_min_val_std
            ))
        print("\hline")

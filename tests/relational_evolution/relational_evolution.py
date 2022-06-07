"""Script for plotting the evolution of relational measures during training.

Authors
-------
 * Mateus Riva (mateus.riva@telecom-paris.fr)"""
import os, sys
import json
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("/home/mriva/Recherche/PhD/SATANN/SATANN_synth")
from utils import mkdir

path_results_base = "/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/relational_evolution/results"

# Set of colors: train loss, val loss, precision, recall, CSPE, RMO
#colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ["#0006bf", "#3b89ff", "#2ca02c", "#006355", "#d62728", "#e37412"]

def plot_individual_model(path_base, dataset_size, experiment_label):
    """Plot individual model's training curves"""
    model_path = os.path.join(path_base, "dataset_{}".format(dataset_size), experiment_label)
    path_results = os.path.join(path_results_base, "dataset_{}".format(dataset_size))

    epoch_count = len(next(os.walk(os.path.join(model_path, "val")))[1])
    train_losses, val_losses, mean_fg_dices = [], [], []
    mean_fg_precisions, mean_fg_recalls = [], []
    train_crit_losses, train_rel_losses = [], []
    val_crit_losses, val_rel_losses = [], []
    rel_scores = [[], []]
    for epoch in range(epoch_count):
        with open(os.path.join(model_path, "val", "epoch_{}".format(epoch), "summary.json")) as f:
            epoch_results = json.load(f)
            train_losses.append(epoch_results["train_loss_all"])
            train_crit_losses.append(epoch_results["train_loss_crit"])
            train_rel_losses.append(epoch_results["train_loss_rel"])
            val_losses.append(epoch_results["val_loss_all"])
            val_crit_losses.append(epoch_results["val_loss_crit"])
            val_rel_losses.append(epoch_results["val_loss_rel"])
            #mean_fg_dices.append(np.mean([epoch_results["mean"][_class]["Dice"] for _class in list(epoch_results["mean"].keys())[1:]]))
            mean_fg_precisions.append(np.mean([epoch_results["mean"][_class]["Precision"] for _class in list(epoch_results["mean"].keys())[1:]]))
            mean_fg_recalls.append(np.mean([epoch_results["mean"][_class]["Recall"] for _class in list(epoch_results["mean"].keys())[1:]]))
            rel_scores[0].append(1-(np.mean([epoch_results["mean"][_class]["Relational Losses"][0] for _class in list(epoch_results["mean"].keys())[1:]])/(2*sqrt(2))))
            rel_scores[1].append(1-np.mean([epoch_results["mean"][_class]["Relational Losses"][1] for _class in list(epoch_results["mean"].keys())[1:]]))
    
    # Finding best epoch
    best_epoch = np.argmin(val_losses)

    # Cleaning outliers
    mean_train = np.mean(train_losses)
    std_train = np.std(train_losses)
    clean_train = [x for x in train_losses if abs(x-mean_train) < 2*std_train]
    max_train = np.max(clean_train)
    mean_val = np.mean(val_losses)
    std_val = np.std(val_losses)
    clean_val = [x for x in val_losses if abs(x-mean_val) < 2*std_val]
    max_val = np.max(clean_val)
    max_loss = max([max_train, max_val])

    # Hack: force max loss to 0.1
    max_loss = 0.1

    # Starting loss and metrics plot
    fig, ax1 = plt.subplots()
    # Plotting best model line
    ax1.vlines(best_epoch,0,max_loss,"m","dashed")
    # Plotting losses per epoch (left axis)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(top=max_loss)
    train_plot, = ax1.plot(range(epoch_count), train_losses, colors[0], label="Train Loss")
    val_plot, = ax1.plot(range(epoch_count), val_losses, colors[1], label="Val Loss")

    # Plotting metrics per epoch (right axis)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Score")
    ax2.set_ylim(bottom=0, top=1.0)
    precision_plot, = ax2.plot(range(epoch_count), mean_fg_precisions, colors[2], label="Mean Precision")
    recall_plot, = ax2.plot(range(epoch_count), mean_fg_recalls, colors[3], label="Mean Recall")
    cspe_plot, = ax2.plot(range(epoch_count), rel_scores[0], colors[4], label="Mean CSPS")
    rmo_plot, = ax2.plot(range(epoch_count), rel_scores[1], colors[5], label="Mean RMO")

    # Setting legend
    ax1.legend(handles=[train_plot, val_plot, precision_plot, recall_plot, cspe_plot, rmo_plot], loc="center right")
    #ax1.set_title("D={}, {}: Mean Loss and Metrics during training".format(dataset_size, experiment_label))

    mkdir(path_results)
    plt.savefig(os.path.join(path_results, "loss_metrics_per_epoch_{}.png".format(experiment_label)), bbox_inches="tight")
    plt.close()
    plt.clf()

    # Making no-loss plot
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim((0.0,1.0))
    plt.vlines(best_epoch,0,1.0,"m","dashed")
    precision_plot, = plt.plot(range(epoch_count), mean_fg_precisions, colors[2], label="Mean Precision")
    recall_plot, = plt.plot(range(epoch_count), mean_fg_recalls, colors[3], label="Mean Recall")
    cspe_plot = plt.plot(range(epoch_count), rel_scores[0], colors[4], label="Mean CSPS")
    rmo_plot = plt.plot(range(epoch_count), rel_scores[1], colors[5], label="Mean RMO")
    plt.legend()
    #plt.title("D={}, {}: Mean Metrics during training".format(dataset_size, experiment_label))
    plt.savefig(os.path.join(path_results, "metrics_per_epoch_{}.png".format(experiment_label)), bbox_inches="tight")
    plt.close()
    plt.clf()


if __name__ == "__main__":
    path_base = "/media/mriva/LaCie/SATANN/synthetic_fine_segmentation_results/results_strict/"

    dataset_sizes = [1000, 5000, 10000, 50000]
    model_seeds = range(5)
    dataset_seeds = range(5)

    for dataset_size in dataset_sizes:
        for model_seed in model_seeds:
            for dataset_seed in dataset_seeds:
                plot_individual_model(path_base, dataset_size, "T_strict_noise_m{}_d{}_a0".format(model_seed, dataset_seed))
import os
import json

import matplotlib.pyplot as plt
import numpy as np


def plot_experiment(path_base, experiment_label):
    print("Plotting {}".format(experiment_label))

    path_results = os.path.join(path_base, experiment_label)
    # Plotting losses and mean foreground iou over epochs
    epoch_count = len(next(os.walk(os.path.join(path_results, "val")))[1])
    train_losses, val_losses, mean_fg_ious = [], [], []
    train_crit_losses, train_rel_losses = [], []
    val_crit_losses, val_rel_losses = [], []
    for epoch in range(epoch_count):
        with open(os.path.join(path_results, "val", "epoch_{}".format(epoch), "summary.json")) as f:
            epoch_results = json.load(f)
            train_losses.append(epoch_results["train_loss_all"])
            train_crit_losses.append(epoch_results["train_loss_crit"])
            train_rel_losses.append(epoch_results["train_loss_rel"])
            val_losses.append(epoch_results["val_loss_all"])
            val_crit_losses.append(epoch_results["val_loss_crit"])
            val_rel_losses.append(epoch_results["val_loss_rel"])
            mean_fg_ious.append(np.mean([epoch_results["mean"][_class]["Jaccard"] for _class in list(epoch_results["mean"].keys())[1:]]))

    # Outlier cleanup
    mean_train = np.mean(train_losses)
    std_train = np.std(train_losses)
    clean_train = [x for x in train_losses if abs(x-mean_train) < 2*std_train]
    max_train = np.max(clean_train)
    mean_val = np.mean(val_losses)
    std_val = np.std(val_losses)
    clean_val = [x for x in val_losses if abs(x-mean_val) < 2*std_val]
    max_val = np.max(clean_val)
    max_loss = max([max_train, max_val])

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(top=max_loss)
    train_plot, = ax1.plot(range(epoch_count), train_losses, "r", label="Train Loss")
    val_plot, = ax1.plot(range(epoch_count), val_losses, "b", label="Val Loss")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Jaccard")
    ax2.set_ylim(bottom=0, top=1.0)
    iou_plot, = ax2.plot(range(epoch_count), mean_fg_ious, "g", label="Mean OI IoU")

    ax1.legend(handles=[train_plot, val_plot, iou_plot], loc="center right")
    ax1.set_title("{}: Mean Loss and iou during training".format(experiment_label))

    plt.savefig(os.path.join(path_results, "val", "loss_iou_per_epoch.png"))
    plt.close()

    # Plotting split losses over epochs
    mean_crit_train = np.mean(train_crit_losses)
    std_crit_train = np.std(train_crit_losses)
    clean_crit_train = [x for x in train_crit_losses if abs(x-mean_crit_train) < 2*std_crit_train]
    mean_rel_train = np.mean(train_rel_losses)
    std_rel_train = np.std(train_rel_losses)
    clean_rel_train = [x for x in train_rel_losses if abs(x-mean_rel_train) < 2*std_rel_train]
    max_train = np.max(np.concatenate([clean_crit_train, clean_rel_train]))
    mean_crit_val = np.mean(val_crit_losses)
    std_crit_val = np.std(val_crit_losses)
    clean_crit_val = [x for x in val_crit_losses if abs(x-mean_crit_val) < 2*std_crit_val]
    mean_rel_val = np.mean(val_rel_losses)
    std_rel_val = np.std(val_rel_losses)
    clean_rel_val = [x for x in val_rel_losses if abs(x-mean_rel_val) < 2*std_rel_val]
    max_val = np.max(np.concatenate([clean_crit_val, clean_rel_val]))
    max_loss = max([max_train, max_val])

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_ylim(top=max_loss)
    train_crit_plot, = ax1.plot(range(epoch_count), train_crit_losses, color=(0.5,0,0), label="Train Crit. Loss")
    train_rel_plot, = ax1.plot(range(epoch_count), train_rel_losses, color=(1,0,0), label="Train Rel. Loss")
    val_crit_plot, = ax1.plot(range(epoch_count), val_crit_losses, color=(0,0,0.5), label="Val Crit. Loss")
    val_rel_plot, = ax1.plot(range(epoch_count), val_rel_losses, color=(0,0,1), label="Val Rel. Loss")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Jaccard")
    ax2.set_ylim(bottom=0, top=1.0)
    iou_plot, = ax2.plot(range(epoch_count), mean_fg_ious, "g", label="Mean OI iou")

    ax1.legend(handles=[train_crit_plot, train_rel_plot, val_crit_plot, val_rel_plot, iou_plot], loc="center right")
    ax1.set_title("{}: All Loss and iou during training".format(experiment_label))

    plt.savefig(os.path.join(path_results, "val", "split_loss_iou_per_epoch.png"))
    plt.close()


    # Plotting relational loss over epochs
    epoch_count = len(next(os.walk(os.path.join(path_results, "val")))[1])
    relational_losses, ccs_per_class = [], {}
    for epoch in range(epoch_count):
        with open(os.path.join(path_results, "val", "epoch_{}".format(epoch), "summary.json")) as f:
            epoch_results = json.load(f)
            relational_losses.append(epoch_results["mean"]["0"]["Relational Loss"])

    fig, ax1 = plt.subplots()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Relational Loss")
    rel_plot, = ax1.plot(range(epoch_count), relational_losses, "r", label="Relational Loss")

    ax1.legend(handles=[rel_plot], loc="best")
    ax1.set_title("{}: Mean Relational Loss during training".format(experiment_label))

    plt.savefig(os.path.join(path_results, "val", "rel_per_epoch.png"))
    plt.close()

    # Plotting foreground class iou per test output
    test_class_ious = {}
    with open(os.path.join(path_results, "test", "summary.json")) as f:
        test_results = json.load(f)
        for test_item in test_results["all"]:
            for _class in test_item:
                if _class not in test_class_ious:
                    test_class_ious[_class] = []
                
                test_class_ious[_class].append(test_item[_class]["Jaccard"])

    plt.figure(figsize=(12,5))
    for i, _class in list(enumerate(test_class_ious))[1:]:  # Skip background
        plt.plot(np.arange(len(test_class_ious[_class])), test_class_ious[_class], color=plt.get_cmap("tab10")(i), label="Class {}".format(_class))

    # same/rota/swap/dist test divider
    plt.text(0,1,"Original")
    plt.axvline([30], color="k", linestyle="--")
    plt.text(30,1,"Rotation")
    plt.axvline([60], color="k", linestyle="--")
    plt.text(60,1,"Permutation")
    plt.axvline([90], color="k", linestyle="--")
    plt.text(90,1,"Distant")
    plt.title("{}: Test-time Object of Interest iou per class".format(experiment_label))
    plt.ylim(bottom=0, top=1)
    plt.legend()

    plt.savefig(os.path.join(path_results, "test", "iou_per_class.png"))
    plt.close()

    # Plotting relational score per test output
    test_rels = []
    with open(os.path.join(path_results, "test", "summary.json")) as f:
        test_results = json.load(f)
        for test_item in test_results["all"]:
                
            test_rels.append(test_item["0"]["Relational Loss"])

    plt.figure(figsize=(12,5))
    plt.plot(np.arange(len(test_rels)), test_rels, label="Relational loss")

    # same/rota/swap/dist test divider
    plt.axvline([30], color="r", linestyle="--")
    plt.axvline([60], color="r", linestyle="--")
    plt.axvline([90], color="r", linestyle="--")
    plt.title("{}: Test-time Relational Loss".format(experiment_label))
    plt.ylim(bottom=0)
    plt.legend()

    plt.savefig(os.path.join(path_results, "test", "relational_loss.png"))
    plt.close()

if __name__ == "__main__":
    path_base = "/home/mriva/Recherche/PhD/SATANN/SATANN_synth/results/results_det/dataset_400"
    experiment_labels = ["T_easy_noise", "T_hard_noise", "T_veryhard_noise"]
    model_seeds = range(2)
    dataset_seeds = range(2)
    #alphas = [0, 0.2, 0.5, 0.7]
    alphas = [0, 0.5]

    for experiment_label in experiment_labels:
        for model_seed in model_seeds:
            for dataset_seed in dataset_seeds:
                for alpha in alphas:
                    plot_experiment(path_base, "{}_m{}_d{}_a{}".format(experiment_label, model_seed, dataset_seed, alpha))
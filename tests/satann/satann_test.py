"""Script for producing test tables from trained models."""
import os, sys
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.functional import softmax
import torchvision as tv

sys.path.append("/home/mriva/Recherche/PhD/SATANN/SATANN_synth")
from datasets.clostob.clostob_dataset import CloStObDataset
from unet import UNet
from metrics import precision, recall
from utils import targetToTensor, mkdir

def label_to_name(fg_label, label):
    if "strict" in label:
        return "{}-Strict".format(fg_label.upper())
    if "hard" in label:
        return "{}-Hard".format(fg_label.upper())

if __name__ == "__main__":
    base_path = "/media/mriva/LaCie/SATANN/results_alpha"
    # Experiment configurations
    dataset_sizes = [1000, 5000, 10000]
    config_labels = ["strict", "hard"]
    rcs = [0,1]
    alphas = [0.0, 0.25, 0.5, 0.75, 0.99]
    model_seeds = range(3)
    dataset_split_seeds = range(3)


    # Preparing the foreground
    fg_label = "T"  # Change here for other configurations. It's ugly, I know.

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

    experimental_configs = {"strict": {"label": "strict", "bg_classes": [0], "bg_amount": 3, "position_noise": 0, "bg_bboxes": bg_bboxes},
                            "hard": {"label": "hard", "bg_classes": [0], "bg_amount": 3, "position_noise": 0.1, "bg_bboxes": None},
                            "easy": {"label": "easy", "bg_classes": [7], "bg_amount": 3, "position_noise": 0.1, "bg_bboxes": None}}
    relational_criterions_labels = ["CSPE", "RMO"]

    # Iterating over each dataset size
    for dataset_size in dataset_sizes:
        base_dataset_path = os.path.join(base_path, "dataset_{}".format(dataset_size))

        for config_label in config_labels:
            experimental_config = experimental_configs[config_label]
            # Dataset Test params
            test_set_size = 100

            num_classes = len(fg_classes)
            classes = range(1,num_classes+1)
            crit_classes = [1,2,3]
            model_output_channels = len(crit_classes)+1


            # Also setting the image dimensions in advance
            image_dimensions = [160, 160]
            

            # Preparing dataset transforms:
            transform = tv.transforms.Compose(                                  # For the images:
                [tv.transforms.ToTensor(),                                      # Convert to torch.Tensor CxHxW type
                tv.transforms.Normalize((255/2,), (255/2,), inplace=True)])    # Normalize from [0,255] to [-1,1] range
            target_transform = tv.transforms.Compose(                           # For the labelmaps:
                [targetToTensor()])                                             # Convert to torch.Tensor type


            # Preparing the test set
            test_dataset = CloStObDataset(base_dataset_name="fashion",
                                            image_dimensions=image_dimensions,
                                            size=test_set_size,
                                            fg_classes=fg_classes,
                                            fg_positions=base_fg_positions,
                                            position_translation=position_translation,
                                            position_noise=experimental_config["position_noise"],
                                            bg_classes=experimental_config["bg_classes"], # Background class from config
                                            bg_amount=experimental_config["bg_amount"],
                                            bg_bboxes=experimental_config["bg_bboxes"],
                                            flattened=False,
                                            lazy_load=False,
                                            fine_segment=True,
                                            transform=transform,
                                            target_transform=target_transform,
                                            start_seed=100020)


            for rc in rcs:
                rc_label = relational_criterions_labels[rc]

                for alpha in alphas:
                    experiment_label = "{}_{}_{}_a{}".format(fg_label, config_label, rc_label, alpha)

                    # Plot parameters
                    save_image_amount = 20
                    plot_classes = [1,2,3]
                    base_plot_path = "/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/satann/results"
                    plot_path = os.path.join(base_plot_path, "dataset_{}".format(dataset_size), experiment_label)
                    mkdir(plot_path)


                    precisions = {_class : None for _class in classes}
                    recalls = {_class : None for _class in classes}
                    model_has_converged = [True for _ in range(len(model_seeds)*len(dataset_split_seeds))]

                    for model_seed_idx, model_seed in enumerate(model_seeds):
                        for dataset_seed_idx, dataset_split_seed in enumerate(dataset_split_seeds):
                            init_idx = (model_seed_idx*len(model_seeds)) + dataset_seed_idx
                            model_label = "{}_{}_{}_c0,1,2,3_m{}_d{}_a{}".format(fg_label, config_label, rc_label, model_seed, dataset_split_seed, alpha)
                            model_path = os.path.join(base_dataset_path, model_label)

                            # Preparing the data loader
                            data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, num_workers=2)

                            # Loading the specified model
                            model_path = os.path.join(model_path, "best_model.pth")
                            model = UNet(input_channels=1, output_channels=model_output_channels).to(device="cuda")
                            model.load_state_dict(torch.load(model_path))
                            model.eval()

                            # Running the model on the test data
                            inputs, outputs, truths = [], [], []
                            for item_pair in data_loader:
                                input = item_pair["image"]
                                inputs.append(input)
                                with torch.set_grad_enabled(False):
                                    outputs.append(model(input.to(device="cuda")).detach().cpu())
                                truths.append(item_pair["labelmap"])
                            
                            inputs = torch.cat(inputs, dim=0)
                            outputs = torch.cat(outputs, dim=0)
                            truths = torch.cat(truths, dim=0)

                            outputs_softmax = softmax(outputs, dim=1)  # Softmax outputs along class dimension
                            outputs_argmax = outputs_softmax.argmax(dim=1)  # Argmax outputs along class dimension

                            # computing metrics for all classes
                            for _class in crit_classes:
                                class_precisions = precision(outputs_argmax, truths, _class)
                                class_recalls = recall(outputs_argmax, truths, _class)

                                # Checking convergence
                                #model_has_converged[init_idx] = torch.mean(class_precisions).item() > 0.5 and torch.mean(class_recalls).item() > 0.5

                                # Adding to mean only if converged
                                #if model_has_converged[init_idx]:
                                if True:
                                    if precisions[_class] is None:
                                        precisions[_class] = class_precisions
                                    else:
                                        precisions[_class] = torch.cat([precisions[_class], class_precisions], dim=0)
                                    if recalls[_class] is None:
                                        recalls[_class] = class_recalls
                                    else:
                                        recalls[_class] = torch.cat([recalls[_class], class_recalls], dim=0)
                            

                                    # Printing partial results
                                    print("D={}, {}, precision: ".format(dataset_size, model_label),end="")
                                    for _class in crit_classes:
                                        class_precisions = precision(outputs_argmax, truths, _class)
                                        print("C{} {:.3f} +- {:.3f}\t".format(_class, torch.mean(class_precisions).item(), torch.std(class_precisions).item()), end="")
                                        if model_has_converged[init_idx]: print("  *", end="")
                                    print("")
                                    print("D={}, {}, recall:    ".format(dataset_size, model_label),end="")
                                    for _class in crit_classes:
                                        class_recalls = recall(outputs_argmax, truths, _class)
                                        print("C{} {:.3f} +- {:.3f}\t".format(_class, torch.mean(class_recalls).item(), torch.std(class_recalls).item()),end="")
                                        if model_has_converged[init_idx]: print("  *", end="")
                                    print("\n")

                                    # Saving test outputs as images
                                    for test_idx, (input, target, output) in list(enumerate(zip(inputs, truths, outputs_argmax)))[:save_image_amount]:
                                        for _class in plot_classes:
                                            # Converting the input tensor to a 3-channel image
                                            rgb_image = ((np.repeat(input.detach().cpu().numpy().squeeze()[...,None],3,axis=2) + 1) / 2).astype(np.float32)
                                            # Coloring TPs, FPs, FNs,
                                            # Coloring true positives green
                                            rgb_image[(target==_class) & (output==_class)] = (0,1,0)
                                            # Coloring false positives yellow
                                            rgb_image[(target!=_class) & (output==_class)] = (1,1,0)
                                            # Coloring false negatives blue
                                            rgb_image[(target==_class) & (output!=_class)] = (0,0,1)
                                            convergence_marker = "_C" if model_has_converged[init_idx] else "_N"
                                            # Saving image
                                            mkdir(os.path.join(plot_path, model_label+convergence_marker))
                                            plt.imshow(rgb_image)
                                            plt.axis("off")
                                            plt.savefig(os.path.join(plot_path, model_label+convergence_marker, "test{}-{}.png".format(test_idx,_class)), bbox_inches="tight")
                                            #plt.savefig(os.path.join(plot_path, model_label+convergence_marker, "test{}.eps".format(test_idx)), bbox_inches="tight")
                                            plt.clf()

                    print("")
                        
                    # Printing results (latex format)
                    # Config. | RelLoss | D. | Alpha | Precision1 | Recall1 | (other classes?) | ConvergeRate
                    print(label_to_name(fg_label, experimental_config["label"]), end=" & ")
                    print(rc_label, end=" & ")
                    print(dataset_size, end=" & ")
                    print(alpha, end=" & ")
                    for _class in crit_classes:
                        if any(model_has_converged):
                            mean_class_precision, std_class_precision = precisions[_class].mean().item(), precisions[_class].std().item()
                            mean_class_recall, std_class_recall = recalls[_class].mean().item(), recalls[_class].std().item()
                            print("${:.2} \pm {:.2}$".format(mean_class_precision, std_class_precision), end=" & ")
                            print("${:.2} \pm {:.2}$".format(mean_class_recall, std_class_recall), end=" & ")
                        else:  # No models have converged
                            print("$N/A$", end=" & ")
                            print("$N/A$", end=" & ")
                    print("{}/{}".format(sum(model_has_converged), len(model_has_converged)), end="")
                    print("\\\\")
                    print(model_has_converged)
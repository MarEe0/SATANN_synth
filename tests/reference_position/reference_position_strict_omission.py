"""Script for producing sliding reference position precision and recall heatmaps"""
from cmath import sqrt
from math import floor
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

def label_to_name(label):
    if "strict" in label:
        return "{}-Strict".format(label[0].upper())

if __name__ == "__main__":
    base_path = "/media/mriva/LaCie/SATANN/synthetic_fine_segmentation_results/results_strict"
    dataset_sizes = [1000,5000,10000,50000]
    # Acquire these from the proof_of_convergence
    # Format: list of size D, containing list of size labels
    convergence_lists = [[[True, True, False, False, False, False, False, True, False, True, False, False, True, False, False, False, False, False, False, False, False, True, False, False, False]],
                         [[False, True, False, True, False, True, True, True, False, False, True, False, True, True, True, True, False, False, False, True, False, True, False, True, True]],
                         [[True, True, True, True, True, True, True, True, False, True, False, True, True, True, True, True, False, True, True, True, True, False, True, True, True]],
                         [[True, True, True, True, True, False, True, True, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False]]]
    
    # Iterating over each dataset size
    for dataset_size, convergence_list in zip(dataset_sizes, convergence_lists):
        base_dataset_path = os.path.join(base_path, "dataset_{}".format(dataset_size))

        test_set_size = 20
        element_shape = (28,28)
        stride = 20

        # Preparing the foreground
        fg_label = "T"
        fg_classes = [0, 1, 8]
        base_fg_positions = [(0.65, 0.3), (0.65, 0.7), (0.35, 0.7)]
        position_translation=0.0
        position_noise=0.0
        bg_bboxes = (0.4, 0.0, 0.9, 0.5)

        num_classes = len(fg_classes)
        classes = range(1,num_classes+1)
        crit_classes =[1]
        omission_classes=[int(x) for x in sys.argv[2:]]

        # Also setting the image dimensions in advance
        image_dimensions = [160, 160]
        

        # Preparing dataset transforms:
        transform = tv.transforms.Compose(                                  # For the images:
            [tv.transforms.ToTensor(),                                      # Convert to torch.Tensor CxHxW type
            tv.transforms.Normalize((255/2,), (255/2,), inplace=True)])    # Normalize from [0,255] to [-1,1] range
        target_transform = tv.transforms.Compose(                           # For the labelmaps:
            [targetToTensor()])                                             # Convert to torch.Tensor type

        # Experiment configurations
        experimental_configs =  [{"label": fg_label + "_strict_noise", "bg_classes": [0], "bg_amount": 3}]
                                #[{"label": fg_label + "_hard_noise", "bg_classes": [0], "bg_amount": 3}]#,
                                #{"label": fg_label + "_easy_noise", "bg_classes": [7], "bg_amount": 3},
                                #{"label": fg_label + "_veryhard_noise", "bg_classes": [0,1,8], "bg_amount": 6}]
        
        # Getting results for a specific experiment configuration
        for experimental_config, config_convergence in zip(experimental_configs, convergence_list):
            # Preparing the test set
            test_dataset = CloStObDataset(base_dataset_name="fashion",
                                            image_dimensions=image_dimensions,
                                            size=test_set_size,
                                            fg_classes=fg_classes,
                                            fg_positions=base_fg_positions,
                                            position_translation=position_translation,
                                            position_noise=position_noise,
                                            bg_classes=experimental_config["bg_classes"], # Background class from config
                                            bg_amount=experimental_config["bg_amount"],
                                            bg_bboxes=bg_bboxes,
                                            flattened=False,
                                            lazy_load=False,
                                            fine_segment=True,
                                            transform=transform,
                                            target_transform=target_transform,
                                            start_seed=100000)
            # Getting results for each reference class shift
            for reference_class in [int(sys.argv[1])]:
                # Note: CSO functions take "which" class, like '0' for shirt or '8' for bag
                # While this test takes 1,2,3, hence the need for conversion
                converted_class = fg_classes[reference_class-1]
                shifts_anchors_set = [test_dataset.generate_reference_shifts(i, converted_class, element_shape=element_shape, stride=stride, omission_idxs=omission_classes) for i in range(test_set_size)]
                shifts_set = [item for sublist in shifts_anchors_set for item in sublist[0]]  # Flattening
                anchors_set = [item for sublist in shifts_anchors_set for item in sublist[1]]  # Flattening
                all_anchors = anchors_set[:(len(anchors_set)//test_set_size)]

                # REFERENCE POSITION:
                #   Getting test-time precision and recall per class, per anchor for all converged inits
                initialization_paths = sorted(list(glob(os.path.join(base_dataset_path, experimental_config["label"] + "*"))))
                initialization_paths = [item for item in initialization_paths if item[-2:] != ".5"]  # skipping SATANN examples (where alpha > 0)
                initialization_paths = [item for idx, item in enumerate(initialization_paths) if config_convergence[idx]]  # skipping non-converged models

                precisions = {_class : {anchor: torch.zeros(test_set_size*len(initialization_paths)) for anchor in all_anchors} for _class in crit_classes}
                recalls = {_class : {anchor: torch.zeros(test_set_size*len(initialization_paths)) for anchor in all_anchors} for _class in crit_classes}
                
                for init_idx, initialization_path in enumerate(initialization_paths):
                    print("Doing {}, ref {} om {}, {}".format(dataset_size, reference_class, omission_classes, os.path.split(initialization_path)[-1]))
                    # Loading the specified model
                    model_path = os.path.join(initialization_path, "best_model.pth")
                    model = UNet(input_channels=1, output_channels=len(crit_classes)+1).to(device="cuda")
                    model.load_state_dict(torch.load(model_path))
                    model.eval()

                    # Running the model on the test data
                    outputs, truths = [], []
                    for item_pair in shifts_set:
                        inputs = item_pair["image"].unsqueeze(0).to(device="cuda")
                        with torch.set_grad_enabled(False):
                            outputs.append(model(inputs).detach().cpu())
                        truths.append(item_pair["labelmap"])
                    
                    outputs = torch.cat(outputs, dim=0)
                    truths = torch.stack(truths)

                    outputs_softmax = softmax(outputs, dim=1)  # Softmax outputs along class dimension
                    outputs_argmax = outputs_softmax.argmax(dim=1)  # Argmax outputs along class dimension
                
                    # computing metrics for all classes
                    for _class in crit_classes:
                        class_precisions = precision(outputs_argmax, truths, _class)
                        class_recalls = recall(outputs_argmax, truths, _class)

                        for idx, (class_precision, anchor) in enumerate(zip(class_precisions, anchors_set)):
                            current_idx = floor((idx/len(all_anchors))+(init_idx*test_set_size))
                            precisions[_class][anchor][current_idx] = class_precision
                        for idx, (class_recall, anchor) in enumerate(zip(class_recalls, anchors_set)):
                            current_idx = floor((idx/len(all_anchors))+(init_idx*test_set_size))
                            recalls[_class][anchor][current_idx] = class_recall
                    
                    #for item_pair, output_argmax, anchor in zip(shifts_set, outputs_argmax, anchors_set):
                    #    plt.subplot(121); plt.imshow(item_pair["image"][0].detach().cpu().numpy(), cmap="gray")
                    #    plt.subplot(122); plt.imshow(output_argmax.detach().cpu().numpy())
                    #    plt.title("Anchor {} - $\\mu$R={:.3f}".format(anchor, torch.mean(recalls[1][anchor][:((init_idx+1)*test_set_size)]).item()))
                    #    plt.show()

                # Got all results for this configuration for this reference class
                # Averaging each anchor point
                for _class in crit_classes:
                    for anchor in all_anchors:
                        precisions[_class][anchor] = torch.mean(precisions[_class][anchor])
                        recalls[_class][anchor] = torch.mean(recalls[_class][anchor])
                
                # One heatmap per metric per base class
                for _class in crit_classes:
                    # Preparing heatmaps (shifting to numpy)
                    recall_values_per_pixel = [[[] for _ in range(image_dimensions[1])] for _ in range(image_dimensions[0])]
                    precision_values_per_pixel = [[[] for _ in range(image_dimensions[1])] for _ in range(image_dimensions[0])]
                    recall_heatmap = np.zeros(image_dimensions)
                    precision_heatmap = np.zeros(image_dimensions)
                    # Assembling heatmaps
                    for anchor in all_anchors:
                        other_end = tuple(map(sum, zip(anchor,element_shape)))
                        coordinates_set = tuple(np.s_[origin:end] for origin, end in zip(anchor, other_end))
                        for i in range(anchor[0], other_end[0]):
                            for j in range(anchor[1], other_end[1]):
                                recall_values_per_pixel[i][j].append(recalls[_class][anchor].item())
                                precision_values_per_pixel[i][j].append(precisions[_class][anchor].item())
        
                    for i in range(image_dimensions[0]):
                        for j in range(image_dimensions[1]):
                            recall_heatmap[i,j] = np.mean(recall_values_per_pixel[i][j])
                            precision_heatmap[i,j] = np.mean(precision_values_per_pixel[i][j])

                    # Preparing images
                    plt.imshow(recall_heatmap, vmin=0, vmax=1)
                    #plt.title("Effects of reference {} on class {} recall".format(reference_class, _class))
                    mkdir("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/reference_position/results_strict/dataset_{}/{}".format(dataset_size, experimental_config["label"]))
                    plt.axis("off")
                    plt.savefig("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/reference_position/results_strict/dataset_{}/{}/ref{}on{}_omission{}_recall.png".format(dataset_size, experimental_config["label"], reference_class, _class, omission_classes), bbox_inches="tight")
                    plt.clf()
                    np.save("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/reference_position/results_strict/dataset_{}/{}/ref{}on{}_omission{}_recall.npy".format(dataset_size, experimental_config["label"], reference_class, _class, omission_classes), recall_heatmap)
                    plt.imshow(precision_heatmap, vmin=0, vmax=1)
                    #plt.title("Effects of reference {} on class {} precision".format(reference_class, _class))
                    mkdir("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/reference_position/results_strict/dataset_{}/{}".format(dataset_size, experimental_config["label"]))
                    plt.axis("off")
                    plt.savefig("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/reference_position/results_strict/dataset_{}/{}/ref{}on{}_omission{}_precision.png".format(dataset_size, experimental_config["label"], reference_class, _class, omission_classes), bbox_inches="tight")
                    plt.clf()
                    np.save("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/reference_position/results_strict/dataset_{}/{}/ref{}on{}_omission{}_precision.npy".format(dataset_size, experimental_config["label"], reference_class, _class, omission_classes), precision_heatmap)


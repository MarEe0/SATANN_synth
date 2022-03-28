"""Script for producing meaningless information maps from trained models."""
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

if __name__=="__main__":
    base_path = "/media/mriva/LaCie/SATANN/synthetic_fine_segmentation_results/results_strict"
    # Iterating over each dataset size
    for dataset_size in [50000]:
        base_dataset_path = os.path.join(base_path, "dataset_{}".format(dataset_size))

        test_set_size = 10

        # Preparing the foreground
        fg_label = "T"
        fg_classes = [0, 1, 8]
        base_fg_positions = [(0.65, 0.3), (0.65, 0.7), (0.35, 0.7)]
        position_translation=0.0
        position_noise=0
        bg_bboxes = (0.4, 0.0, 0.9, 0.5)

        num_classes = len(fg_classes)
        classes = range(1,num_classes+1)
        omission_classes = [[0], [0,1], [0,8]]

        # Also setting the image dimensions in advance
        image_dimensions = [160, 160]

        # Preparing dataset transforms:
        transform = tv.transforms.Compose(                                  # For the images:
            [tv.transforms.ToTensor(),                                      # Convert to torch.Tensor CxHxW type
            tv.transforms.Normalize((255/2,), (255/2,), inplace=True)])    # Normalize from [0,255] to [-1,1] range
        target_transform = tv.transforms.Compose(                           # For the labelmaps:
            [targetToTensor()])                                             # Convert to torch.Tensor type

        # Experiment configurations
        experimental_configs = [{"label": fg_label + "_strict_noise", "bg_classes": [0], "bg_amount": 3}]
        
        # Getting results for a specific experiment configuration
        for experimental_config in experimental_configs:
            # Getting the test image
            test_dataset = CloStObDataset(base_dataset_name="fashion",
                                            image_dimensions=image_dimensions,
                                            size=test_set_size,
                                            fg_classes=fg_classes,
                                            fg_positions=base_fg_positions,
                                            position_translation=position_translation,
                                            position_noise=position_noise,
                                            bg_classes=experimental_config["bg_classes"], # Background class from config
                                            bg_amount=experimental_config["bg_amount"],
                                            bg_bboxes=(0.4, 0.0, 0.9, 0.5),
                                            fine_segment=True,
                                            flattened=False,
                                            lazy_load=True,
                                            transform=transform,
                                            target_transform=target_transform)
            
            # MEANINGLESS INFORMATION:
            #   Obtain the probability map of the non-represented class for all permutations
            #   of other classes' omissions, for all inits
            for initialization_path in sorted(list(glob(os.path.join(base_dataset_path, experimental_config["label"] + "*")))):
                # skipping SATANN examples (where alpha > 0)
                if initialization_path[-2:] == ".5": continue
                model_label = os.path.split(initialization_path)[-1]

                # Loading the specified model
                model_path = os.path.join(initialization_path, "best_model.pth")
                model = UNet(input_channels=1, output_channels=2).to(device="cuda")
                model.load_state_dict(torch.load(model_path))
                model.eval()

                for omission_class in omission_classes:
                    mkdir("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/meaningless_information/results_strict/dataset_{}/{}/omissions_{}".format(dataset_size, model_label, omission_class))
                    
                    # List of probability maps to obtain
                    probability_maps = np.empty((test_set_size, *image_dimensions))
                    for test_idx in range(test_set_size):
                        # Getting a meaningless image with a single reference
                        test_image = test_dataset.generate_meaningless_image(idx=test_idx, omitted_classes=omission_class, add_bg_noise=False)["image"].to(device="cuda")

                        # Running the model on the test image
                        with torch.set_grad_enabled(False):
                            output = model(test_image.unsqueeze(0)).detach().cpu()
                        
                        # Softmaxing alongside class dimension
                        output_softmax = softmax(output, dim=1)
                        probability_maps[test_idx] = output_softmax[0,1]

                        # Showing current results
                        plt.subplot(121)
                        plt.imshow(test_image[0].cpu(), cmap="gray", vmin=-1, vmax=1)
                        plt.axis("off")
                        plt.subplot(122)
                        plt.imshow(output_softmax[0,1], cmap="jet", vmin=0, vmax=1)
                        plt.axis("off")
                        plt.suptitle("D={}, {}, omission: {}, i={}".format(dataset_size, model_label, omission_class, test_idx))
                        plt.savefig("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/meaningless_information/results_strict/dataset_{}/{}/omissions_{}/i{}.png".format(dataset_size, model_label, omission_class, test_idx), bbox_inches="tight")
                        plt.clf()

                    # Getting mean probability map for this omission case
                    mean_probability_map = probability_maps.mean(0)
                    plt.imshow(mean_probability_map, cmap="jet", vmin=0, vmax=1)
                    plt.axis("off")
                    plt.title("Mean for D={}, {}, omission: {}".format(dataset_size, model_label, omission_class))
                    plt.savefig("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/meaningless_information/results_strict/dataset_{}/{}/omissions_{}.png".format(dataset_size, model_label, omission_class), bbox_inches="tight")
                    plt.clf()


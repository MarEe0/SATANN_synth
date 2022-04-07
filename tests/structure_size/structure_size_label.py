"""Computes the Receptive Field Reference Overlap measure for structure size experiments.

Author
------
 * Mateus Riva (mateus.riva@telecom-paris.fr)
"""
import os
from shutil import which
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.append("/home/mriva/Recherche/PhD/SATANN/SATANN_synth")
from utils import mkdir, multi_logical_or
from datasets.clostob.clostob_dataset import CloStObDataset

def receptive_field_reference_overlap(labelmap, interest_class, reference_classes, rf_size, which="furthest"):
    """Computes the Receptive Field Reference Overlap for a given sample.
    
    The RFRO is the measure of which percentage of pixels belonging to the
    `reference_classes` is inside a `rf_size`-shaped square centered on the
    `interest_class` pixel further from the `reference_classes`.
    
    Currently only works in 2D.
    
    Arguments
    ---------
    which: "furthest", "center", "closest"
        From which pixel in the interest class to compute the overlap from."""
    # If rf_size is a single number, repeat it to match labelmap dimension
    if type(rf_size) is int:
        rf_size = np.repeat(rf_size, len(labelmap.shape))

    # getting reference_classes map
    reference_classes_map = multi_logical_or([labelmap == reference_class for reference_class in reference_classes])
    
    # Getting interest pixel
    if which=="furthest" or which=="closest":  # Furthest/closest from reference
        # Computing pixel distance-from-reference_classes map
        distance_map = distance_transform_edt(np.logical_not(reference_classes_map))
        # getting distance in interest_class map
        distance_interest_map = distance_map
        distance_interest_map[labelmap != interest_class] = 0

        # Setting function to use
        measure = np.max if which=="furthest" else np.min
        extreme_value = measure(distance_interest_map[distance_interest_map > 0])

        # Finding index of furthest pixel in the interest_class
        interest_pixel_idx = [x[0] for x in np.where(distance_interest_map == extreme_value)]
    elif which=="center":  # Center of interest
        interest_pixel_idx = np.mean(np.transpose((labelmap==interest_class).nonzero()), axis=0).astype(int)
    else:
        raise ValueError("which must be 'center', 'closest',  or 'furthest'; got {}".format(which))
    # Finding the RF of this pixel
    left_rf = max(0,interest_pixel_idx[0]-(rf_size[0]//2))
    right_rf = min(interest_pixel_idx[0]+(rf_size[0]//2), labelmap.shape[0])
    top_rf = max(0,interest_pixel_idx[1]-(rf_size[1]//2))
    bottom_rf = min(interest_pixel_idx[1]+(rf_size[1]//2), labelmap.shape[1])
    # Counting reference_classes pixels in the rf of this pixel
    seen_pixels = np.sum(reference_classes_map[left_rf:right_rf,top_rf:bottom_rf])
    # Computing percentage of pixels of reference_classes_map seen
    return seen_pixels/np.sum(reference_classes_map)


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.segmentation import mark_boundaries
def show_sample_with_receptive_field(sample, interest_class, reference_classes, rf_size, which, color=(1,0,0)):
    """Function for plotting a singular sample with the receptive fields overlayed"""
    image = sample["image"] / 256.0
    labelmap = sample["labelmap"]
    image = mark_boundaries(image, sample["labelmap"]==1, color=plt.get_cmap("tab10")(1)[:3], mode="thick", background_label=0)
    plt.imshow(image)  # Plotting the image
    
    # If rf_size is a single number, repeat it to match labelmap dimension
    if type(rf_size) is int:
        rf_size = np.repeat(rf_size, len(labelmap.shape))

    # getting reference_classes map
    reference_classes_map = multi_logical_or([labelmap == reference_class for reference_class in reference_classes])
    
    # Getting interest pixel
    if which=="furthest" or which=="closest":  # Furthest/closest from reference
        # Computing pixel distance-from-reference_classes map
        distance_map = distance_transform_edt(np.logical_not(reference_classes_map))
        # getting distance in interest_class map
        distance_interest_map = distance_map
        distance_interest_map[labelmap != interest_class] = 0

        # Setting function to use
        measure = np.max if which=="furthest" else np.min
        extreme_value = measure(distance_interest_map[distance_interest_map > 0])

        # Finding index of furthest pixel in the interest_class
        interest_pixel_idx = [x[0] for x in np.where(distance_interest_map == extreme_value)]
    elif which=="center":  # Center of interest
        interest_pixel_idx = np.mean(np.transpose((labelmap==interest_class).nonzero()), axis=0).astype(int)
    else:
        raise ValueError("which must be 'center', 'closest',  or 'furthest'; got {}".format(which))

    # Drawing receptive fields
    rf_patch = patches.Rectangle(np.array(interest_pixel_idx[::-1]) - (np.array(rf_size)//2), *rf_size,
                                            linewidth=1, edgecolor=color, facecolor="none")
    plt.gca().add_patch(rf_patch)

    plt.axis("off")



if __name__ == "__main__":
    test_set_size = 100
    
    # Preparing the fixed foreground information
    fg_label = "T"
    fg_classes = [0, 1, 8]
    #base_fg_positions = [(0.65, 0.3), (0.65, 0.7), (0.35, 0.7)]
    #position_translation=0.5
    position_noise=0
    #bg_bboxes = (0.4, 0.0, 0.9, 0.5)

    # Also setting the image dimensions in advance
    image_dimensions = [256,256]
    object_dimensions = [28,28]  # This does not control object size, but is used as reference
    rf_labels = ["BF","OF"]
    rf_sizes = [61,101]

    # Which images of the dataset to plot
    idxs_to_plot = 4

    # Preparing the limited cross entropy targets
    crit_classes = [0,1]  # BG and first class (shirts)

    experimental_configs = [{"label": fg_label + "_strict_noise_incbf", "bg_classes": [0], "bg_amount": 3, "structure_size": [24,32,40]},
                            {"label": fg_label + "_strict_noise_medbf", "bg_classes": [0], "bg_amount": 3, "structure_size": [30,40,50]},
                            {"label": fg_label + "_strict_noise_inbf", "bg_classes": [0], "bg_amount": 3, "structure_size": [33,44,55]},
                            {"label": fg_label + "_strict_noise_medof", "bg_classes": [0], "bg_amount": 3, "structure_size": [48,64,80]},
                            {"label": fg_label + "_strict_noise_outbf", "bg_classes": [0], "bg_amount": 3, "structure_size": [60,80,100]},
                            {"label": fg_label + "_strict_noise_outof", "bg_classes": [0], "bg_amount": 3, "structure_size": [84,112,140]}]

    # Running experiments
    for experimental_config in experimental_configs:
        # Label of experiment:
        experiment_label = experimental_config["label"]
        print(experiment_label)
        print(experimental_config["structure_size"])
        # Computing the base fg position for the experimental config - HARDCODED ON TRIANGLE
        fractional_structure_size = np.repeat(np.array(experimental_config["structure_size"])[...,None],2,axis=-1) / image_dimensions
        bottom_left_position = (0.5+(fractional_structure_size[0][0]/2), 0.5-(fractional_structure_size[1][1]/2))
        bottom_right_position = (0.5+(fractional_structure_size[0][0]/2), 0.5+(fractional_structure_size[1][1]/2))
        top_right_position = (0.5-(fractional_structure_size[0][0]/2), 0.5+(fractional_structure_size[1][1]/2))
        base_fg_positions = [bottom_left_position, bottom_right_position, top_right_position]

        # Computing the maximal translational noise
        object_size_buffer = np.array(object_dimensions)/image_dimensions
        left_clearance = min(pos[1] for pos in base_fg_positions) - object_size_buffer[1]
        right_clearance = (1-max(pos[1] for pos in base_fg_positions)) - object_size_buffer[1]
        top_clearance = min(pos[0] for pos in base_fg_positions) - object_size_buffer[0]
        bottom_clearance = (1-max(pos[0] for pos in base_fg_positions)) - object_size_buffer[0]
        position_translation = min(left_clearance, right_clearance, top_clearance, bottom_clearance)*2

        # Computing the bg bboxes
        bg_bboxes = [base_fg_positions[0][0]-(position_translation/2), base_fg_positions[0][1]-(position_translation/2),
                        base_fg_positions[0][0]+(position_translation/2), base_fg_positions[0][1]+(position_translation/2)]


        # Preparing train dataset
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
                                    fine_segment=True,
                                    flattened=False,
                                    lazy_load=True,
                                    transform=None,
                                    target_transform=None)

        # Getting RFROs
        which_labels = ["furthest", "center", "closest"]
        rfros = {rf_label : {which_label: [] for which_label in which_labels} for rf_label in rf_labels}
        for idx, sample in enumerate(test_dataset):
            if idx >= 100: break  # hack to solve the bizarre infinite dataset bug?
            for rf_label, rf_size in zip(rf_labels,rf_sizes):
                for which_label in which_labels:
                    rfros[rf_label][which_label].append(receptive_field_reference_overlap(sample["labelmap"],[1],[2,3],rf_size=rf_size, which=which_label))

                    if idx < idxs_to_plot: # Plotting
                        show_sample_with_receptive_field(sample,[1],[2,3],rf_size=rf_size, which=which_label)
                        plt.savefig("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/structure_size/examples/{}_{}_{}_{}.png".format(experiment_label, rf_size, which_label, idx), bbox_inches="tight"); plt.clf()

        
        # Outputting results
        print("\t{}".format(",".join(which_labels)))
        for rf_label in rf_labels:
            print("\t{}:\t".format(rf_label), end="")
            print(",".join(["{:.4f} \pm {:.4f}".format(np.mean(rfros[rf_label][which_label]), np.std(rfros[rf_label][which_label])) for which_label in which_labels]))
        print("")


"""Collection of utility functions."""
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

class targetToTensor:  # Simple tensor conversion (because tv.ToTensor normalizes)
    def __call__(self, pic):
        return torch.as_tensor(pic)
    def __repr__(self):
        return self.__class__.__name__ + "()"

def multi_logical_or(tensors):
    if len(tensors) <= 1:
        return tensors[0]
    return torch.logical_or(tensors[0], multi_logical_or(tensors[1:]))

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def mkdir(path):
    all_paths = os.path.normpath(path).split(os.sep)
    # If path is absolute, first path needs a hard fix
    if all_paths[0] == '':
        all_paths[0] = "/"
    all_paths = [os.path.join(*all_paths[:i+1]) for i in range(len(all_paths))]
    for sub_path in all_paths:
        if not os.path.isdir(sub_path):
            os.mkdir(sub_path)

def plot_output(image, target, outputs, dest_path=None):
    """Takes a triplet image-target-output and makes a subplot.
    Optionally takes a path to save; if path is None then shows.
    Images can be Tensors - they will be converted."""
    # Converting from tensors if needed
    if type(image) == torch.Tensor:
        image = image.detach().cpu().numpy()
    if type(target) == torch.Tensor:
        target = target.detach().cpu().numpy()
    if type(outputs) == torch.Tensor:
        outputs = outputs.detach().cpu().numpy()
    
    # Squeezing single-channel image if needed
    if image.shape[0] == 1:
        image = image.squeeze(axis=0)

    num_classes = outputs.shape[0]

    # Preparing the figure
    plt.figure(figsize=[6*(num_classes+3), 6])

    # Plotting the image
    plt.subplot(1, num_classes+3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    # Plotting the labelmap/target
    plt.subplot(1, num_classes+3, 2)
    plt.imshow(target, cmap="tab10", vmax=9)
    plt.title("Target")
    plt.axis("off")

    # Plotting the output
    plt.subplot(1, num_classes+3, 3)
    plt.imshow(np.argmax(outputs, axis=0), cmap="tab10", vmax=9)
    plt.title("Output")
    plt.axis("off")

    # Plotting each output probability map
    for i, output in enumerate(outputs):
        subplot_idx = i + 3 + 1
        plt.subplot(1, num_classes+3, subplot_idx)

        class_color = plt.get_cmap("tab10")(i)
        prob_map = np.full((*output.shape, 4), class_color)
        prob_map[:,:,3] = output  # Setting transparency as probability

        plt.imshow(prob_map)
        plt.title("Class {} probability".format(i))
        plt.axis("off")
    
    # Saving or showing
    plt.tight_layout(pad=1.5)
    if dest_path == None:
        plt.show()
    else:
        plt.savefig(dest_path)
        plt.clf()
        plt.close()


def bbox_to_plot(bbox, image_dimensions):
    """Takes a bbox (center_x, center_y, width, height) and outputs ((topleft_x, topleft_y), width, height),
    while also converting to image size"""
    image_dimensions = np.concatenate([image_dimensions, image_dimensions])
    bbox = bbox * image_dimensions
    return [(bbox[1] - (bbox[3]//2), bbox[0] - (bbox[2]//2)), bbox[3], bbox[2]]


def plot_output_det(image, target, outputs, dest_path=None):
    """Plots bounding box outputs"""
    # Converting from tensors if needed
    if type(image) == torch.Tensor:
        image = image.detach().cpu().numpy()
    if type(target) == torch.Tensor:
        target = target.detach().cpu().numpy()
    if type(outputs) == torch.Tensor:
        outputs = outputs.detach().cpu().numpy()
    
    # Squeezing single-channel image if needed
    if image.shape[0] == 1:
        image = image.squeeze(axis=0)

    num_classes = outputs.shape[0]

    # Preparing the figure
    plt.figure(figsize=[6*(2), 6])

    # Plotting the image
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Image")
    plt.axis("off")

    # Plotting the labelmap/target
    plt.subplot(1, 3, 2)
    plt.imshow(image, cmap="gray")
    for _class in range(num_classes):
        rect = Rectangle(*bbox_to_plot(target[_class], image.shape), fill=False, ec=plt.get_cmap("tab10")(_class+1))
        plt.gca().add_patch(rect)
    plt.title("Target")
    plt.axis("off")

    # Plotting the output
    plt.subplot(1, 3, 3)
    plt.imshow(image, cmap="gray")
    for _class in range(num_classes):
        rect = Rectangle(*bbox_to_plot(outputs[_class], image.shape), fill=False, ec=plt.get_cmap("tab10")(_class+1))
        plt.gca().add_patch(rect)
    plt.title("Output")
    plt.axis("off")
    
    # Saving or showing
    plt.tight_layout(pad=1.5)
    if dest_path == None:
        plt.show()
    else:
        plt.savefig(dest_path)
        plt.clf()
        plt.close()
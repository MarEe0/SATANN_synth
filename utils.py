"""Collection of utility functions."""
import os
import math

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
    if type(tensors[0]) is torch.Tensor:
        if len(tensors) <= 1:
            return tensors[0]
        return torch.logical_or(tensors[0], multi_logical_or(tensors[1:]))
    if type(tensors[0]) is np.ndarray:
        if len(tensors) <= 1:
            return tensors[0]
        return np.logical_or(tensors[0], multi_logical_or(tensors[1:]))


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

def create_relational_kernel(distance, angle, distance_slack=10, aperture=math.pi/12):
    """Creates a relational kernel for a given distance and relation angle with specified aperture.

    NOTE: since kernels are square, directions that are not square (i.e. multiples of `pi/2`) will not be fully
    represented.

    Parameters
    ----------
    distance : int
        Distance in pixels between objects. Kernel size will be `distance*2 + distance_slack`.
    angle : float in {0, 2*pi}
        Angle, in radians, of the relation. `0` is the "to the right of" relation; `pi` is "to the left of".
    aperture : float in {0, 2*pi}
        Angle, in radians, of the relation aperture. An aperture of `2*pi` has no directional meaning. Default: `pi/12`.
    distance_slack : int
        Slack to be given to the distance. Typically, should be the expected maximal value of the distance from the
        target object's center to its furthermost point from the source (so that the relational map can contain the
        full target object).

    Returns
    -------
    kernel : Tensor
        The convolutional kernel that encodes the desired relation.
    """
    kernel_size = distance*2+distance_slack
    if kernel_size % 2 == 0: kernel_size += 1  # Kernel size ought to be odd
    kernel_size = int(kernel_size)
    kernel = torch.zeros((kernel_size,kernel_size))

    # angle += math.pi # Angle correction for coordinate set
    if angle > 2*math.pi: angle -= 2*math.pi  # angle rollover

    epsilon = 10e-7

    for i in range(kernel_size):
        for j in range(kernel_size):
            # Converting to -1,1 coordinate space
            x,y = (2*i)/(kernel_size)-1, (2*j)/(kernel_size)-1
            # Computing intensity of this pixel
            angle_of_point = math.atan(y/(x+epsilon))  # Getting angle
            if x < 0: angle_of_point += math.pi  # Fixing negative half-circle
            angle_of_point += math.pi/2  # Fixing mysterious offset
            if angle_of_point < 0: angle_of_point = 0  # Fixing mysterious offset outliers

            # Computing distance of angle of point to target angle, both ways
            distance = math.pi - abs(abs(angle_of_point - angle) - math.pi)

            # Computing pixel intensity with a fall-off for the aperture
            intensity = 1 - ((2*distance) / aperture)
            if intensity < 0: intensity = 0
            #print("x: {:.2f}, y: {:.2f}, angle: {:.2f} ({:.2f}º)".format(x,y,math.atan(y/(x+epsilon)), math.degrees(math.atan(y/(x+epsilon)))))
            #print("\tdistance: {:.2f}, 2*k/phi: {:.2f}".format(distance, (distance) / aperture))

            kernel[i,j] = intensity

    kernel[kernel_size//2,kernel_size//2] = 1
    return kernel
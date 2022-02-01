"""Cloud of Structured Objects PyTorch Dataset class.

This file contains the class for generating a PyTorch dataset of CloStOb data.

Authors
-------
 * Mateus Riva (mateus.riva@telecom-paris.fr)"""
#%%
import os
import numpy as np
from torch.utils.data import Dataset
from skimage.transform import rescale

def load_dataset(base_dataset_name):
    """Loads one of the prepared base datasets for CloStOb.

    NOTE: this currently only supports Fashion-MNIST (as "fashion").

    :param base_dataset_name: name of the base dataset folder
    :return: a dict of image lists, keyed by classes
    """
    if os.path.split(base_dataset_name)[1] != "fashion": raise NotImplementedError("Only 'fashion' dataset has been implemented")

    base_filepath = os.path.join(os.path.dirname(__file__), base_dataset_name)

    with open(os.path.join(base_filepath, "labels"), "rb") as labels_file:
        labels = np.frombuffer(labels_file.read(), dtype=np.uint8, offset=8)

    with open(os.path.join(base_filepath, "images"), "rb") as images_file:
        images = np.frombuffer(images_file.read(), dtype=np.uint8, offset=16).reshape(len(labels), 28, 28)

    base_dataset = {}
    for image, label in zip(images, labels):
        if label not in base_dataset:
            base_dataset[label] = [image]
        else:
            base_dataset[label].append(image)
    return base_dataset


def generate_image(seed, base_dataset, image_dimensions: tuple, fg_classes: list, fg_positions: list,
                   position_translation: float, position_noise: float, rescale_classes: list, 
                   rescale_range: tuple, occlusion_classes: list, occlusion_range: tuple, 
                   bg_classes: list, bg_amount: float, fine_segment: bool, flattened: bool):
    """Generates a single CloStOb image and corresponding label map.

    :param seed: Random number generator seed for this image.
    :param base_dataset: loaded base dataset.
    :param image_dimensions: dimensions of dataset images.
    :param fg_classes: list of class labels to be considered foreground.
    :param fg_positions: list of normalised coordinates for distributing each foreground element, with origin on the
    top left. E.g. (0,0) is the top left of the image, (0.5,0) is top center, (1,1) is bottom right.
    :param position_translation: maximal grouped foreground translation, as a percentage.
    :param position_noise: maximal individual foreground positional noise, as a percentage.
    :param rescale_classes: which foreground classes to receive rescale transformations.
    :param rescale_range: the range upon which to sample the random scale.
    :param occlusion_classes: which foreground classes to receive occlusion transformations.
    :param occlusion_ranges: min and max size of occlusion square.
    :param bg_classes: list of class labels to be added to the background.
    :param bg_amount: number (or range of numbers) of random images to draw on the background
    :param fine_segment: If True, labelmaps are cut to the positive part of images.
    :param flattened: If True, return images as flattened 1D arrays. Else, return images shaped like `size`.
    :return: a tuple (image, labelmap) containing the image and corresponding labelmap.
    """
    # Initialising RNG with specified seed
    rng = np.random.default_rng(seed)

    # Creating empty base image and labelmap
    image, labelmap = np.zeros(image_dimensions, dtype="float32"), np.zeros(image_dimensions, dtype=int)
    # Getting shape of base dataset images
    base_shape = base_dataset[bg_classes[0]][0].shape
    # Initialising limits on coordinates to avoid images "leaking out the border"
    coordinates_limit = tuple(np.array(image_dimensions) - base_shape)

    # Distributing background images
    for bg_element in rng.choice(
            [item for sublist in [base_dataset[bg_class] for bg_class in bg_classes] for item in sublist], bg_amount):
        # Choosing random coordinates
        bg_origin_coords = rng.integers(np.zeros_like(image_dimensions), coordinates_limit)
        bg_element_coords = tuple(
            np.s_[origin:end] for origin, end in zip(bg_origin_coords, bg_origin_coords + base_shape))
        # Adding bg element
        image[bg_element_coords] = bg_element

    # Preparing fg coordinates
    fg_positions = np.array(fg_positions)
    # Global translation
    fg_positions += rng.uniform(low=-position_translation / 2, high=position_translation / 2,
                                size=fg_positions.shape[1:])
    # Individual translations (structural noise)
    fg_positions += rng.uniform(low=-position_noise / 2, high=position_noise / 2, size=fg_positions.shape)
    # Converting to real pixel coordinates
    fg_origin_coords_set = (fg_positions * image_dimensions - np.array(base_shape) // 2).astype(int)

    # Distributing fg images
    to_rescale = {fg_class : fg_class in rescale_classes for fg_class in fg_classes}
    to_occlude = {fg_class : fg_class in occlusion_classes for fg_class in fg_classes}
    for idx, pack in enumerate(zip(fg_classes, fg_origin_coords_set)):
        fg_class, fg_origin_coords = pack
        fg_element = rng.choice(base_dataset[fg_class])

        # Applying resizing
        if to_rescale[fg_class]:
            fg_element = rescale(fg_element, scale=rng.uniform(low=rescale_range[0], high=rescale_range[1]), preserve_range=True)
        
        # Applying occlusion:
        if to_occlude[fg_class]:
            occlusion_size = rng.integers(low=occlusion_range[0], high=occlusion_range[1])
            occlusion_point_x = rng.integers(low=0, high=fg_element.shape[0]-occlusion_size)
            occlusion_point_y = rng.integers(low=0, high=fg_element.shape[1]-occlusion_size)
            fg_element[occlusion_point_x:occlusion_point_x+occlusion_size, occlusion_point_y:occlusion_point_y+occlusion_size] = 0

        fg_element_coords = tuple(
            np.s_[origin:end] for origin, end in zip(fg_origin_coords, fg_origin_coords + fg_element.shape))
        # Adding fg element
        image[fg_element_coords] = fg_element
        # Adding labelmap element        
        map_element = np.full(fg_element.shape, idx+1)
        if fine_segment:  # If the labelmap should cut out the zero part
            map_element[fg_element == 0] = 0
        labelmap[fg_element_coords] = map_element

    # Flattening image if necessary
    if flattened:
        image = image.flatten()

    return {"image": image, "labelmap": labelmap}


class CloStObDataset(Dataset):
    def __init__(self, base_dataset_name: str, image_dimensions: tuple, size: int, fg_classes: list, fg_positions: list,
                 bg_classes: list, bg_amount: float, position_translation: float = 0.0, position_noise: float = 0.0,
                 rescale_classes: list = [], rescale_range: tuple = (1,1), occlusion_classes: list = [], occlusion_range: tuple = (0,0),
                 fine_segment: bool = False,
                 flattened: bool = False, lazy_load: bool = False, transform = None, target_transform = None, start_seed: int = 0):
        """The constructor for CloStObDataset class.

        :param base_dataset_name: name of the base dataset folder.
        :param image_dimensions: dimensions of dataset images.
        :param size: size of the dataset to be generated.
        :param fg_classes: list of class labels to be considered foreground.
        :param fg_positions: list of normalised coordinates for distributing each foreground element, with origin on the top left. E.g. (0,0) is the top left of the image, (0.5,0) is top center, (1,1) is bottom right.
        :param position_translation: maximal grouped foreground translation, as a percentage. Default: 0.0.
        :param position_noise: maximal individual foreground positional noise, as a percentage. Default: 0.0.
        :param rescale_classes: which foreground classes to receive rescale transformations.
        :param rescale_range: the range upon which to sample the random scale.
        :param occlusion_classes: which foreground classes to receive occlusion transformations.
        :param occlusion_ranges: min and max size of occlusion square.
        :param bg_classes: list of class labels to be added to the background.
        :param bg_amount: number (or range of numbers) of random images to draw on the background.
        :param fine_segment: If True, labelmaps are cut to the positive part of images.
        :param flattened: If True, return images as flattened 1D arrays. Else, return images shaped like `image_dimensions`. Default: False.
        :param lazy_load: If True, generate images on-the-fly. Else, generate and store all images in memory on initialization.
        :param transform: (optional) callable/transform to be applied to each image.
        :param target_transform: (optional) callable/transform to be applied to each labelmap.
        :param start_seed: seed to offset by.
        """
        # Sanity checking on parameters
        assert len(image_dimensions) == 2, "Only 2D images are currently supported"
        assert len(fg_classes) == len(fg_positions), "Length of fg_classes ({}) and fg_positions ({}) mismatch".format(
            len(fg_classes), len(fg_positions))

        # Loading base dataset
        self.base_dataset = load_dataset(base_dataset_name)

        # Setting flags and inner attributes
        self.lazy_load = lazy_load
        self.transform = transform
        self.target_transform = target_transform
        self.number_of_classes = 1 + len(fg_classes)
        self.image_dimensions = image_dimensions
        self.fg_classes = fg_classes
        self.fg_positions = fg_positions
        self.position_translation = position_translation
        self.position_noise = position_noise
        self.rescale_classes = rescale_classes
        self.rescale_range = rescale_range
        self.occlusion_classes = occlusion_classes
        self.occlusion_range = occlusion_range
        self.bg_classes = bg_classes
        self.bg_amount = bg_amount
        self.flattened = flattened
        self.fine_segment = fine_segment
        self.start_seed = 0

        # If preloading, generate a list of CloStOb images and apply transforms
        if not self.lazy_load:
            self.samples = [self.apply_transforms(generate_image(i, self.base_dataset, self.image_dimensions, self.fg_classes, self.fg_positions, self.position_translation, self.position_noise, self.rescale_classes, self.rescale_range, self.occlusion_classes, self.occlusion_range, self.bg_classes, self.bg_amount, self.fine_segment, self.flattened)) for i in range(size)]

        self.size = size

    def __getitem__(self, idx):
        idx = idx + self.start_seed
        if not self.lazy_load:
            sample = self.samples[idx]
        else:
            sample = generate_image(idx, self.base_dataset, self.image_dimensions, self.fg_classes, self.fg_positions, self.position_translation, self.position_noise, self.rescale_classes, self.rescale_range, self.occlusion_classes, self.occlusion_range, self.bg_classes, self.bg_amount, self.fine_segment, self.flattened)
            sample = self.apply_transforms(sample)
        return sample

    def __len__(self):
        return self.size

    def apply_transforms(self, sample):
        """ Applies relevant transforms to sample.

        :param sample: dict containing a single sample's "image" and "labelmap".
        :return: the sample, after in-place transformations.
        """
        if self.transform is not None:
            sample["image"] = self.transform(sample["image"])
        if self.target_transform is not None:
            sample["labelmap"] = self.target_transform(sample["labelmap"])
        return sample

#%%
if __name__ == '__main__':
    clostob = CloStObDataset(base_dataset_name="fashion",
                             image_dimensions=(200, 200),
                             size=100,
                             fg_classes=[0, 1],
                             fg_positions=[(0.5, 0.25), (0.5, 0.75)],
                             #rescale_classes=[0],
                             #rescale_range=(2.0,2.0),
                             #occlusion_classes=[0],
                             #occlusion_range = [5,15],
                             bg_classes=[5, 7],
                             bg_amount=3,
                             position_translation=0.2,
                             position_noise=0.1,
                             fine_segment=True,
                             flattened=False)

    import matplotlib.pyplot as plt

    for i in range(3):
        plt.subplot(121)
        plt.imshow(clostob[i]["image"], cmap="gray")
        plt.subplot(122)
        plt.imshow(clostob[i]["labelmap"])
        plt.show()

# %%

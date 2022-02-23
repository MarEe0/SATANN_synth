"""Cloud of Structured Objects PyTorch Dataset class.

This file contains the class for generating a PyTorch dataset of CloStOb data.

Authors
-------
 * Mateus Riva (mateus.riva@telecom-paris.fr)"""
#%%
import os
import numpy as np
import sys
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
                   bg_classes: list, bg_amount: float, bg_bbox: tuple, fine_segment: bool, flattened: bool):
    """Generates a single CloStOb image and corresponding label map and bounding boxes.

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
    :param bg_bbox: normalised limit coordinates for the bg images, as (x_min,y_min,x_max,y_max)
    :param fine_segment: If True, labelmaps are cut to the positive part of images.
    :param flattened: If True, return images as flattened 1D arrays. Else, return images shaped like `size`.
    :return: a tuple (image, labelmap) containing the image and corresponding labelmap.
    """
    # Initialising RNG with specified seed
    rng = np.random.default_rng(seed)

    # Creating empty base image and labelmap
    image, labelmap = np.zeros(image_dimensions, dtype="float32"), np.zeros(image_dimensions, dtype=int)
    # Creating empty bounding boxes list - one (x,y,w,h) tuple for each fg_class
    bboxes = np.zeros((len(fg_classes), 4))
    # Creating background labelmap
    bg_labelmap = np.ones(image_dimensions, dtype=int)*-1
    # Getting shape of base dataset images
    base_shape = base_dataset[bg_classes[0]][0].shape
    # Initialising limits on coordinates to avoid images "leaking out the border"
    coordinates_limit = tuple(np.array(image_dimensions) - base_shape)
    # Binding the coordinate limits to the bg_bbox
    bg_coordinates_limit = (max(0, bg_bbox[0]*image_dimensions[0]), max(0, bg_bbox[1]*image_dimensions[1]),
                            min(coordinates_limit[0], bg_bbox[2]*image_dimensions[0]), min(coordinates_limit[1], bg_bbox[3]*image_dimensions[1]))

    # Preparing list of potential background elements
    bg_chosen_classes = rng.choice(bg_classes, bg_amount, replace=True)
    bg_elements = [(rng.choice(base_dataset[bg_class]), bg_class) for bg_class in bg_chosen_classes]
    # Distributing background images
    for idx, bg_element in enumerate(bg_elements):
        # Choosing random coordinates
        bg_origin_coords = rng.integers(low=bg_coordinates_limit[:2], high=bg_coordinates_limit[2:], size=len(image_dimensions))
        bg_element_coords = tuple(
            np.s_[origin:end] for origin, end in zip(bg_origin_coords, bg_origin_coords + base_shape))
        # Adding bg element
        image[bg_element_coords] = bg_element[0]
        # Adding to background labelmap
        bg_map_element = np.full(bg_element[0].shape, bg_element[1])
        if fine_segment:  # If the labelmap should cut out the zero part
            bg_map_element[bg_element[0] == 0] = 0
        bg_labelmap[bg_element_coords] = bg_map_element

    # Preparing fg coordinates
    fg_positions = np.array(fg_positions)
    # Global translation
    fg_positions += rng.uniform(low=-position_translation / 2, high=position_translation / 2,
                                size=fg_positions.shape[1:])
    # Individual translations (structural noise)
    fg_positions += rng.uniform(low=-position_noise / 2, high=position_noise / 2, size=fg_positions.shape)
    # Converting to real pixel coordinates
    fg_origin_coords_set = (fg_positions * image_dimensions - np.array(base_shape) // 2).astype(int)
    fg_origin_coords_set[fg_origin_coords_set<0] = 0  # Guaranteeing no underflow
    fg_origin_coords_set[:,0][fg_origin_coords_set[:,0]>coordinates_limit[0]] = coordinates_limit[0]  # Guaranteeing no overflow
    fg_origin_coords_set[:,1][fg_origin_coords_set[:,1]>coordinates_limit[1]] = coordinates_limit[1]  # Guaranteeing no overflow

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
        # Adding bounding box element
        bboxes[idx] = np.divide([*(fg_origin_coords + np.floor_divide(fg_element.shape,2)), *fg_element.shape], [*image_dimensions,*image_dimensions])

    # Flattening image if necessary
    if flattened:
        image = image.flatten()

    return {"image": image, "labelmap": labelmap, "bboxes": bboxes, "bg_labelmap": bg_labelmap}


class CloStObDataset(Dataset):
    def __init__(self, base_dataset_name: str, image_dimensions: tuple, size: int, fg_classes: list, fg_positions: list,
                 bg_classes: list, bg_amount: float, bg_bboxes: tuple = None, position_translation: float = 0.0, position_noise: float = 0.0,
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
        :param bg_bbox: normalised limit coordinates for the bg images, as (x_min,y_min,x_max,y_max). If None, image_dimensions is used.
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
        if bg_bboxes is None:
            self.bg_bboxes=(0,0,*self.image_dimensions)
        else:
            self.bg_bboxes = bg_bboxes
        self.flattened = flattened
        self.fine_segment = fine_segment
        self.start_seed = 0

        # If preloading, generate a list of CloStOb images and apply transforms
        if not self.lazy_load:
            self.samples = [self.apply_transforms(generate_image(i, self.base_dataset, self.image_dimensions, self.fg_classes, self.fg_positions, self.position_translation, self.position_noise, self.rescale_classes, self.rescale_range, self.occlusion_classes, self.occlusion_range, self.bg_classes, self.bg_amount, self.bg_bboxes, self.fine_segment, self.flattened)) for i in range(size)]

        self.size = size

    def __getitem__(self, idx):
        idx = idx + self.start_seed
        if not self.lazy_load:
            sample = self.samples[idx]
        else:
            sample = generate_image(idx, self.base_dataset, self.image_dimensions, self.fg_classes, self.fg_positions, self.position_translation, self.position_noise, self.rescale_classes, self.rescale_range, self.occlusion_classes, self.occlusion_range, self.bg_classes, self.bg_amount, self.bg_bboxes, self.fine_segment, self.flattened)
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
    
    def generate_meaningless_image(self, _class):
        raise NotImplementedError("Still not implemented")
    
    def generate_reference_shifts(self, idx, reference_class, stride=32, element_shape=(28,28)):
        """Generates a set of images with shifted references."""
        set_of_shifts = []
        shifted_fg_positions = self.fg_positions
        idx_to_shift = self.fg_classes.index(reference_class)
        normed_element_shape = ((element_shape[0]//2)/self.image_dimensions[0],
                                (element_shape[1]//2)/self.image_dimensions[1])
        
        set_of_anchors = [(x,y) for x in range(0, self.image_dimensions[0]-element_shape[0], stride) for y in range(0, self.image_dimensions[1]-element_shape[1], stride)]
        for x_anchor_base, y_anchor_base in set_of_anchors:
                # Putting the reference in its proper place
                x_anchor, y_anchor = x_anchor_base/self.image_dimensions[0], y_anchor_base/self.image_dimensions[1]
                shifted_fg_positions[idx_to_shift] = (x_anchor + normed_element_shape[0], y_anchor + normed_element_shape[1])

                sample = generate_image(idx, self.base_dataset, self.image_dimensions, self.fg_classes, shifted_fg_positions, 0, 0, self.rescale_classes, self.rescale_range, self.occlusion_classes, self.occlusion_range, self.bg_classes, self.bg_amount, self.bg_bboxes, self.fine_segment, self.flattened)
                sample = self.apply_transforms(sample)
                set_of_shifts.append(sample)
        return set_of_shifts, set_of_anchors

#%%
if __name__ == '__main__':
    test_set_size = 20
    element_shape = (28,28)
    stride = 28

    # Preparing the foreground
    fg_label = "T"
    fg_classes = [0, 1, 8]
    base_fg_positions = [(0.65, 0.25), (0.65, 0.65), (0.35, 0.65)]
    position_translation=0.25
    position_noise=0.1

    num_classes = len(fg_classes)
    classes = range(1,num_classes+1)

    # Also setting the image dimensions in advance
    image_dimensions = [160, 160]
    

    # Preparing dataset transforms:
    
    sys.path.append("/home/mriva/Recherche/PhD/SATANN/SATANN_synth")
    import torchvision as tv
    from utils import targetToTensor
    transform = tv.transforms.Compose(                                  # For the images:
        [tv.transforms.ToTensor(),                                      # Convert to torch.Tensor CxHxW type
        tv.transforms.Normalize((255/2,), (255/2,), inplace=True)])    # Normalize from [0,255] to [-1,1] range
    target_transform = tv.transforms.Compose(                           # For the labelmaps:
        [targetToTensor()])                                             # Convert to torch.Tensor type
    test_dataset = CloStObDataset(base_dataset_name="fashion",
                                            image_dimensions=image_dimensions,
                                            size=test_set_size,
                                            fg_classes=fg_classes,
                                            fg_positions=base_fg_positions,
                                            position_translation=0.25,
                                            position_noise=0.1,
                                            bg_classes=[0], # Background class from config
                                            bg_amount=3,
                                            #bg_bboxes=(0.4, 0.0, 0.9, 0.5),
                                            flattened=False,
                                            lazy_load=False,
                                            fine_segment=True,
                                            transform=transform,
                                            target_transform=target_transform,
                                            start_seed=100000)

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from skimage.segmentation import mark_boundaries

    #shifts, anchors = test_dataset.generate_reference_shifts(11, 0, 32)
    #print(len(shifts))
    #for i, shift in enumerate([shifts[0], shifts[1], shifts[42]]):
    #    plt.subplot(121)
    #    plt.title("Input")
    #    plt.imshow(shift["image"][0], cmap="gray")
    #    plt.subplot(122)
    #    plt.title("GT")
    #    plt.imshow(shift["labelmap"])
    #    plt.savefig("./work{}.png".format(i+10))
    #    plt.clf()

    for i in range(10):
        #plt.subplot(131)
        image = (test_dataset[i]["image"][0] + 1.0)/2.0
        labelmap = test_dataset[i]["labelmap"]
        for label, color in zip(range(1,4), [plt.get_cmap("tab10")(1)[:3], plt.get_cmap("tab10")(2)[:3], plt.get_cmap("tab10")(3)[:3]]):
            image = mark_boundaries(image, labelmap==label, color=color, mode="thick", background_label=0)
        plt.tight_layout()
        plt.axis("off")
        plt.imshow(image, cmap="gray")
        plt.savefig("./hard{}.eps".format(i), bbox_inches='tight')
        plt.savefig("./hard{}.png".format(i), bbox_inches='tight')
        #plt.subplot(132)
        #plt.imshow(test_dataset[i]["labelmap"])
        #for bbox in test_dataset[i]["bboxes"]:
        #    bbox = bbox*200
        #    plt.scatter([bbox[1]],[bbox[0]])
        #    rect = patches.Rectangle((bbox[:2][::-1] - np.floor_divide(*bbox[2:])), bbox[2], bbox[3])
        #    plt.gca().add_patch(rect)
        #plt.subplot(133)
        #plt.imshow(test_dataset[i]["bg_labelmap"])
        #print(test_dataset[i]["bboxes"])

# %%

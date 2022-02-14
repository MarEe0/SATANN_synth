"""Script for computing CAM with object overlaps."""
import os, sys
from glob import glob

import numpy as np
import skimage.filters

import torch
from torch.nn.functional import softmax
import torchvision as tv

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

sys.path.append("/home/mriva/Recherche/PhD/SATANN/SATANN_synth")
from datasets.clostob.clostob_dataset import CloStObDataset
from unet import UNet
from metrics import precision, recall
from utils import targetToTensor

def label_to_name(label):
    if "veryhard" in label:
        return "{}-V.H.".format(label[0].upper())
    if "hard" in label:
        return "{}-Hard".format(label[0].upper())
    if "easy" in label:
        return "{}-Easy".format(label[0].upper())


# Needed for GradCAM with semantic segmentation
class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()


if __name__ == "__main__":
    base_path = "/media/mriva/LaCie/SATANN/synthetic_fine_segmentation_results/results_seg"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Iterating over each dataset size
    for dataset_size in [400]:
        base_dataset_path = os.path.join(base_path, "dataset_{}".format(dataset_size))

        test_set_size = 100

        # Preparing the foreground
        fg_label = "T"
        fg_classes = [0, 1, 8]
        base_fg_positions = [(0.65, 0.3), (0.65, 0.7), (0.35, 0.7)]
        position_translation=0.2
        position_noise=0.1

        num_classes = len(fg_classes)
        classes = range(1,num_classes+1)

        # Also setting the image dimensions in advance
        image_dimensions = [256, 256]
        

        # Preparing dataset transforms:
        transform = tv.transforms.Compose(                                  # For the images:
            [tv.transforms.ToTensor(),                                      # Convert to torch.Tensor CxHxW type
            tv.transforms.Normalize((255/2,), (255/2,), inplace=True)])    # Normalize from [0,255] to [-1,1] range
        target_transform = tv.transforms.Compose(                           # For the labelmaps:
            [targetToTensor()])                                             # Convert to torch.Tensor type

        # Experiment configurations
        experimental_configs = [{"label": fg_label + "_hard_noise", "bg_classes": [0], "bg_amount": 3},
                                {"label": fg_label + "_easy_noise", "bg_classes": [7], "bg_amount": 3},
                                {"label": fg_label + "_veryhard_noise", "bg_classes": [0,1,8], "bg_amount": 6}]
        
        # Getting results for a specific experiment configuration
        for experimental_config in experimental_configs:
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
                                            flattened=False,
                                            lazy_load=False,
                                            fine_segment=True,
                                            transform=transform,
                                            target_transform=target_transform,
                                            start_seed=100000)
            
            # CAM OVERLAP
            #  Getting overlap counts at test-time per class for all inits
            overlap_counts = {_class : {_class : {"oi" : [], "noi": []} for _class in classes} for _class in classes}
            for initialization_path in glob(os.path.join(base_dataset_path, experimental_config["label"] + "*")):
                # skipping SATANN examples (where alpha > 0)
                if initialization_path[-2:] == ".5": continue
                model_label = os.path.split(initialization_path)[-1]

                # Loading the specified model
                model_path = os.path.join(initialization_path, "best_model.pth")
                model = UNet(input_channels=1, output_channels=4).to(device=device)
                model.load_state_dict(torch.load(model_path))
                model.eval()

                # Running the model on the test data
                for i, sample in enumerate(test_dataset):
                    input = sample["image"].unsqueeze(0).to(device="cuda")
                    with torch.set_grad_enabled(False):
                        output = model(input).detach().cpu()
                    oi_labelmap = sample["labelmap"]
                    bg_labelmap = sample["bg_labelmap"]

                    output_softmax = softmax(output, dim=1)  # Softmax outputs along class dimension
                    output_argmax = output_softmax.argmax(dim=1).detach().cpu().numpy()  # Argmax outputs along class dimension
                    
                    # Getting all classes
                    for target_category in [1,2,3]:
                        # Doing GradCAM magic
                        #target_layers = [model.dconv_down4, model.dconv_up3, model.dconv_up2, model.dconv_up1, model.conv_last]
                        target_layers = [model.dconv_down3, model.dconv_down4, model.dconv_up3, model.dconv_up2, model.dconv_up1, model.conv_last]
                        #target_layers = [model.dconv_up1, model.conv_last]
                        target_mask = np.float32(output_argmax == target_category)
                        targets = [SemanticSegmentationTarget(target_category, target_mask)]

                        # Computing the CAM
                        with GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available()) as cam:
                            grayscale_cam = cam(input_tensor=input, targets=targets)[0, :]

                            # Remove cam border artifacts
                            grayscale_cam[:32,:] = 0
                            grayscale_cam[:,:32] = 0
                            grayscale_cam[-32:,:] = 0
                            grayscale_cam[:,-32:] = 0

                            # Yen thresholding
                            threshold_cam = grayscale_cam.copy()
                            threshold_cam[grayscale_cam < skimage.filters.threshold_yen(grayscale_cam)] = 0

                            # Masking the fg labelmap with the activations
                            masked_oi_labelmap = oi_labelmap * np.greater(grayscale_cam, 0)

                            if target_category == 1:
                                import matplotlib.pyplot as plt
                                from skimage.color import label2rgb
                                from skimage.segmentation import mark_boundaries
                                from utils import mkdir
                                mkdir("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/cam_overlap/{}/".format(model_label))
                                rgb_image = ((np.repeat(input.detach().cpu().numpy().squeeze()[...,None],3,axis=2) + 1) / 2).astype(np.float32)
                                rgb_image = mark_boundaries(rgb_image, oi_labelmap.detach().cpu().numpy(), color=(1,0,1), mode="thick", background_label=0)
                                cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
                                plt.imshow(cam_image)
                                plt.savefig("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/cam_overlap/{}/cam_{}.png".format(model_label, i))
                                plt.clf()
                                thresh_image = show_cam_on_image(rgb_image, threshold_cam, use_rgb=True)
                                plt.imshow(thresh_image)
                                plt.savefig("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/cam_overlap/{}/thresh_cam_{}.png".format(model_label, i))
                                #plt.subplot(121); plt.imshow(input[0][0].detach().cpu().numpy(), cmap="gray")
                                #plt.subplot(122); plt.imshow(grayscale_cam)
                                #plt.clf()
                                #plt.imshow(oi_labelmap * np.greater(grayscale_cam, 0))
                                #plt.savefig("/home/mriva/Recherche/PhD/SATANN/SATANN_synth/tests/cam_overlap/oi{}.png".format(target_category))
                    #sys.exit()
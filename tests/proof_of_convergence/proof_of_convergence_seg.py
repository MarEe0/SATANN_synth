"""Script for producing proof of convergence tables from trained models."""
import os, sys
from glob import glob

import torch
from torch.nn.functional import softmax
import torchvision as tv

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

if __name__ == "__main__":
    base_path = "/media/mriva/LaCie/SATANN/synthetic_fine_segmentation_results/results_seg"
    # Iterating over each dataset size
    for dataset_size in [100, 1000, 10000]:
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
        crit_classes = [1]
        model_output_channels = 4 if base_path[-3:] == "seg" else 2

        # Also setting the image dimensions in advance
        image_dimensions = [256, 256]
        

        # Preparing dataset transforms:
        transform = tv.transforms.Compose(                                  # For the images:
            [tv.transforms.ToTensor(),                                      # Convert to torch.Tensor CxHxW type
            tv.transforms.Normalize((255/2,), (255/2,), inplace=True)])    # Normalize from [0,255] to [-1,1] range
        target_transform = tv.transforms.Compose(                           # For the labelmaps:
            [targetToTensor()])                                             # Convert to torch.Tensor type

        # Experiment configurations
        experimental_configs = [{"label": fg_label + "_easy_noise", "bg_classes": [7], "bg_amount": 3},
                                {"label": fg_label + "_hard_noise", "bg_classes": [0], "bg_amount": 3}]
        #                        {"label": fg_label + "_veryhard_noise", "bg_classes": [0,1,8], "bg_amount": 6}]
        
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

            # PROOF OF CONVERGENCE:
            #   Getting test-time precision and recall per class for all inits
            initialization_paths = glob(os.path.join(base_dataset_path, experimental_config["label"] + "*"))
            initialization_paths = [item for item in initialization_paths if item[-2:] != ".5"] # skipping SATANN examples (where alpha > 0)
            precisions = {_class : None for _class in classes}
            recalls = {_class : None for _class in classes}
            model_has_converged = [False for _ in range(len(initialization_paths))]
            for i, initialization_path in enumerate(initialization_paths):
                # skipping SATANN examples (where alpha > 0)
                if initialization_path[-2:] == ".5": continue

                # Preparing the data loader
                data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, num_workers=2)

                # Loading the specified model
                model_path = os.path.join(initialization_path, "best_model.pth")
                model = UNet(input_channels=1, output_channels=model_output_channels).to(device="cuda")
                model.load_state_dict(torch.load(model_path))
                model.eval()

                # Running the model on the test data
                outputs, truths = [], []
                for item_pair in data_loader:
                    inputs = item_pair["image"].to(device="cuda")
                    with torch.set_grad_enabled(False):
                        outputs.append(model(inputs).detach().cpu())
                    truths.append(item_pair["labelmap"])
                
                outputs = torch.cat(outputs, dim=0)
                truths = torch.cat(truths, dim=0)

                outputs_softmax = softmax(outputs, dim=1)  # Softmax outputs along class dimension
                outputs_argmax = outputs_softmax.argmax(dim=1)  # Argmax outputs along class dimension

                # computing metrics for all classes
                for _class in crit_classes:
                    class_precisions = precision(outputs_argmax, truths, _class)
                    class_recalls = recall(outputs_argmax, truths, _class)

                    # Checking convergence
                    model_has_converged[i] = torch.mean(class_precisions).item() > 0.5 and torch.mean(class_recalls).item() > 0.5

                    # Adding to mean only if converged
                    if model_has_converged[i]:
                        if precisions[_class] is None:
                            precisions[_class] = class_precisions
                        else:
                            precisions[_class] = torch.cat([precisions[_class], class_precisions], dim=0)
                        if recalls[_class] is None:
                            recalls[_class] = class_recalls
                        else:
                            recalls[_class] = torch.cat([recalls[_class], class_recalls], dim=0)
                

                # Printing partial results
                model_label = os.path.split(initialization_path)[-1]
                print("D={}, {}, precision: ".format(dataset_size, model_label),end="")
                for _class in crit_classes:
                    class_precisions = precision(outputs_argmax, truths, _class)
                    print("C{} {:.3f} +- {:.3f}\t".format(_class, torch.mean(class_precisions).item(), torch.std(class_precisions).item()), end="")
                    if model_has_converged[i]: print("  *", end="")
                print("")
                print("D={}, {}, recall:    ".format(dataset_size, model_label),end="")
                for _class in crit_classes:
                    class_recalls = recall(outputs_argmax, truths, _class)
                    print("C{} {:.3f} +- {:.3f}\t".format(_class, torch.mean(class_recalls).item(), torch.std(class_recalls).item()),end="")
                    if model_has_converged[i]: print("  *", end="")
                print("\n")


            print("")
                
            # Printing results (latex format)
            # Config. | D. | Precision1 | Recall1 | (other classes?) | ConvergeRate
            print(label_to_name(experimental_config["label"]), end=" & ")
            print(dataset_size, end=" & ")
            for _class in crit_classes:
                mean_class_precision, std_class_precision = precisions[_class].mean().item(), precisions[_class].std().item()
                mean_class_recall, std_class_recall = recalls[_class].mean().item(), recalls[_class].std().item()
                print("${:.2} \pm {:.2}$".format(mean_class_precision, std_class_precision), end=" & ")
                print("${:.2} \pm {:.2}$".format(mean_class_recall, std_class_recall), end=" & ")
            print("{}/{}".format(sum(model_has_converged), len(model_has_converged)), end="")
            print("\\\\")
"""Spatial loss functions and classes.

Author
------
 * Mateus Riva (mateus.riva@telecom-paris.fr)
"""
from math import isnan

import torch
from torch import nn
from torch.nn.functional import conv2d

"""=================================================
            CENTRAL SPATIAL PRIOR ERROR
================================================="""

class SpatialPriorError(nn.Module):
    def __init__(self, relations):
        """Spatial relationship prior loss. Inherit this class and implement "forward()".

        Args:
            relations (list): List of spatial relationships in the format `(source, target, dy, dx)`
        """
        super(SpatialPriorError, self).__init__()

        self.relations = relations
    
    def compute_errors(self, centroids_y, centroids_x):
        """Computes the errors per coordinate for a given set of centroids.
        Centroids must have be of shape (B,C) where B is the batch size and C
        is the number of classes.
        """
        # Computing loss per relation
        if torch.cuda.is_available():
            dy_all = torch.empty(len(self.relations), centroids_y.size()[0], device="cuda")
            dx_all = torch.empty(len(self.relations), centroids_x.size()[0], device="cuda")
        else:
            dy_all = torch.empty(len(self.relations), centroids_y.size()[0])
            dx_all = torch.empty(len(self.relations), centroids_x.size()[0])

        for relation_index, relation in enumerate(self.relations):
            i, j, dy_gt, dx_gt = relation
            dy = centroids_y[:, i] - centroids_y[:, j]
            dx = centroids_x[:, i] - centroids_x[:, j]

            diff_y = dy - dy_gt
            diff_x = dx - dx_gt

            dy_error = torch.square(
                torch.nan_to_num(diff_y, nan=1, posinf=1, neginf=1))
            dx_error = torch.square(
                torch.nan_to_num(diff_x, nan=1, posinf=1, neginf=1))

            dy_all[relation_index] = dy_error
            dx_all[relation_index] = dx_error
        
        return dy_all, dx_all


def get_coordinates_map(image_dimensions, device="cpu"):
    if len(image_dimensions) == 2:  # 2-dimensional input (h x w)
        h, w = image_dimensions
    elif len(image_dimensions) == 4:  # 2-dimensional batch input (n x c x h x w)
        _, _, h, w = image_dimensions
    else:
        raise ValueError("Image dimensions has shape {}, only 2 and 4 are accepted".format(image_dimensions.shape))
    coordinates_map = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing="ij")
    coordinates_map = (coordinates_map[0].to(torch.device(device))/h, coordinates_map[1].to(torch.device(device))/w)  # normalizing
    return coordinates_map

class SpatialPriorErrorSegmentation(SpatialPriorError):
    def __init__(self, relations, image_dimensions=None, num_classes=None, crit_classes=None, device="cpu"):
        """Spatial prior loss for segmentation tasks.

        Args:
            relations (list): List of spatial relationships in the format `(source, target, dy, dx)`
            image_dimensions (tuple or None): shape of input images. If None, computed on-the-fly.
            crit_classes (list): classes being used in the criterion. If len(crit_classes) < num_classes,
                the complementary classes will be taken from the ground truth. Only used for metrics.
        """
        super(SpatialPriorErrorSegmentation, self).__init__(relations)

        if image_dimensions is not None:
            self.image_dimensions = image_dimensions
            self.coordinates_map = get_coordinates_map(image_dimensions, device)
        else:
            self.image_dimensions = None
            self.coordinates_map = None

        if num_classes:
            self.threshold = nn.Threshold(1.0/num_classes, 0)
        else:
            self.threshold = None
        
        self.crit_classes = crit_classes
        if crit_classes is not None:
            self.uncrit_classes = [x for x in range(num_classes+1) if x not in crit_classes]

    def compute_centroids(self, output):
        # Computing centroids
        if self.image_dimensions is None:  # Initializing coordinates_map on-the-fly
            self.coordinates_map = get_coordinates_map(output.size(), output.device)

        # Sanity checking device
        if self.coordinates_map[0].device != output.device:
            self.coordinates_map = get_coordinates_map(output.size(), output.device)

        coords_y, coords_x = self.coordinates_map  # Getting coordinates map


        coords_y = coords_y.expand(output.size())  # Fitting to the batch shape
        coords_x = coords_x.expand(output.size())


        # Thresholding with the defined threshold
        output_thresholded = self.threshold(output)
        # The total sum will be used for norm
        output_sum = torch.sum(output_thresholded, dim=[2, 3])

        centroids_y = torch.sum(output_thresholded * coords_y,
                                dim=[2, 3]) / output_sum
        centroids_x = torch.sum(output_thresholded * coords_x,
                                dim=[2, 3]) / output_sum
        
        return centroids_y, centroids_x

    def forward(self, output, truths=None):
        """Compute forward pass.
        
        Output should be of format (B,C,H,W)"""
        if self.crit_classes is not None:
            full_output = torch.empty((output.shape[0], max(max(self.crit_classes), max(self.uncrit_classes))+1, *output.shape[2:]), device=output.device)
            for i, crit_class in enumerate(self.crit_classes):
                full_output[:,crit_class] = output[:,i]
            for i, uncrit_class in enumerate(self.uncrit_classes):
                full_output[:,uncrit_class] = (truths==uncrit_class).double()
        else:
            full_output = output
        
        centroids_y, centroids_x = self.compute_centroids(full_output)
        dy_all, dx_all = self.compute_errors(centroids_y, centroids_x)

        # Aggregating the errors - TODO: other aggregations?
        error = dy_all.sum() + dx_all.sum()
        return error
    
    def compute_metric(self, output, truths):
        """Like forward, but it return the value per object"""
        if self.crit_classes is not None:
            full_output = torch.empty((output.shape[0], max(max(self.crit_classes), max(self.uncrit_classes))+1, *output.shape[2:]), device=output.device)
            for i, crit_class in enumerate(self.crit_classes):
                full_output[:,crit_class] = output[:,i]
            for i, uncrit_class in enumerate(self.uncrit_classes):
                full_output[:,uncrit_class] = (truths==uncrit_class).double()
        else:
            full_output = output

        centroids_y, centroids_x = self.compute_centroids(full_output)
        dy_all, dx_all = self.compute_errors(centroids_y, centroids_x)  

        # Aggregating the errors **over the relations only**
        error = dy_all.sum(dim=0) + dx_all.sum(dim=0)
        return error


class SpatialPriorErrorDetection(SpatialPriorError):
    def __init__(self, relations):
        """Spatial relationship prior loss for detection tasks.

        Args:
            relations (list): List of spatial relationships in the format `(source, target, dy, dx)`
        """
        super(SpatialPriorErrorDetection, self).__init__(relations)
    
    def forward(self, output):
        """Compute forward pass.
        
        Output should be of format (B,C,4)"""
        # Adding a dummy "background" centroid
        centroids_y = torch.zeros((output.size(0), output.size(1)+1))
        centroids_x = torch.zeros((output.size(0), output.size(1)+1))
        centroids_y[:,1:], centroids_x[:,1:] = output[:,0], output[:,1]
        dy_all, dx_all = self.compute_errors(centroids_y, centroids_x)
        
        # Aggregating the errors - TODO: other aggregations?
        error = dy_all.sum() + dx_all.sum()
        return error
    
    def compute_metric(self, output):
        """Like forward, but it returns the value per object"""
        # Adding a dummy "background" centroid
        centroids_y = torch.zeros((output.size(0), output.size(1)+1))
        centroids_x = torch.zeros((output.size(0), output.size(1)+1))
        centroids_y[:,1:], centroids_x[:,1:] = output[:,:,0], output[:,:,1]
        dy_all, dx_all = self.compute_errors(centroids_y, centroids_x)  

        # Aggregating the errors **over the relations only**
        error = dy_all.sum(dim=0) + dx_all.sum(dim=0)
        return error

"""=================================================
             RELATIONAL MAP OVERLAP
================================================="""

class RelationalMapOverlap(nn.Module):
    def __init__(self, relations, num_classes=None, crit_classes=None, device="cpu") -> None:
        """Relational Map Overlap loss class.
        
        Given a set of relationships, defined as a triplet `source, target, kernel`,
        compute the RMO error for a given input labelmap."""
        super(RelationalMapOverlap, self).__init__()

        self.relations = relations
        self.device = device

        self.rel_sources = [relation[0] for relation in self.relations]
        self.rel_targets = [relation[1] for relation in self.relations]
        self.rel_kernels = [relation[2] for relation in self.relations]

        # A division epsilon for empty maps
        self.epsilon = 1e-7

        # Getting criterion classes list
        self.crit_classes = crit_classes
        if crit_classes is not None:
            self.uncrit_classes = [x for x in range(num_classes+1) if x not in crit_classes]

    def forward(self, output, truths=None):
        """Compute forward pass of RMO loss.
        
        Output should be of format (B,C,H,W)"""
        # If only a few classes are part of the criterion, reassemble the full output
        if self.crit_classes is not None:
            full_output = torch.empty((output.shape[0], max(max(self.crit_classes), max(self.uncrit_classes))+1, *output.shape[2:]), device=output.device)
            for i, crit_class in enumerate(self.crit_classes):
                full_output[:,crit_class] = output[:,i]
            for i, uncrit_class in enumerate(self.uncrit_classes):
                full_output[:,uncrit_class] = (truths==uncrit_class).double()
        else:
            full_output = output

        # Convolve all sources with their respective kernels        
        rel_maps = torch.stack([conv2d(
                                    full_output[:, source].unsqueeze(1),    # Output maps of class `source` (B, 1, H, W)
                                    kernel.view(1,1, *kernel.size()),       # Kernel of relationship (1,1,H',W')
                                    padding="same")                         # Padding as same
                                for source, _, kernel in self.relations]
                               ).squeeze().permute(1,0,2,3)                 # Output of each conv is (B,1,H,W); output of stack is (R,B,1,H,W); final is (B,R,H,W)

        # Normalise all relationship maps to [0..1]
        #   Note: this normalisation is not perfect, spec. due to shape effects as discussed in the paper, but it should
        #   work for now
        # Computing extrema per map (views are for guaranteeing dimensional collapse); conversion is for division
        min_per_map = rel_maps.view(rel_maps.size(0),rel_maps.size(1),-1).min(dim=2)[0].to(dtype=torch.float)
        max_per_map = rel_maps.view(rel_maps.size(0),rel_maps.size(1),-1).max(dim=2)[0].to(dtype=torch.float)

        rel_maps = torch.div(
            torch.sub(
                rel_maps.view(*max_per_map.size(), -1).permute(2, 0, 1),min_per_map),
            max_per_map-min_per_map+self.epsilon
        ).permute(1,2,0).view(rel_maps.size())

        # Remove the source probability map for all relational maps
        rel_maps = rel_maps.sub_(full_output[:, self.rel_sources]).clamp_min_(0)

        # Intersect all maps with their respective targets
        rel_intersections = rel_maps * full_output[:, self.rel_targets]

        # Computing each map sum-score and dividing it by the overall size of the target object
        rel_sums = torch.sum(rel_intersections,dim=[2,3])
        target_sums = torch.sum(full_output[:, self.rel_targets], dim=[2,3])
        rel_scores = torch.div(rel_sums,target_sums)

        if isnan(torch.div(torch.sum(1-rel_scores), rel_scores.nelement()).item()):
            raise ValueError("NaN in Spatial Map Loss", "RMisNaN")

        # Computing final loss as the average of the sum of the complement of the scores
        #if self.reduction == "mean":
        #    return torch.mean(1-rel_scores)
        #if self.reduction == "sum":
        #    return torch.sum(1-rel_scores)
        #if self.reduction == "none" or self.reduction is None:
        #    return 1-rel_scores
        return torch.sum(1-rel_scores)

    def compute_metric(self, output, truths=None):
        """Same as the forward pass, but returns the value per object."""
        # If only a few classes are part of the criterion, reassemble the full output
        if self.crit_classes is not None:
            full_output = torch.empty((output.shape[0], max(max(self.crit_classes), max(self.uncrit_classes))+1, *output.shape[2:]), device=output.device)
            for i, crit_class in enumerate(self.crit_classes):
                full_output[:,crit_class] = output[:,i]
            for i, uncrit_class in enumerate(self.uncrit_classes):
                full_output[:,uncrit_class] = (truths==uncrit_class).double()
        else:
            full_output = output

        # Convolve all sources with their respective kernels
        rel_maps = torch.stack([conv2d(
                                    full_output[:, source].unsqueeze(1),    # Output maps of class `source` (B, 1, H, W)
                                    kernel.view(1,1, *kernel.size()),       # Kernel of relationship (1,1,H',W')
                                    padding="same")                         # Padding as same
                                for source, _, kernel in self.relations]
                               ).squeeze().permute(1,0,2,3)                 # Output of each conv is (B,1,H,W); output of stack is (R,B,1,H,W); final is (B,R,H,W)

        # Normalise all relationship maps to [0..1]
        #   Note: this normalisation is not perfect, spec. due to shape effects as discussed in the paper, but it should
        #   work for now
        # Computing extrema per map (views are for guaranteeing dimensional collapse); conversion is for division
        min_per_map = rel_maps.view(rel_maps.size(0),rel_maps.size(1),-1).min(dim=2)[0].to(dtype=torch.float)
        max_per_map = rel_maps.view(rel_maps.size(0),rel_maps.size(1),-1).max(dim=2)[0].to(dtype=torch.float)

        rel_maps = torch.div(
            torch.sub(
                rel_maps.view(*max_per_map.size(), -1).permute(2, 0, 1),min_per_map),
            max_per_map-min_per_map+self.epsilon
        ).permute(1,2,0).view(rel_maps.size())

        # Remove the source probability map for all relational maps
        rel_maps = rel_maps.sub_(full_output[:, self.rel_sources]).clamp_min_(0)

        # Intersect all maps with their respective targets
        rel_intersections = rel_maps * full_output[:, self.rel_targets]

        # Computing each map sum-score and dividing it by the overall size of the target object
        rel_sums = torch.sum(rel_intersections,dim=[2,3])
        target_sums = torch.sum(full_output[:, self.rel_targets], dim=[2,3])
        rel_scores = torch.div(rel_sums,target_sums)

        if isnan(torch.div(torch.sum(1-rel_scores), rel_scores.nelement()).item()):
            raise ValueError("NaN in Spatial Map Loss", "RMisNaN")

        # Returning metric per object
        return torch.sum(1-rel_scores, dim=0)

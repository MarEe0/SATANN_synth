"""Spatial loss functions and classes.

Author
------
 * Mateus Riva (mateus.riva@telecom-paris.fr)
"""
import torch
from torch import nn


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


def get_coordinates_map(image_dimensions):
    if len(image_dimensions) == 2:  # 2-dimensional input (h x w)
        h, w = image_dimensions
    elif len(image_dimensions) == 4:  # 2-dimensional batch input (n x c x h x w)
        _, _, h, w = image_dimensions
    else:
        raise ValueError("Image dimensions has shape {}, only 2 and 4 are accepted".format(image_dimensions.shape))
    coordinates_map = torch.meshgrid([torch.arange(h), torch.arange(w)], indexing="ij")
    coordinates_map = (coordinates_map[0].to(torch.device("cuda"))/h, coordinates_map[1].to(torch.device("cuda"))/w)  # normalizing
    return coordinates_map

class SpatialPriorErrorSegmentation(SpatialPriorError):
    def __init__(self, relations, image_dimensions=None, num_classes=None):
        """Spatial prior loss for segmentation tasks.

        Args:
            relations (list): List of spatial relationships in the format `(source, target, dy, dx)`
            image_dimensions (tuple or None): shape of input images. If None, computed on-the-fly.
        """
        super(SpatialPriorErrorSegmentation, self).__init__(relations)

        if image_dimensions is not None:
            self.image_dimensions = image_dimensions
            self.coordinates_map = get_coordinates_map(image_dimensions)
        else:
            self.image_dimensions = None
            self.coordinates_map = None

        if num_classes:
            self.threshold = nn.Threshold(1.0/num_classes, 0)
        else:
            self.threshold = None

    def compute_centroids(self, output):
        # Computing centroids
        if self.image_dimensions is None:  # Initializing coordinates_map on-the-fly
            self.coordinates_map = get_coordinates_map(output.size())

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

    def forward(self, output):
        """Compute forward pass.
        
        Output should be of format (B,C,H,W)"""
        centroids_y, centroids_x = self.compute_centroids(output)
        dy_all, dx_all = self.compute_errors(centroids_y, centroids_x)

        # Aggregating the errors - TODO: other aggregations?
        error = dy_all.sum() + dx_all.sum()
        return error
    
    def compute_metric(self, output):
        """Like forward, but it return the value per object"""
        centroids_y, centroids_x = self.compute_centroids(output)
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
        
        Output should be of format (B,C,2)"""
        centroids_y, centroids_x = output[...,0], output[...,1]
        dy_all, dx_all = self.compute_errors(centroids_y, centroids_x)
        
        # Aggregating the errors - TODO: other aggregations?
        error = dy_all.sum() + dx_all.sum()
        return error

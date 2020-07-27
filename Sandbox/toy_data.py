"""Implemented by Victor Zuanazzi
Implementation of Datasets Classes with the necessary methods for Pytorch."""

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.utils import make_grid

# my imports
from ..aml_utils import random_idxs


class ToyDataset(Dataset):
    """Abstract class for implementation of synthetic datasets."""

    def __init__(self, n_sweeps=3, n_points=5, max_distance=None, input_dims=3, flow_dim=3, **kwargs):
        """
        Args:
        kwargs:
        """
        self.n_sweeps = n_sweeps
        self.n_points = n_points
        self.max_distance = max_distance
        self.input_dims = input_dims
        self.flow_dim = flow_dim

        # Initialize kwargs
        self.random_transition = kwargs.get("random_transition", False)
        self.acceleration = kwargs.get("acceleration", False)

        # set some needed values:
        self.sigma_transition = 0.1 * self.random_transition  # defines the variance of the transitions
        self.sigma_pc = np.minimum(1.0, np.maximum(self.n_points / 10.0, 0.1))  # variance to generate the point cloud
        self.a = 1 + self.acceleration  # defines if speed is constant or if acceleration is constant.

    def do_eval(self):
        return True

    def plot_annotations(self, ax, c_1=None, c_2=None, tokens=None, flow=None):
        if (c_1 is not None) and (c_2 is not None):
            self.plot_quiver(ax, c_1, c_2, flow)

    def plot_quiver(self, ax, c_1, c_2, flow=None, max_size=40):
        # subsample the indices of the point clouds in case there are too many points to clutter the visualization.
        idx = random_idxs(input_size=c_1.shape[1], output_size=1024) if c_1.shape[1] > (1024 * 1024) else np.arange(
            c_1.shape[1])

        # decomposes the coordinates x, y, z of each point cloud.
        x, y, z = c_1[0, idx], c_1[1, idx], c_1[2, idx]
        u, v, w = c_2[0, idx], c_2[1, idx], c_2[2, idx]

        # Calculate the flow if there are correspondences, or use the provided flow matrix.
        if flow is None:
            f_x, f_y, f_z = u - x, v - y, w - z
        else:
            f_x, f_y, f_z = flow[0, idx], flow[1, idx], flow[2, idx]

        ax_dict = ax.__dict__
        if ax_dict.get("_projection", False):
            # 3D plot

            # Plot the two point clouds in 3d
            ax.scatter(x, y, z)
            ax.scatter(u, v, w)

            # Plot the flow field in 3d
            ax.quiver(x, y, z, f_x, f_y, f_z, color="black")

        else:
            # 2D Plot
            # Plot the two point clouds in 2d.
            ax.scatter(x, y)
            ax.scatter(u, v)

            # Plot the 2d flow field
            ax.quiver(x, y, f_x, f_y, scale=1.0, scale_units='xy', angles='xy')

    def normalize_image(self, img):

        img_min = img.min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        img_max = img.max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]

        img = (img - img_min) / (img_max - img_min)

        return img.clamp(min=0, max=1)

    def plot_scene(self, ax, img_list, max_norm=35):
        # pdb.set_trace()

        # mask_list = [img != 0.0 for img in img_list]
        mask_list = [img.norm(dim=0, keepdim=True) < max_norm for img in img_list]
        img_list = [img * mask for (img, mask) in zip(img_list, mask_list)]
        img_list_normalized = [self.normalize_image(img) for img in img_list]
        img_list_unsqueezed = [img.unsqueeze(0) for img in img_list_normalized]
        mask_list_unsqueezed = [mask.float().unsqueeze(0) for mask in mask_list]
        img_cat = torch.cat(img_list_unsqueezed, dim=0)
        mask_cat = torch.cat(mask_list_unsqueezed, dim=0)
        img_grid = make_grid(img_cat, nrow=1, padding=3, normalize=False, scale_each=False, pad_value=1)
        mask_grid = make_grid(mask_cat, nrow=1, padding=3, normalize=False, pad_value=1)

        # small trick for better visualization
        img_grid = (mask_grid * img_grid) #  ** (1 / 3)

        ax.imshow(img_grid.permute(1, 2, 0))

    def plot_overlap(self, ax, img_1, img_2, img_3=None):

        img_list = [img_1, img_2]
        if img_3 is None:
            img_3 = torch.zeros(1, img_1.shape[1], img_1.shape[2])
        else:
            img_list += [img_3]

        img_overlap = torch.cat((img_1.mean(dim=0, keepdim=True),
                                 img_2.mean(dim=0, keepdim=True),
                                 img_3.mean(dim=0, keepdim=True)), dim=0)

        img_list += [img_overlap]

        self.plot_scene(ax, img_list)

    def len(self):
        return -1

    def __len__(self):
        """ required by pytorch, but it has no meaning
        :return returns 5 """
        return 4242

    def __getitem__(self, idx):
        """Example of method for getittem"""

        # uses idxs as seed for the point cloud.
        # I am not entirely sure why this is necessary, but without this line __getitem__() outputs always the same
        # point cloud.
        np.random.seed(idx)

        # creates first point cloud
        x = np.random.normal(loc=0,
                             scale=self.sigma_pc,
                             size=(self.input_dims, self.n_points))

        # define transition
        direction = 1 + np.random.choice([-2, 0]) * self.random_transition  # direction of movement of the pointcloud.
        transitions = np.random.normal(loc=direction,
                                       scale=self.sigma_transition,
                                       size=(self.n_sweeps, self.input_dims, self.n_points))

        # list of translated point clouds
        data_points = [x + t * (i ** self.a) for i, t in enumerate(transitions)]

        # lists are not useful, arrays are useful:
        data_points = np.concatenate(data_points, axis=1)

        return data_points, {"flow": transitions[-1, :, :] * (self.n_sweeps ** self.a)}


class SquaresToyDataset(ToyDataset):
    def __getitem__(self, idx, height=None):
        """Pytorch required function for get item."""
        # As suggested by Ola, the transition (aka flow) is the ground truth.

        # uses idxs as seed for the point cloud.
        # I am not entirely sure why this is necessary, but without this line __getitem__() outputs always the same
        # point cloud.
        np.random.seed(idx)
        n_p = self.n_points // 2
        space_dims = self.input_dims

        # initialize centers and boundaries:
        center_static = np.random.uniform(low=-3, high=3, size=space_dims)
        half_size_static = np.random.uniform(low=0.25, high=0.75)
        boundaries_static = (center_static - half_size_static,
                             center_static + half_size_static)  # (low corner, high corner)

        center_dynamic = np.random.uniform(low=-3, high=3, size=space_dims)
        half_size_dynamic = np.random.uniform(low=0.25, high=0.75)
        boundaries_dynamic = (center_dynamic - half_size_dynamic,
                              center_dynamic + half_size_dynamic)  # (low corner, high corner)

        # define the transitions
        direction = np.random.uniform(low=-1.1,
                                      high=1.1,
                                      size=(space_dims,)) if self.random_transition else np.ones(space_dims)

        data_points = np.zeros((self.input_dims, self.n_points * self.n_sweeps))
        flow = np.zeros((space_dims, self.n_points * (self.n_sweeps - 1)))
        mask = np.ones(self.n_points * (self.n_sweeps - 1))
        for s in range(self.n_sweeps):
            # number of points in ach box:
            if self.random_transition:
                prop = half_size_static / (half_size_dynamic + half_size_static)  # prop ranges in [0.25, 0.75]
                n_p_static = np.random.randint(low=(self.n_points * prop * 0.5),
                                               high=int(self.n_points * (prop ** 0.5)))
                n_p_dynamic = self.n_points - n_p_static
            else:
                n_p_static, n_p_dynamic = n_p, n_p

            # (re)samples points of the dynamic square:
            samples_dyn = np.random.uniform(low=0, high=1, size=(space_dims, n_p_dynamic))
            samples_dyn += boundaries_dynamic[0].reshape(space_dims, 1)

            # (re)sample points from the static square
            samples_sta = np.random.uniform(low=0, high=1, size=(space_dims, n_p_static))
            samples_sta += boundaries_static[0].reshape(space_dims, 1)

            # all points of the last dimension are on the "top" of the box
            samples_dyn[-1, :] = boundaries_dynamic[1][-1]
            samples_sta[-1, :] = boundaries_static[1][-1]

            # check for occlusions:
            dist = abs(center_dynamic[:-1] - center_static[:-1])
            touching_dist = half_size_dynamic + half_size_static
            if (touching_dist > dist).all():
                # points located in the overlapped region are assigned to the highest surface
                up_corners = np.minimum(boundaries_dynamic[1][:-1], boundaries_static[1][:-1])
                down_corners = np.maximum(boundaries_dynamic[0][:-1], boundaries_static[0][:-1])

                top_height = max(boundaries_dynamic[1][-1], boundaries_static[1][-1])
                cases_dyn = (samples_dyn[:-1, :] < up_corners.reshape(space_dims - 1, 1)).all(axis=0)
                cases_dyn *= (samples_dyn[:-1, :] > down_corners.reshape(space_dims - 1, 1)).all(axis=0)
                cases_sta = (samples_sta[:-1, :] < up_corners.reshape(space_dims - 1, 1)).all(axis=0)
                cases_sta *= (samples_sta[:-1, :] > down_corners.reshape(space_dims - 1, 1)).all(axis=0)
                samples_dyn[-1, cases_dyn] = top_height
                samples_sta[-1, cases_sta] = top_height
            else:
                cases_dyn, cases_sta = False, False

            # updates the data matrices
            start_dyn = s * self.n_points
            end_dyn = start_dyn + n_p_dynamic
            start_sta = end_dyn
            end_sta = start_sta + n_p_static
            data_points[:space_dims, start_dyn:end_dyn] = samples_dyn
            data_points[:space_dims, start_sta:end_sta] = samples_sta

            if s == self.n_sweeps - 1:
                # there is no flow in the last frame
                continue

            # moves the dynamic square:
            translation = direction + np.random.normal(loc=0.0,
                                                       scale=self.sigma_transition,
                                                       size=direction.shape)

            samples_dyn += translation.reshape(space_dims, 1)
            center_dynamic += translation
            boundaries_dynamic = (center_dynamic - half_size_dynamic, center_dynamic + half_size_dynamic)

            # check for occlusions:
            dist = abs(center_dynamic[:-1] - center_static[:-1])
            touching_dist = half_size_dynamic + half_size_static
            if (touching_dist > dist).all():
                up_corners = np.minimum(boundaries_dynamic[1][:-1], boundaries_static[1][:-1])
                down_corners = np.maximum(boundaries_dynamic[0][:-1], boundaries_static[0][:-1])

                # points located in the overlapped region are assigned to the highest surface
                top_height = max(boundaries_dynamic[1][-1], boundaries_static[1][-1])
                cases_dyn_2 = (samples_dyn[:-1, :] < up_corners.reshape(space_dims - 1, 1)).all(axis=0)
                cases_dyn_2 *= (samples_dyn[:-1, :] > down_corners.reshape(space_dims - 1, 1)).all(axis=0)
                cases_sta_2 = (samples_sta[:-1, :] < up_corners.reshape(space_dims - 1, 1)).all(axis=0)
                cases_sta_2 *= (samples_sta[:-1, :] > down_corners.reshape(space_dims - 1, 1)).all(axis=0)
                cases_dyn += cases_dyn_2
                cases_sta += cases_sta_2
                samples_dyn[-1, cases_dyn] = top_height
                samples_sta[-1, cases_sta] = top_height

            # updates flow and mask
            flow[:space_dims, start_dyn:end_dyn] = np.zeros_like(samples_dyn) + translation.reshape(space_dims, 1)
            mask[start_dyn:end_dyn] = 1 - np.array(cases_dyn).clip(min=0, max=1).astype(int)
            mask[start_sta:end_sta] = 1 - np.array(cases_sta).clip(min=0, max=1).astype(int)

        return data_points, {"flow": flow[:self.flow_dim, :], "mask": mask}

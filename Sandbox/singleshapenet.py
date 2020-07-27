import numpy as np
import torch
from torch_geometric.datasets import ShapeNet
from torch_geometric.data import DataLoader as DataLoader_geo

# my imports
from ..aml_utils import random_idxs
from toy_data import ToyDataset


class SingleShapeNet(ToyDataset):
    """Dataset implemented for quantitative evaluation of flow models."""

    def __init__(self, n_sweeps=2, n_points=5, max_distance=None, flow_dim=3, **kwargs):
        """
        Args:
        kwargs:
        """
        self.n_sweeps = n_sweeps
        self.n_points = n_points
        self.max_distance = max_distance
        self.flow_dim = flow_dim
        self.partition = kwargs.get("partition", "train")
        self.overfit = kwargs.get("overfit", False)
        self.data_size = kwargs.get("data_size", 4242)
        if (self.data_size is None) or (self.data_size < 1):
            self.data_size = 4242

        self.dataset_len = 4242 if self.partition == "train" else 242
        self.odds = 2 if self.partition == "train" else 10

        # keep datapoints in memory for overfitting tests and speed up training
        self.cache = {}

        # root_path = "../../../media/deepstorage01/datasets_external/"
        root_path = "data/"

        self.dataset = ShapeNet(root=root_path,
                                # transform=T.SamplePoints(n_sweeps * n_points),
                                # pre_transform=T.Cartesian(),
                                split=self.partition)

        self.loader = DataLoader_geo(self.dataset, batch_size=1, shuffle=True)

        # Initialize kwargs
        self.random_transition = kwargs.get("random_transition", False)
        self.acceleration = kwargs.get("acceleration", False)
        self.rotate = kwargs.get("rotate", True)

        # set some needed values:
        self.sigma_transition = 0.1 * self.random_transition  # defines the variance of the transitions
        self.sigma_pc = np.minimum(1.0, np.maximum(self.n_points / 10.0, 0.1))  # variance to generate the point cloud
        self.a = 1 + self.acceleration  # defines if speed is constant or if acceleration is constant.

    def make_rotation_matrix(self, angle, stretch):
        """make a rotation matrix given the angle
        angle: torch.tensor, with shape (3,) containing angle in rads
        strech: torch.tensor with shape (3,) containing the stretch coeficient"""

        cos = angle.cos()
        sin = angle.sin()
        rx = torch.tensor([[stretch[0], 0, 0], [0, cos[0], -sin[0]], [0, sin[0], cos[0]]])
        ry = torch.tensor([[cos[1], 0, sin[1]], [0, stretch[1], 0], [-sin[1], 0, cos[1]]])
        rz = torch.tensor([[cos[2], -sin[2], 0], [sin[2], cos[2], 0], [0, 0, stretch[2]]])

        rotation_matrix = rx @ ry @ rz

        return rotation_matrix

    def __len__(self):
        """ required by pytorch, but it has no meaning
        """
        return self.dataset_len

    def __getitem__(self, idx):
        """Pytorch required function for get item."""

        idx = idx % self.data_size

        if idx in self.cache and (self.overfit or np.random.randint(self.odds)):
            return self.cache[idx]

        # uses idxs as seed for the point cloud.
        torch.manual_seed(idx)
        cloud_generator = iter(self.loader)
        data = next(cloud_generator)

        clouds = data.pos
        clouds.transpose_(0, 1)

        # normals = data.x
        # normals.transpose_(0, 1)

        classes = data.y

        # initial transform
        init_stretch = 5 + torch.rand((3,))
        init_angle = torch.rand((3,)) * np.pi * 2
        init_trans = torch.rand((3, 1)) * 2 - 1
        init_rot = self.make_rotation_matrix(init_angle, init_stretch)
        clouds = (init_rot @ clouds) + init_trans

        # parameters for displacement
        translation = torch.rand((3, 1)) * 0.5 - 0.25
        angle = ((torch.rand((3,)) * np.pi - (np.pi / 2)) / 15.0) * self.rotate
        stretch = ((torch.rand((3,)) * 2 - 1) * 0.1 + 1) if self.rotate else torch.ones((3,))
        rotation_matrix = self.make_rotation_matrix(angle, stretch)
        # translation = (torch.rand((3, 1)) * 0.5 - 0.25) / 5
        # angle = ((torch.rand((3,)) * np.pi - (np.pi / 2)) / 30.0) * self.rotate
        # stretch = ((torch.rand((3,)) * 2 - 1) * 0.01 + 1) if self.rotate else torch.ones((3,))
        # rotation_matrix = self.make_rotation_matrix(angle, stretch)

        data_points = torch.zeros((3, self.n_sweeps * self.n_points))
        data_classes = torch.zeros(self.n_sweeps * self.n_points)
        flow = torch.zeros((3, (self.n_sweeps - 1) * self.n_points))
        mask = np.ones_like(flow)

        for s in range(self.n_sweeps):
            start = s * self.n_points
            end = start + self.n_points

            # samples points for the point cloud
            rnd_idx = random_idxs(input_size=clouds.shape[1], output_size=self.n_points)
            data_points[:, start:end] = clouds[:, rnd_idx].clone()
            data_classes[start:end] = classes[rnd_idx]

            if s < (self.n_sweeps - 1):
                # calculates the displacement of the hole thing
                clouds = (rotation_matrix @ clouds) + translation
                flow[:, start:end] = clouds[:, rnd_idx] - data_points[:, start:end]

        self.cache[idx] = (data_points, {"flow": flow[:self.flow_dim, :], "mask": mask, "classes": data_classes})

        return data_points, {"flow": flow[:self.flow_dim, :], "mask": mask, "classes": data_classes}
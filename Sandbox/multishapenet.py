import numpy as np
import torch
from torch_geometric.datasets import ShapeNet
from torch_geometric.data import DataLoader as DataLoader_geo

# my imports
from ..aml_utils import random_idxs
from toy_data import ToyDataset


class MultiShapeNet(ToyDataset):
    """Dataset implemented for quantitative evaluation of flow models."""

    def __init__(self, n_sweeps=2, n_points=512, max_distance=None, flow_dim=3, **kwargs):
        """
        Args:
        kwargs:
        """
        self.n_sweeps = n_sweeps
        self.n_points = n_points
        self.max_distance = max_distance
        self.flow_dim = flow_dim
        self.occlusion = kwargs.get("occlusion", True)
        self.partition = kwargs.get("partition", "train")
        self.overfit = kwargs.get("overfit", False)
        self.data_size = kwargs.get("data_size", 4242)
        if (self.data_size is None) or (self.data_size < 1):
            self.data_size = 4242
        self.dataset_len = 4242 if self.partition == "train" else 242
        self.max_objects = 10

        self.odds = 2 if self.partition == "train" else 10

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

        self.cache = {}

    def make_rotation_matrix(self, angle, stretch):
        """make a rotation matrix given the angle
        angle: torch.tensor, with shape (n, 3,) containing angle in rads
        strech: torch.tensor with shape (n, 3,) containing the stretch coeficient"""

        cos = angle.cos()
        sin = angle.sin()
        zeros = torch.zeros(size=(angle.shape[0], 1, 1))

        rx = torch.cat((torch.cat((stretch[..., 0:1], zeros, zeros), dim=2),
                        torch.cat((zeros, cos[..., 0:1], -sin[..., 0:1]), dim=2),
                        torch.cat((zeros, sin[..., 0:1], cos[..., 0:1]), dim=2)), dim=1)
        ry = torch.cat((torch.cat((cos[..., 1:2], zeros, sin[..., 1:2]), dim=2),
                        torch.cat((zeros, stretch[..., 1:2], zeros), dim=2),
                        torch.cat((-sin[..., 1:2], zeros, cos[..., 1:2]), dim=2)), dim=1)
        rz = torch.cat((torch.cat((cos[..., 2:3], -sin[..., 2:3], zeros), dim=2),
                        torch.cat((sin[..., 2:3], cos[..., 2:3], zeros), dim=2),
                        torch.cat((zeros, zeros, stretch[..., 2:3]), dim=2)), dim=1)

        rotation_matrix = rx @ ry @ rz

        return rotation_matrix.transpose(2, 1)

    @staticmethod
    def make_bounding_boxes(clouds):
        return clouds.max(dim=1)[0], clouds.min(dim=1)[0]

    def instance_creation(self, clouds, rotation_matrix, translation):
        """this method creates makes a list containing information about all objects in the cloud.
        input:
            clouds: torch.tensor(N objects, N points, 3), a tensor containing the points of the object.
            rotation_matrix: torch.tensor(N_bjects, 3, 3), a tensor containing one rotation matrix per object.
            translation: torch.tensor(N_objects, 3), a tensor containing one translation vector per object.
        return:
            list(dict, None), a list of len max number of objects, with dictionaries conaining informantion of each
                object and filled with None if the number of objects is not the maximum."""
        zeros = torch.zeros(3)
        fake_dict = {"corners": (zeros, zeros),
                     "center": zeros,
                     "size": zeros,
                     "id": -1,
                     "rotation_matrix": torch.zeros(3, 3),
                     "translation": torch.zeros(1, 3)}
        objects_list = [fake_dict] * self.max_objects
        up_corners = clouds.max(dim=1)[0]
        low_corners = clouds.min(dim=1)[0]

        for id, (rot_matrix, trans, uc, lc) in enumerate(zip(rotation_matrix, translation, up_corners, low_corners)):
            objects_list[id] = {}
            objects_list[id]["corners"] = (lc, uc)
            objects_list[id]["center"] = (lc + uc) / 2
            objects_list[id]["size"] = (uc - lc)
            objects_list[id]["id"] = id
            objects_list[id]["rotation_matrix"] = rot_matrix
            objects_list[id]["translation"] = trans

        return objects_list

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
        n_objects = np.random.randint(low=2, high=self.max_objects)

        # retrieves the data from disc
        cloud_generator = iter(self.loader)
        # data comes as [N, 3]

        min_n_points = np.inf
        cloud_list = []
        normal_list = []
        classes_list = []
        for n in range(n_objects):
            data = next(cloud_generator)
            cloud_list.append(data.pos)
            normal_list.append(data.x)  # surface normal vectors
            classes_list.append(data.y)  # segmentation classes
            min_n_points = min(min_n_points, cloud_list[-1].shape[0])

        # stores the data into a torch tensor (all dims have to be the same so we can make use of super fast matrix
        # operations.
        cloud_list_same_shape = [cl[:min_n_points, :].view(1, -1, 3) for cl in cloud_list]
        clouds = torch.cat(cloud_list_same_shape, dim=0)  # [B, N, 3]

        normal_list_same_shape = [cl[:min_n_points, :].view(1, -1, 3) for cl in normal_list]
        normals = torch.cat(normal_list_same_shape, dim=0)  # [B, N, 3]

        classes_list_same_shape = [cl[:min_n_points] for cl in classes_list]
        classes = torch.cat(classes_list_same_shape, dim=0).view(-1)  # [B, N, 1]

        # initial transform, each object has a different transform
        init_stretch = (torch.rand((n_objects, 1, 3)) + torch.randint(low=3, high=7, size=(n_objects, 1, 1)))
        init_angle = torch.rand((n_objects, 1, 3)) * np.pi * 2
        init_trans = torch.rand((n_objects, 1, 3)) * 20 - 10
        init_rot = self.make_rotation_matrix(init_angle, init_stretch)

        # the sequence for the matrix multiplication is inverserd because clouds is the transpose of the most common
        # convention
        clouds = (clouds @ init_rot) + init_trans
        normals = (normals @ init_rot)

        # initial point of view and sphere boundaries
        pov = torch.rand((3,)) * 2 - 1
        norms = clouds.norm(dim=2)
        w = torch.rand((1,))
        radius_max = w * norms.max() + (1 - w) * norms.mean()

        # parameters for displacement
        stretch = ((torch.rand((n_objects, 1, 3)) - 0.5) * 0.1 + 1) if self.rotate else torch.ones((n_objects, 1, 3))
        angle = ((torch.rand((n_objects, 1, 3)) * np.pi - (np.pi / 2)) / 15.0) * self.rotate
        translation = (torch.rand((n_objects, 1, 3)) - 0.5) / 2
        rotation_matrix = self.make_rotation_matrix(angle, stretch)

        data_points = torch.zeros((self.n_sweeps * self.n_points, 3))
        data_classes = torch.zeros((self.n_sweeps * self.n_points,))
        flow = torch.zeros(((self.n_sweeps - 1) * self.n_points), 3)
        mask = torch.ones(((self.n_sweeps - 1) * self.n_points,))
        object_instances = []

        for s in range(self.n_sweeps):
            start = s * self.n_points
            end = start + self.n_points

            # samples points for the point cloud
            if self.occlusion:
                valid_points = ((clouds.norm(dim=2) < radius_max) * (normals @ pov > 0.0)).view(-1)
                if (valid_points.float().sum().item()) == 0:
                    # in case al points are occluded:
                    rand_i = np.random.randint(len(self.cache))
                    return self.cache[rand_i]
            else:
                valid_points = torch.ones(clouds.view(-1, 3).shape[0], dtype=bool)

            valid_cloud = clouds.view(-1, 3)[valid_points, :]
            rnd_idx = random_idxs(input_size=(valid_points.float().sum().item()), output_size=self.n_points)
            data_points[start:end, :] = valid_cloud[rnd_idx, :].clone()
            data_classes[start:end] = classes[valid_points][rnd_idx]
            # rnd_idx = random_idxs(input_size=(clouds.shape[1] * n_objects), output_size=self.n_points)
            # data_points[start:end, :] = clouds.view(-1, 3)[rnd_idx, :].clone()
            object_instances.append(self.instance_creation(clouds, rotation_matrix.transpose(2, 1), translation))

            if s < (self.n_sweeps - 1):
                # calculates the displacement of the hole thing
                clouds = (clouds @ rotation_matrix) + translation
                flow[start:end, :] = clouds.view(-1, 3)[valid_points, :][rnd_idx, :] - data_points[start:end, :]

                if self.occlusion:
                    normals = normals @ rotation_matrix
                    valid = ((clouds.view(-1, 3)[valid_points, :][rnd_idx, :].norm(dim=-1) < radius_max) * (
                            normals.view(-1, 3)[valid_points, :][rnd_idx, :] @ pov > 0.0)).view(-1)
                    mask[start:end] *= valid

        # now we transpose everything to the convention
        data_points.transpose_(0, 1)
        flow.transpose_(0, 1)

        self.cache[idx] = data_points, {"flow": flow[:self.flow_dim, :],
                                        "mask": mask,
                                        "classes": data_classes,
                                        "object_instances": object_instances}

        return self.cache[idx]
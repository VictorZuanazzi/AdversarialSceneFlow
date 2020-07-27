import numpy as np
from torch.utils.data import Dataset
from lyft_dataset_sdk.lyftdataset import LyftDataset
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud
import pdb

# my imports
from ..aml_utils import random_idxs


class TorchLyftDataset(Dataset):
    """Wrapper of lyft_dataset_sdk.lyftdataset.LyftDataset for use with Pytorch"""

    def __init__(self, n_sweeps=3, sensor="LIDAR_TOP", n_points=60_000, max_distance=15, verbose=True, input_dims=4,
                 **kwargs):
        """
        Args:
            n_sweeps: int, number of output sweeps for the data loader;
            n_points: int, number of points per sweep. On Lyft a sweep has between 60_000 and 120_000 points. n_points
                will be sampled from which sweep with replacement.
            sensor: str, "LIDAR_TOP" is the only supported option.
        kwargs:
            root_dir: str, path to the directory of lyft.
            json_path: str, extension of root_dir to the folder where the lyft tables are stored.
        """

        self.sensor = sensor
        self.n_sweeps = n_sweeps
        self.n_points = n_points
        self.max_distance = max_distance
        self.relative_ego = True
        self.like_kitti = True

        # Initialize kwargs
        if "root_dir" in kwargs.keys():
            self.root_dir = kwargs["root_dir"]
        else:
            self.root_dir = "../../../media/deepstorage01/datasets_external/" + "lyft/train/"

        if "json_path" in kwargs.keys():
            self.json_path = kwargs["json_path"]
        else:
            self.json_path = "v1.0-train"

        # Initialize the dataset
        self._dataset = LyftDataset(data_path=self.root_dir, json_path=self.root_dir + self.json_path, verbose=verbose)

        # get the right number of dimensions
        p, _ = self.__getitem__(0)  # sample a point cloud
        self.input_dims = p.shape[0]

    def do_eval(self):
        return False

    def len(self):
        return len(self._dataset.sample)

    def __len__(self):
        """:return the number of samples"""
        return len(self._dataset.sample)

    def __getitem__(self, idx):
        """Pytorch required function for get item."""

        data_points = []
        data_tokens = []

        sample_token = self._dataset.sample[idx]["token"]
        first_token = sample_token
        position = 0
        not_last = True

        # That has to happen in a loop, we must be able to use sweeps independently.
        while len(data_points) < self.n_sweeps:

            sample = self._dataset.get("sample", sample_token)
            sample_data_token = sample["data"]["LIDAR_TOP"]
            sample_data = self._dataset.get("sample_data", sample_data_token)
            file_name = sample_data["filename"]
            file_path = self._dataset.data_path / file_name

            # Load lidar data [x, y , z]
            lidar_data = LidarPointCloud.from_file(file_path)

            # Exclude points that are beyond a distance threshold
            # Only take x, y and z, as the other dimensions are useless for this dataset.
            idx_close_points = (abs(lidar_data.points[:3, :]) < self.max_distance).all(axis=0)
            points = lidar_data.points[:3, idx_close_points]

            # select kitti like lidar data
            if self.like_kitti:
                mask_neg_x = points[0, :] > 0.0
                points = points[:, mask_neg_x]

            # (almost) Uniform sampling of the points.
            rnd_points = random_idxs(input_size=points.shape[1], output_size=self.n_points)
            points = points[:, rnd_points]

            # Insert items in the correct position.
            data_points.insert(position, points)
            data_tokens.insert(position, sample_data_token)

            # While the end of the scene is not reached, we progress forward.
            if sample["next"] and not_last:
                position = len(data_tokens)
                sample_token = sample["next"]

            # If the end of the scene was reached, then we take the frames before the first one.
            else:
                not_last = False
                position = 0
                sample_token = self._dataset.get("sample", first_token)["prev"]
                first_token = sample_token

        # lists are not useful, arrays are useful:
        data_points = np.concatenate(data_points, axis=1)

        return data_points.astype(float), {"tokens": data_tokens}

    def plot_annotations(self, ax, c_1=None, c_2=None, token=None):
        if token is not None:
            self._dataset.render_sample_data(token, ax=ax)

    def plot_quiver(self, ax, c_1, c_2):

        # subsample the indices of the point clouds in case there are too many points to clutter the visualization.
        idx = random_idxs(input_size=c_1.shape[1], output_size=1024) if c_1.shape[1] > 1024 else np.arange(c_1.shape[1])

        x, y = c_1[0, idx], c_1[1, idx]
        u, v = c_2[0, idx] - x, c_2[1, idx] - y

        ax.quiver(x, y, u, v, scale=1.0, scale_units='xy', angles='xy')
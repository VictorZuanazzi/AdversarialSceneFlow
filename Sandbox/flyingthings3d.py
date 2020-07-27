import numpy as np
import glob
import os

# my imports
from ..aml_utils import random_idxs
from toy_data import ToyDataset


class FlyingThings3D(ToyDataset):
    # Heavily inspired by https://github.com/hyangwinter/flownet3d_pytorch

    def __init__(self, n_points=1024, max_distance=None, input_dims=6, **kwargs):
        """
        Args:
        kwargs:
        """
        self.n_sweeps = 2
        self.n_points = n_points
        self.max_distance = max_distance
        self.flow_dims = 3

        self.input_dims = min(input_dims, 6)

        # Initialize kwargs
        self.root_dir = kwargs.get("root_dir", "../../../media/deepstorage01/datasets_external/" + "flyingthings3d/")
        self.partition = kwargs.get("partition", "train")
        self.verbose = kwargs.get("verbose", False)
        self.load_data_in_memory = kwargs.get("load_data_in_memory", True)
        self.temporal_flip = kwargs.get("temporal_flip", False)
        self.percentage_partition = kwargs.get("percentage_partition", 1.0)

        if kwargs.get("debug_mode", False):
            self.partition = "test"

        # creates a containing the path to all files.

        if self.verbose:
            print(f"Loading {self.partition} data from {self.root_dir}. " +
                  f"\nLoad data in memory is: {self.load_data_in_memory}.")

        if self.partition == 'train':
            self.datapath = glob.glob(os.path.join(self.root_dir, 'TRAIN*.npz'))

            # deal with one bad datapoint with nan value
            self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]

        else:
            self.datapath = glob.glob(os.path.join(self.root_dir, 'TEST*.npz'))

        last_idx = int(len(self.datapath) * self.percentage_partition)
        self.datapath = self.datapath[:last_idx]

        # Cache is a cool trick to speed up the dataloader.
        # It keeps a number of datapoints in memory, when those are requested, it __getitem__() feches them in memroy
        # instead of from disk.
        self.cache = {}
        self.cache_size = len(self.datapath) if self.load_data_in_memory else 32768

        # To benefit the most from cache, the size of the dataset is increased enormously so we loop in the already
        # stored indexes many times before restarting the dataloader.
        self.fake_dataset_size = len(self.datapath)

        if self.verbose:
            print(f"Dataset initialized. Number of examples: {len(self.datapath)}, " +
                  f"fake size is: {self.fake_dataset_size}.")

    def len(self):
        return len(self.datapath)

    def do_eval(self):
        return True

    def __len__(self):
        """ required by pytorch
        :return  lengh of the dataset"""
        return self.fake_dataset_size

    def __getitem__(self, idx):
        """Pytorch required function for get item.
        Pointcloud has the shape: [input_dims, n_points * n_sweeps]"""

        # to make sure cache works.
        idx = idx % len(self.datapath)

        def get_data(idx):
            """inner function to load the data from file"""
            file_name = self.datapath[idx]
            with open(file_name, "rb") as file:
                data = np.load(file)
                # xyz points are augmented by the rgb data.
                cloud1 = np.concatenate((data["points1"].astype("float32"),
                                         data["color1"].astype("float32")), axis=1)[:, :self.input_dims].T
                cloud2 = np.concatenate((data["points2"].astype("float32"),
                                         data["color2"].astype("float32")), axis=1)[:, :self.input_dims].T
                flow = data["flow"].astype("float32").T
                mask = data["valid_mask1"]

            return cloud1, cloud2, flow, mask

        if self.load_data_in_memory:
            if idx in self.cache:
                cloud1, cloud2, flow, mask = self.cache[idx]
            else:
                cloud1, cloud2, flow, mask = get_data(idx)
                try:
                    # cache the data if there is still memory
                    self.cache[idx] = (cloud1, cloud2, flow, mask)
                except OSError:
                    # else we may be dangerously without memory
                    del self.cache[list(self.cache.keys())[0]]

        else:
            cloud1, cloud2, flow, mask = get_data(idx)

        # randomly select points from the clouds
        rnd_sample_idx = random_idxs(input_size=cloud1.shape[1], output_size=self.n_points)
        c1 = cloud1[:, rnd_sample_idx]
        flow = flow[:, rnd_sample_idx]
        mask = mask[rnd_sample_idx]

        rnd_sample_idx = random_idxs(input_size=cloud2.shape[1], output_size=self.n_points)
        c2 = cloud2[:, rnd_sample_idx]

        # point clouds are put together in one array
        data_points = np.concatenate((c1, c2), axis=1)

        return data_points, {"flow": flow, "mask": mask}
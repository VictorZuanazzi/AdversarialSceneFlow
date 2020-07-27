import numpy as np
import glob
import os
import os.path as osp
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torch_geometric.datasets import ShapeNet
from torch_geometric.data import DataLoader as DataLoader_geo
import matplotlib.pyplot as plt
from flownet3d.data_preprocessing import load_pfm
import cv2
import pdb
import time

# my imports
from ..aml_utils import random_idxs

class KITTI(ToyDataset):
    # Heavily inspired by https://github.com/hyangwinter/flownet3d_pytorch

    def __init__(self, n_points=1024, max_distance=15, **kwargs):
        """
        Args:
        kwargs:
        """
        self.n_sweeps = 2
        self.n_points = n_points
        self.max_distance = max_distance

        # Initialize kwargs
        # self.root_dir = kwargs.get("root_dir", "../../../media/deepstorage01/datasets_external/" + "kitti/")
        # kitti_pc  has point clouds in the convension (Width, Height, Depth)
        # kitti_xyz has point clouds in the convention (Depth, Width, Height) this convetion is more intuitive
        # for plotting.
        self.root_dir = kwargs.get("root_dir", "data/" + "kitti_xyz/")
        self.verbose = kwargs.get("verbose", False)
        self.load_data_in_memory = kwargs.get("load_data_in_memory", True)
        self.percentage_partition = kwargs.get("percentage_partition", 1.0)
        self.remove_ground = kwargs.get("remove_ground", False)
        self.partition = kwargs.get("partition", "train")

        # creates a containing the path to all files.

        if self.verbose:
            print(f"Loading data from {self.root_dir}. " +
                  f"\nLoad data in memory is: {self.load_data_in_memory}.")

        self.datapath = glob.glob(os.path.join(self.root_dir, '*.npz'))
        if self.partition == "train":
            self.datapath = self.datapath[:150]
        else:
            self.datapath = self.datapath[150:]

        last_idx = int(len(self.datapath) * self.percentage_partition)
        self.datapath = self.datapath[:last_idx]

        # Cache is a cool trick to speed up the dataloader.
        # It keeps a number of datapoints in memory, when those are requested, it __getitem__() feches them in memroy
        # instead of from disk.
        self.cache = {}

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
                cloud1 = data["pc1"].astype("float32")
                cloud2 = data["pc2"].astype("float32")
                flow = data["flow"].astype("float32")
                mask = data["mask"].squeeze()

                # filter out points too far away
                valid_mask = (abs(cloud1[:3, :]) < self.max_distance).all(axis=0)
                cloud1 = cloud1[:, valid_mask]
                flow = flow[:, valid_mask]
                mask = mask[valid_mask]

                valid_mask = (abs(cloud2[:3, :]) < self.max_distance).all(axis=0)
                cloud2 = cloud2[:, valid_mask]

            if self.remove_ground:
                # http://www.cvlibs.net/datasets/kitti/setup.php
                # the lindar is mounted at 1.73 m on the car from the road level.
                # We can consider ground everything that is < - 1.4
                floor_level = -1.4  # cloud1[2, :].min() + 0.3

                idx_keep = cloud1[2, :] > floor_level
                cloud1 = cloud1[:, idx_keep]
                flow = flow[:, idx_keep]
                mask = mask[idx_keep]
                cloud2 = cloud2[:, cloud2[2, :] > floor_level]

                # include occluded points by ground removal in the mask
                iof_mask = cloud1[2, :] + flow[2, :] > floor_level
                mask = mask * iof_mask

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


class KittiNoGround(ToyDataset):
    # Heavily inspired by https://github.com/hyangwinter/flownet3d_pytorch

    def __init__(self, n_points=1024, max_distance=15, **kwargs):
        """
        Args:
        kwargs:
        """
        self.n_sweeps = 2
        self.n_points = n_points
        self.max_distance = max_distance

        # Initialize kwargs
        # self.root_dir = kwargs.get("root_dir", "../../../media/deepstorage01/datasets_external/" + "kitti/")
        # kitti_pc  has point clouds in the convension (Width, Height, Depth)
        # kitti_xyz has point clouds in the convention (Depth, Width, Height) this convetion is more intuitive
        # for plotting.
        self.root_dir = kwargs.get("root_dir", "data/" + "kitti_no_ground/")
        self.verbose = kwargs.get("verbose", False)
        self.load_data_in_memory = kwargs.get("load_data_in_memory", True)
        self.percentage_partition = kwargs.get("percentage_partition", 1.0)
        self.remove_ground = kwargs.get("remove_ground", False)
        self.partition = kwargs.get("partition", "train")

        # creates a containing the path to all files.

        if self.verbose:
            print(f"Loading data from {self.root_dir}. " +
                  f"\nLoad data in memory is: {self.load_data_in_memory}.")

        self.datapath = glob.glob(os.path.join(self.root_dir, '*.npz'))
        # if self.partition == "train":
        #     self.datapath = self.datapath[:100]
        # else:
        #     self.datapath = self.datapath[100:]

        last_idx = int(len(self.datapath) * self.percentage_partition)
        self.datapath = self.datapath[:last_idx]

        # Cache is a cool trick to speed up the dataloader.
        # It keeps a number of datapoints in memory, when those are requested, it __getitem__() feches them in memroy
        # instead of from disk.
        self.cache = {}

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
                cloud1 = data["pos1"].astype("float32").transpose(1, 0)
                cloud2 = data["pos2"].astype("float32").transpose(1, 0)
                flow = data["gt"].astype("float32").transpose(1, 0)
                mask = np.ones_like(flow[0, :])

            return cloud1, cloud2, flow, mask

        if self.load_data_in_memory:
            if idx in self.cache:
                cloud1, cloud2, flow, mask = self.cache[idx]
            else:
                cloud1, cloud2, flow, mask = get_data(idx)

                # filter out points too far away
                valid_mask = (abs(cloud1[:3, :]) < self.max_distance).all(axis=0)
                cloud1 = cloud1[:, valid_mask]
                flow = flow[:, valid_mask]
                mask = mask[valid_mask]

                valid_mask = (abs(cloud2[:3, :]) < self.max_distance).all(axis=0)
                cloud2 = cloud2[:, valid_mask]

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
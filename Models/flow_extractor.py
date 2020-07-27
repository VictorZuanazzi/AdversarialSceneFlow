"""Implemented by Victor Zuanazzi
Implemented models that learn 3D flow from point clouds. Mostly focused on LiDAR.
All done using the Pytorch framework."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from kaolin.models import PointNet
import pdb
import sys
sys.path.insert(0, 'PointPWC')

# import external models
from PointPWC.models import PointConvSceneFlowPWC8192selfglobalPointConv as PWC
from flownet_pytorch.model import FlowNet3D


def include_temp_embedding(x, n):
    """concatenate a time embeeding in the point cloud.
    x: torch.tensor(B, C, N) the point cloud,
    n: number of points"""
    # creates the temporal embedding
    t_emb = torch.arange(x.shape[2], device=x.device, dtype=x.dtype) // n
    t_emb = t_emb.view(-1, 1, x.shape[2])

    # batch the temporal embedding
    temp_emb = torch.cat([t_emb] * x.shape[0], dim=0)

    # concatenate it to the input tensor
    x_t = torch.cat((x, temp_emb), dim=1)
    return x_t


def remove_temp_embedding(x):
    return x[:, :-1, :]


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)


# I am well aware that monkey patching isn't among the best programming practices,
# Kaolin does not give me another option though!
class PointNetSegmenter(PointNet.PointNetSegmenter):
    """PointNetSegmenter without the annoying print!"""

    def forward(self, x):
        r"""Forward pass through the PointNet segmentation model.

                Args:
                    x (torch.Tensor): Tensor representing a pointcloud
                        shape: :math:`B \times N \times D`, where :math:`B`
                        is the batchsize, :math:`N` is the number of points
                        in the pointcloud, and :math:`D` is the dimensionality
                        of each point in the pointcloud.
                        If self.transposed_input is True, then the shape is
                        :math:`B \times D \times N`.

                """
        batchsize = x.shape[0]
        num_points = x.shape[2] if self.transposed_input else x.shape[1]
        x = self.feature_extractor(x)
        for idx in range(len(self.conv_layers)):
            if self.batchnorm:
                x = self.activation(self.bn_layers[idx](
                    self.conv_layers[idx](x)))
            else:
                x = self.activation(self.conv_layers[idx](x))
        x = self.last_conv_layer(x)
        x = x.transpose(2, 1).contiguous()
        return x.view(batchsize, num_points, self.num_classes)


class Dummy(nn.Module):
    def __init__(self, n_points):
        """Dummy model """
        super(Dummy, self).__init__()
        self.n_points = n_points
        self.lin = nn.Linear(1, 1)

    def forward(self, x):
        return x[:, :, :self.n_points], x[:, :, :self.n_points]


class ZeroFlow(nn.Module):
    def __init__(self):
        """Predict zero flow"""
        super(ZeroFlow, self).__init__()

    def forward(self, pc1, pc2):
        return torch.zeros_like(pc1)


class AvgFlow(nn.Module):
    def __init__(self):
        """Predict average flow"""
        super(AvgFlow, self).__init__()

    def forward(self, pc1, pc2):

        avg_mv = pc2.mean(dim=-1, keepdim=True) - pc1.mean(dim=-1, keepdim=True)
        flow = torch.ones_like(pc1) * avg_mv

        return flow


class knnFlow(nn.Module):
    def __init__(self, radius=0.00125):
        """Predict nearest neighgbor flow"""
        super(knnFlow, self).__init__()
        self.radius=radius

    def forward(self, pc1, pc2):
        B, D, N1 = pc1.shape
        N2 = pc2.shape[-1]
        pc1_e = pc1.unsqueeze(-1).expand(B, D, N1, N2)
        pc2_e = pc2.unsqueeze(-2).expand(B, D, N1, N2)

        # calculate distances
        pc_diff = pc2_e - pc1_e
        dists_pt = pc_diff.norm(dim=1)
        mask = dists_pt == dists_pt.min(dim=-1, keepdim=True)[0]
        flow = (pc_diff * mask.unsqueeze(1)).sum(dim=-1)

        return flow


class Segmenter(nn.Module):
    """Flow Extractor implemented using a PointNet Segmenter"""
    def __init__(self, n_points, n_sweeps=2, in_channels=3,
                 feat_size=1024,
                 num_classes=3,
                 classifier_layer_dims=[512, 256],
                 feat_layer_dims=[64, 128],
                 activation=F.relu,
                 batchnorm=True,
                 transposed_input=True):
        super(Segmenter, self).__init__()

        self.n_sweeps = n_sweeps
        self.n_points = n_points

        # layer responsible to find the flow
        self.flow_layer = PointNetSegmenter(in_channels=in_channels + 1,
                                            feat_size=feat_size,
                                            num_classes=num_classes,
                                            classifier_layer_dims=classifier_layer_dims,
                                            feat_layer_dims=feat_layer_dims,
                                            activation=activation,
                                            batchnorm=batchnorm,
                                            transposed_input=transposed_input)

    def forward(self, clouds, c2=None):
        if c2 is not None:
            clouds = torch.cat((clouds, c2), dim=-1)

        clouds = include_temp_embedding(clouds, self.n_points)

        # find flow
        flow_all = self.flow_layer(clouds)
        flow_all.transpose_(1, 2)
        clouds = remove_temp_embedding(clouds)

        # return the flow wrt the first point cloud
        n_points = clouds.shape[-1] // self.n_sweeps
        flow_1 = flow_all[:, :, :n_points]

        return flow_1


class PPWC(nn.Module):
    """Warper around PointPWC-net for use in our setup"""
    def __init__(self):
        super(PPWC, self).__init__()
        self.pointpwc = PWC()

    def forward(self, pc1, pc2):

        pc1.transpose_(1, 2)
        pc2.transpose_(1, 2)
        flows, _, _, _, _ = self.pointpwc(pc1.contiguous(), pc2.contiguous(),
                                          pc1.clone().contiguous(), pc2.clone().contiguous())

        return flows[0]


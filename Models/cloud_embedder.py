import torch.nn as nn
import torch.nn.functional as F
from kaolin.models.PointNet import PointNetFeatureExtractor
import sys
sys.path.insert(0, 'PointPWC')
from PointPWC.models import PointConvSceneFlowPWC8192selfglobalPointConv as PWC
from flownet_pytorch.util import PointNetSetAbstraction, PointNetFeaturePropogation, FlowEmbedding, PointNetSetUpConv


class PointNetFlowFeature(nn.Module):
    def __init__(self, use_batch_norm=True, feat_dim=4096):
        super(PointNetFlowFeature, self).__init__()

        self.feat_layer_1 = PointNetFeatureExtractor(in_channels=3,
                                                     feat_size=256,  # 512
                                                     layer_dims=[64, 128],  # [128, 256, 512],
                                                     global_feat=False,  # True
                                                     activation=F.relu,
                                                     batchnorm=use_batch_norm,
                                                     transposed_input=True)

        self.feat_layer_2 = PointNetFeatureExtractor(in_channels=256 + 64,
                                                     feat_size=feat_dim // 4,
                                                     layer_dims=[512, 1024],
                                                     global_feat=True,
                                                     activation=F.relu,
                                                     batchnorm=use_batch_norm,
                                                     transposed_input=True)

        self.feat_layer_3 = nn.Sequential(nn.Linear(in_features=feat_dim // 4, out_features=feat_dim // 4),
                                          nn.BatchNorm1d(feat_dim // 4), nn.ReLU(),
                                          nn.Linear(in_features=feat_dim // 4, out_features=feat_dim // 2),
                                          nn.BatchNorm1d(feat_dim // 2), nn.ReLU())

        self.feat_layer_4 = nn.Sequential(nn.Linear(in_features=feat_dim // 2, out_features=feat_dim),
                                          nn.BatchNorm1d(feat_dim), nn.ReLU(),
                                          nn.Linear(in_features=feat_dim, out_features=feat_dim))

    def forward(self, _, cloud):
        f1 = self.feat_layer_1(cloud)
        f2 = self.feat_layer_2(f1)
        f3 = self.feat_layer_3(f2)
        f4 = self.feat_layer_4(f3)

        # flattens f1 into a vector
        f1_vec = f1.max(dim=-1)[0]

        return f4, (f3, f2, f1_vec)


class PointNetFlowFeature2c(nn.Module):
    def __init__(self, use_batch_norm=True, feat_dim=4096):
        super(PointNetFlowFeature2c, self).__init__()

        self.feat_layer_1 = PointNetFeatureExtractor(in_channels=4,
                                                     feat_size=256,  # 512
                                                     layer_dims=[64, 128],  # [128, 256, 512],
                                                     global_feat=False,  # True
                                                     activation=F.relu,
                                                     batchnorm=use_batch_norm,
                                                     transposed_input=True)

        self.feat_layer_2 = PointNetFeatureExtractor(in_channels=256 + 64,
                                                     feat_size=feat_dim // 4,
                                                     layer_dims=[512, 1024],
                                                     global_feat=True,
                                                     activation=F.relu,
                                                     batchnorm=use_batch_norm,
                                                     transposed_input=True)

        self.feat_layer_3 = nn.Sequential(nn.Linear(in_features=feat_dim // 4, out_features=feat_dim // 4),
                                          nn.BatchNorm1d(feat_dim // 4), nn.ReLU(),
                                          nn.Linear(in_features=feat_dim // 4, out_features=feat_dim // 2),
                                          nn.BatchNorm1d(feat_dim // 2), nn.ReLU())

        self.feat_layer_4 = nn.Sequential(nn.Linear(in_features=feat_dim // 2, out_features=feat_dim),
                                          nn.BatchNorm1d(feat_dim), nn.ReLU(),
                                          nn.Linear(in_features=feat_dim, out_features=feat_dim))

    def forward(self, c1, c2):
        c1_t = torch.cat((c1, torch.zeros(c1.shape[0], 1, c1.shape[2], device=c1.device)), dim=1)
        c2_t = torch.cat((c2, torch.ones(c2.shape[0], 1, c2.shape[2], device=c2.device)), dim=1)
        f1_c1 = self.feat_layer_1(c1_t)
        f1_c2 = self.feat_layer_1(c2_t)

        f1 = torch.cat((f1_c1, f1_c2), dim=2)

        f2 = self.feat_layer_2(f1)
        f3 = self.feat_layer_3(f2)
        f4 = self.feat_layer_4(f3)

        # flattens f1 into a vector
        f1_vec = f1.max(dim=-1)[0]

        return f4, (f3, f2, f1_vec)


class FlowNet3Discriminator(nn.Module):
    def __init__(self, feat_dim=4096, n_points=1024):
        super(FlowNet3Discriminator, self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=n_points, radius=0.5, nsample=16, in_channel=3, mlp=[32, 32, 64],
                                          group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=1.0, nsample=16, in_channel=64, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=2.0, nsample=8, in_channel=128, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=4.0, nsample=8, in_channel=256, mlp=[256, 256, 512],
                                          group_all=False)

        self.fe_layer = FlowEmbedding(radius=10.0, nsample=64, in_channel=128, mlp=[128, 128, 128], pooling='max',
                                      corr_func='concat')

        h_dim = 2 * 64 + 2 * 128 + 256 + 512 + 128

        self.seq1 = nn.Sequential(nn.Linear(in_features=h_dim, out_features=feat_dim // 4),
                                  nn.BatchNorm1d(feat_dim // 4), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 4, out_features=feat_dim // 2),
                                  nn.BatchNorm1d(feat_dim // 2), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 2, out_features=feat_dim))

    def forward(self, pc1, pc2):
        # pdb.set_trace()

        pc2_no_grad = pc2.clone()
        pc2_no_grad.detach_()

        # get feature embedding of the input
        l1_pc1, l1_f1 = self.sa1(pc1, pc1)
        l2_pc1, l2_f1 = self.sa2(l1_pc1, l1_f1)
        l1_pc2, l1_f2 = self.sa1(pc2_no_grad, pc2)
        l2_pc2, l2_f2 = self.sa2(l1_pc2, l1_f2)
        _, pc_flow = self.fe_layer(l2_pc1, l2_pc2, l2_f1, l2_f2)
        l3_pc2, l3_f = self.sa3(l2_pc2, pc_flow)
        l4_pc2, l4_f = self.sa4(l3_pc2, l3_f)

        l_flat = torch.cat((l1_f1.max(dim=-1)[0], l2_f1.max(dim=-1)[0], l1_f2.max(dim=-1)[0], l2_f2.max(dim=-1)[0],
                            pc_flow.max(dim=-1)[0], l3_f.max(dim=-1)[0], l4_f.max(dim=-1)[0]), dim=-1)

        out = self.seq1(l_flat)

        hidden_flat = (l4_f.mean(dim=-1), l3_f.mean(dim=-1), pc_flow.mean(dim=-1),
                       l2_f2.mean(dim=-1), l1_f2.mean(dim=-1),
                       l2_f1.mean(dim=-1), l1_f1.mean(dim=-1))

        return out, hidden_flat


class FlowNet3DiscriminatorHCat(nn.Module):
    def __init__(self, feat_dim=4096, n_points=1024):
        super(FlowNet3DiscriminatorHCat, self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=n_points, radius=0.5, nsample=16, in_channel=3,
                                          mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=1.0, nsample=16, in_channel=64, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=2.0, nsample=8, in_channel=128, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=4.0, nsample=8, in_channel=256, mlp=[256, 256, 512],
                                          group_all=False)

        self.fe_layer = FlowEmbedding(radius=10.0, nsample=64, in_channel=128, mlp=[128, 128, 128], pooling='max',
                                      corr_func='concat')

        h1 = 64
        h2 = 128 + 2 * h1
        hf = 128 + 2 * h2
        hf_out = max(feat_dim // 4, hf)
        h3 = 256 + hf_out
        h3_out = feat_dim // 2
        h4 = 512 + h3_out
        h4_out = feat_dim

        self.fc1 = nn.Sequential(nn.Linear(in_features=h1, out_features=h1), nn.BatchNorm1d(h1), nn.ReLU(),
                                 nn.Linear(in_features=h1, out_features=h1), nn.BatchNorm1d(h1), nn.ReLU())

        self.fc2 = nn.Sequential(nn.Linear(in_features=h2, out_features=h2), nn.BatchNorm1d(h2), nn.ReLU(),
                                 nn.Linear(in_features=h2, out_features=h2), nn.BatchNorm1d(h2), nn.ReLU())

        self.fcf = nn.Sequential(nn.Linear(in_features=hf, out_features=hf), nn.BatchNorm1d(hf), nn.ReLU(),
                                 nn.Linear(in_features=hf, out_features=hf_out), nn.BatchNorm1d(hf_out), nn.ReLU())

        self.fc3 = nn.Sequential(nn.Linear(in_features=h3, out_features=h3), nn.BatchNorm1d(h3), nn.ReLU(),
                                 nn.Linear(in_features=h3, out_features=h3_out), nn.BatchNorm1d(h3_out), nn.ReLU())

        self.fc4 = nn.Sequential(nn.Linear(in_features=h4, out_features=h4), nn.BatchNorm1d(h4), nn.ReLU(),
                                 nn.Linear(in_features=h4, out_features=h4_out))

    def forward(self, pc1, pc2):
        # pdb.set_trace()
        pc2_no_grad = pc2.clone()
        pc2_no_grad.detach_()

        # get feature embedding of the input
        l1_pc1, l1_f1 = self.sa1(pc1, pc1)
        l2_pc1, l2_f1 = self.sa2(l1_pc1, l1_f1)
        l1_pc2, l1_f2 = self.sa1(pc2_no_grad, pc2)
        l2_pc2, l2_f2 = self.sa2(l1_pc2, l1_f2)
        _, pc_flow = self.fe_layer(l2_pc1, l2_pc2, l2_f1, l2_f2)
        l3_pc2, l3_f = self.sa3(l2_pc2, pc_flow)
        l4_pc2, l4_f = self.sa4(l3_pc2, l3_f)

        f1_c1 = self.fc1(l1_f1.max(dim=-1)[0])
        f1_c2 = self.fc1(l1_f2.max(dim=-1)[0])
        f1 = torch.cat((f1_c1, f1_c2), dim=1)

        f2_c1 = self.fc2(torch.cat((l2_f1.max(dim=-1)[0], f1), dim=1))
        f2_c2 = self.fc2(torch.cat((l2_f2.max(dim=-1)[0], f1), dim=1))
        f2 = torch.cat((f2_c1, f2_c2), dim=1)

        ff = self.fcf(torch.cat((pc_flow.max(dim=-1)[0], f2), dim=1))
        f3 = self.fc3(torch.cat((l3_f.max(dim=-1)[0], ff), dim=1))
        out = self.fc4(torch.cat((l4_f.max(dim=-1)[0], f3), dim=1))

        hidden_flat = (f3, ff, f2, f1,
                       l4_f.mean(dim=-1), l3_f.mean(dim=-1), pc_flow.mean(dim=-1),
                       l2_f2.mean(dim=-1), l1_f2.mean(dim=-1),
                       l2_f1.mean(dim=-1), l1_f1.mean(dim=-1))

        return out, hidden_flat


class FlowNet3DiscriminatorDiff(nn.Module):
    def __init__(self, feat_dim=4096, n_points=1024):
        super(FlowNet3DiscriminatorDiff, self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=n_points, radius=0.5, nsample=16, in_channel=3,
                                          mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=1.0, nsample=16, in_channel=64, mlp=[64, 64, 128],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=2.0, nsample=8, in_channel=128, mlp=[128, 128, 256],
                                          group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=4.0, nsample=8, in_channel=256, mlp=[256, 256, 512],
                                          group_all=False)

        self.fe_layer = FlowEmbedding(radius=10.0, nsample=64, in_channel=128, mlp=[128, 128, 128], pooling='max',
                                      corr_func='concat')

        h1 = 64
        h2 = 128
        hf = 128 + h2 + h1
        hf_out = max(feat_dim // 4, hf)
        h3 = 256 + hf_out
        h3_out = feat_dim // 2
        h4 = 512 + h3_out
        h4_out = feat_dim

        self.fc1 = nn.Sequential(nn.Linear(in_features=h1, out_features=h1), nn.BatchNorm1d(h1), nn.ReLU(),
                                 nn.Linear(in_features=h1, out_features=h1), nn.BatchNorm1d(h1), nn.ReLU())

        self.fc2 = nn.Sequential(nn.Linear(in_features=h2, out_features=h2), nn.BatchNorm1d(h2), nn.ReLU(),
                                 nn.Linear(in_features=h2, out_features=h2), nn.BatchNorm1d(h2), nn.ReLU())

        self.fcf = nn.Sequential(nn.Linear(in_features=hf, out_features=hf), nn.BatchNorm1d(hf), nn.ReLU(),
                                 nn.Linear(in_features=hf, out_features=hf_out), nn.BatchNorm1d(hf_out), nn.ReLU())

        self.fc3 = nn.Sequential(nn.Linear(in_features=h3, out_features=h3), nn.BatchNorm1d(h3), nn.ReLU(),
                                 nn.Linear(in_features=h3, out_features=h3_out), nn.BatchNorm1d(h3_out), nn.ReLU())

        self.fc4 = nn.Sequential(nn.Linear(in_features=h4, out_features=h4), nn.BatchNorm1d(h4), nn.ReLU(),
                                 nn.Linear(in_features=h4, out_features=h4_out))

    def forward(self, pc1, pc2):
        # pdb.set_trace()

        pc2_no_grad = pc2.clone()
        pc2_no_grad.detach_()

        # get feature embedding of the input
        l1_pc1, l1_f1 = self.sa1(pc1, pc1)
        l2_pc1, l2_f1 = self.sa2(l1_pc1, l1_f1)
        l1_pc2, l1_f2 = self.sa1(pc2_no_grad, pc2)
        l2_pc2, l2_f2 = self.sa2(l1_pc2, l1_f2)
        _, pc_flow = self.fe_layer(l2_pc1, l2_pc2, l2_f1, l2_f2)
        l3_pc2, l3_f = self.sa3(l2_pc2, pc_flow)
        l4_pc2, l4_f = self.sa4(l3_pc2, l3_f)

        f1_c1 = self.fc1(l1_f1.max(dim=-1)[0])
        f1_c2 = self.fc1(l1_f2.max(dim=-1)[0])
        f1_diff = f1_c2 - f1_c1

        f2_c1 = self.fc2(l2_f1.max(dim=-1)[0])
        f2_c2 = self.fc2(l2_f2.max(dim=-1)[0])
        f2_diff = f2_c2 - f2_c1

        ff = self.fcf(torch.cat((pc_flow.max(dim=-1)[0], f2_diff, f1_diff), dim=1))
        f3 = self.fc3(torch.cat((l3_f.max(dim=-1)[0], ff), dim=1))
        out = self.fc4(torch.cat((l4_f.max(dim=-1)[0], f3), dim=1))

        hidden_flat = (f3, ff, f2_diff, f1_diff,
                       l4_f.mean(dim=-1), l3_f.mean(dim=-1), pc_flow.mean(dim=-1),
                       l2_f2.mean(dim=-1), l1_f2.mean(dim=-1),
                       l2_f1.mean(dim=-1), l1_f1.mean(dim=-1))

        return out, hidden_flat


class FlowNet3DFeature(nn.Module):
    def __init__(self, feat_dim=4096, n_points=1024):
        super(FlowNet3DFeature, self).__init__()

        feat_dim = max(64, feat_dim)

        self.sa1 = PointNetSetAbstraction(npoint=n_points, radius=0.5, nsample=16, in_channel=3,
                                          mlp=[32, 64, feat_dim // 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=1.0, nsample=16, in_channel=feat_dim // 64,
                                          mlp=[feat_dim // 64, 128, feat_dim // 16], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=2.0, nsample=8, in_channel=feat_dim // 16,
                                          mlp=[feat_dim // 16, 512, feat_dim // 4], group_all=False)

        self.seq1 = nn.Sequential(nn.Linear(in_features=feat_dim // 64, out_features=feat_dim // 32),
                                  nn.BatchNorm1d(feat_dim // 32), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 32, out_features=feat_dim // 16),
                                  nn.BatchNorm1d(feat_dim // 16), nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(in_features=feat_dim // 16, out_features=feat_dim // 8),
                                  nn.BatchNorm1d(feat_dim // 8), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 8, out_features=feat_dim // 4),
                                  nn.BatchNorm1d(feat_dim // 4), nn.ReLU())
        self.seq3 = nn.Sequential(nn.Linear(in_features=feat_dim // 4, out_features=feat_dim // 2),
                                  nn.BatchNorm1d(feat_dim // 2), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 2, out_features=feat_dim))

    def forward(self, _, pc2):
        # pdb.set_trace()

        pc2_no_grad = pc2.clone()
        pc2_no_grad.detach_()

        # get feature embedding of the input pointcloud
        l1_pc2, l1_feature2 = self.sa1(pc2_no_grad, pc2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)
        l3_pc2, l3_feature2 = self.sa3(l2_pc2, l2_feature2)

        # fine expansion of the max feature embeddings
        f1 = self.seq1(l1_feature2.max(dim=-1)[0])
        f2 = self.seq2(l2_feature2.max(dim=-1)[0] + f1)
        out = self.seq3(l3_feature2.max(dim=-1)[0] + f2)

        hidden_flat = (f2, f1,
                       l3_feature2.mean(dim=-1),
                       l2_feature2.mean(dim=-1),
                       l1_feature2.mean(dim=-1))

        return out, hidden_flat


class FlowNet3DFeatureCat(nn.Module):
    def __init__(self, feat_dim=4096, n_points=1024):
        super(FlowNet3DFeatureCat, self).__init__()

        feat_dim = max(64, feat_dim)

        self.sa1 = PointNetSetAbstraction(npoint=n_points, radius=0.5, nsample=16, in_channel=3,
                                          mlp=[32, 64, feat_dim // 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=1.0, nsample=16, in_channel=feat_dim // 64,
                                          mlp=[feat_dim // 64, 128, feat_dim // 16], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=2.0, nsample=8, in_channel=feat_dim // 16,
                                          mlp=[feat_dim // 16, 512, feat_dim // 4], group_all=False)

        self.seq1 = nn.Sequential(nn.Linear(in_features=feat_dim // 64, out_features=feat_dim // 32),
                                  nn.BatchNorm1d(feat_dim // 32), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 32, out_features=feat_dim // 16),
                                  nn.BatchNorm1d(feat_dim // 16), nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(in_features=feat_dim // 8, out_features=feat_dim // 8),
                                  nn.BatchNorm1d(feat_dim // 8), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 8, out_features=feat_dim // 4),
                                  nn.BatchNorm1d(feat_dim // 4), nn.ReLU())
        self.seq3 = nn.Sequential(nn.Linear(in_features=feat_dim // 2, out_features=feat_dim // 2),
                                  nn.BatchNorm1d(feat_dim // 2), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 2, out_features=feat_dim))

    def forward(self, _, pc2):
        # pdb.set_trace()

        pc2_no_grad = pc2.clone()
        pc2_no_grad.detach_()

        # get feature embedding of the input pointcloud
        l1_pc2, l1_feature2 = self.sa1(pc2_no_grad, pc2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)
        l3_pc2, l3_feature2 = self.sa3(l2_pc2, l2_feature2)

        # fine expansion of the max feature embeddings
        f1 = self.seq1(l1_feature2.max(dim=-1)[0])
        f2 = self.seq2(torch.cat((l2_feature2.max(dim=-1)[0], f1), dim=1))
        out = self.seq3(torch.cat((l3_feature2.max(dim=-1)[0], f2), dim=1))

        hidden_flat = (f2, f1,
                       l3_feature2.mean(dim=-1),
                       l2_feature2.mean(dim=-1),
                       l1_feature2.mean(dim=-1))

        return out, hidden_flat


class FlowNet3DFeatureCatAtt(nn.Module):
    def __init__(self, feat_dim=4096, n_points=1024):
        super(FlowNet3DFeatureCat, self).__init__()

        feat_dim = max(64, feat_dim)

        self.sa1 = PointNetSetAbstraction(npoint=n_points, radius=0.5, nsample=16, in_channel=3,
                                          mlp=[32, 64, feat_dim // 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=1.0, nsample=16, in_channel=feat_dim // 64,
                                          mlp=[feat_dim // 64, 128, feat_dim // 16], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=2.0, nsample=8, in_channel=feat_dim // 16,
                                          mlp=[feat_dim // 16, 512, feat_dim // 4], group_all=False)

        self.transformer = nn.Transformer(d_model=feat_dim // 4, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
                                          dim_feedforward=2048, dropout=0.1, activation='relu',
                                          custom_encoder=None, custom_decoder=None)


        self.seq1 = nn.Sequential(nn.Linear(in_features=feat_dim // 64, out_features=feat_dim // 32),
                                  nn.BatchNorm1d(feat_dim // 32), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 32, out_features=feat_dim // 16),
                                  nn.BatchNorm1d(feat_dim // 16), nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(in_features=feat_dim // 8, out_features=feat_dim // 8),
                                  nn.BatchNorm1d(feat_dim // 8), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 8, out_features=feat_dim // 4),
                                  nn.BatchNorm1d(feat_dim // 4), nn.ReLU())
        self.seq3 = nn.Sequential(nn.Linear(in_features=feat_dim // 2, out_features=feat_dim // 2),
                                  nn.BatchNorm1d(feat_dim // 2), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 2, out_features=feat_dim))

    def forward(self, _, pc2):
        # pdb.set_trace()

        pc2_no_grad = pc2.clone()
        pc2_no_grad.detach_()

        # get feature embedding of the input pointcloud
        l1_pc2, l1_feature2 = self.sa1(pc2_no_grad, pc2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)
        l3_pc2, l3_feature2 = self.sa3(l2_pc2, l2_feature2)

        att = self.transformer(l3_feature2.permute(2, 0, 1), l3_feature2.permute(2, 0, 1))
        l3_feature2 = l3_feature2 * att.permute(1, 2, 0).sigmoid_()

        # fine expansion of the max feature embeddings
        f1 = self.seq1(l1_feature2.max(dim=-1)[0])
        f2 = self.seq2(torch.cat((l2_feature2.max(dim=-1)[0], f1), dim=1))
        out = self.seq3(torch.cat((l3_feature2.max(dim=-1)[0], f2), dim=1))

        hidden_flat = (f2, f1,
                       l3_feature2.mean(dim=-1),
                       l2_feature2.mean(dim=-1),
                       l1_feature2.mean(dim=-1))

        return out, hidden_flat


class FlowNet3DFeatureCatFat(nn.Module):
    def __init__(self, feat_dim=4096, n_points=1024):
        super(FlowNet3DFeatureCatFat, self).__init__()

        feat_dim = max(64, feat_dim)

        self.sa1 = PointNetSetAbstraction(npoint=n_points, radius=0.5, nsample=16, in_channel=3,
                                          mlp=[32, 64, feat_dim // 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=1.0, nsample=16, in_channel=feat_dim // 64,
                                          mlp=[feat_dim // 64, 128, feat_dim // 16], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=2.0, nsample=8, in_channel=feat_dim // 16,
                                          mlp=[feat_dim // 16, 512, feat_dim // 4], group_all=False)

        self.seq1 = nn.Sequential(nn.Linear(in_features=feat_dim // 64, out_features=feat_dim // 64),
                                  nn.BatchNorm1d(feat_dim // 64), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 64, out_features=feat_dim // 32),
                                  nn.BatchNorm1d(feat_dim // 32), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 32, out_features=feat_dim // 16),
                                  nn.BatchNorm1d(feat_dim // 16), nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(in_features=feat_dim // 8, out_features=feat_dim // 8),
                                  nn.BatchNorm1d(feat_dim // 8), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 8, out_features=feat_dim // 8),
                                  nn.BatchNorm1d(feat_dim // 8), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 8, out_features=feat_dim // 4),
                                  nn.BatchNorm1d(feat_dim // 4), nn.ReLU())
        self.seq3 = nn.Sequential(nn.Linear(in_features=feat_dim // 2, out_features=feat_dim // 2),
                                  nn.BatchNorm1d(feat_dim // 2), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 2, out_features=feat_dim // 2),
                                  nn.BatchNorm1d(feat_dim // 2), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 2, out_features=feat_dim))

    def forward(self, _, pc2):
        # pdb.set_trace()

        pc2_no_grad = pc2.clone()
        pc2_no_grad.detach_()

        # get feature embedding of the input pointcloud
        l1_pc2, l1_feature2 = self.sa1(pc2_no_grad, pc2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)
        l3_pc2, l3_feature2 = self.sa3(l2_pc2, l2_feature2)

        # fine expansion of the max feature embeddings
        f1 = self.seq1(l1_feature2.max(dim=-1)[0])
        f2 = self.seq2(torch.cat((l2_feature2.max(dim=-1)[0], f1), dim=1))
        out = self.seq3(torch.cat((l3_feature2.max(dim=-1)[0], f2), dim=1))

        hidden_flat = (f2, f1,
                       l3_feature2.mean(dim=-1),
                       l2_feature2.mean(dim=-1),
                       l1_feature2.mean(dim=-1))

        return out, hidden_flat


class FlowNet3DFeatureCat4l(nn.Module):
    def __init__(self, feat_dim=4096, n_points=1024):
        super(FlowNet3DFeatureCat4l, self).__init__()

        feat_dim = max(64, feat_dim)

        self.sa1 = PointNetSetAbstraction(npoint=n_points, radius=0.5, nsample=16, in_channel=3,
                                          mlp=[32, 32, feat_dim // 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=1.0, nsample=16, in_channel=feat_dim // 64,
                                          mlp=[feat_dim // 64, feat_dim // 64, feat_dim // 16],
                                          group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=2.0, nsample=8, in_channel=feat_dim // 16,
                                          mlp=[feat_dim // 16, feat_dim // 16, feat_dim // 4],
                                          group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=4.0, nsample=8, in_channel=feat_dim // 4,
                                          mlp=[feat_dim // 4, feat_dim // 4, feat_dim // 2],
                                          group_all=False)

        self.seq1 = nn.Sequential(nn.Linear(in_features=feat_dim // 64, out_features=feat_dim // 32),
                                  nn.BatchNorm1d(feat_dim // 32), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 32, out_features=feat_dim // 16),
                                  nn.BatchNorm1d(feat_dim // 16), nn.ReLU())
        self.seq2 = nn.Sequential(nn.Linear(in_features=feat_dim // 8, out_features=feat_dim // 8),
                                  nn.BatchNorm1d(feat_dim // 8), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 8, out_features=feat_dim // 4),
                                  nn.BatchNorm1d(feat_dim // 4), nn.ReLU())
        self.seq3 = nn.Sequential(nn.Linear(in_features=feat_dim // 2, out_features=feat_dim // 2),
                                  nn.BatchNorm1d(feat_dim // 2), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim // 2, out_features=feat_dim // 2),
                                  nn.BatchNorm1d(feat_dim // 2), nn.ReLU(), )
        self.seq4 = nn.Sequential(nn.Linear(in_features=feat_dim, out_features=feat_dim),
                                  nn.BatchNorm1d(feat_dim), nn.ReLU(),
                                  nn.Linear(in_features=feat_dim, out_features=feat_dim))

    def forward(self, _, pc2):
        # pdb.set_trace()
        pc2_no_grad = pc2.clone()
        pc2_no_grad.detach_()

        # get feature embedding of the input pointcloud
        l1_pc2, l1_feature2 = self.sa1(pc2_no_grad, pc2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)
        l3_pc2, l3_feature2 = self.sa3(l2_pc2, l2_feature2)
        l4_pc2, l4_feature2 = self.sa4(l3_pc2, l3_feature2)

        # fine expansion of the max feature embeddings
        f1 = self.seq1(l1_feature2.max(dim=-1)[0])
        f2 = self.seq2(torch.cat((l2_feature2.max(dim=-1)[0], f1), dim=1))
        f3 = self.seq3(torch.cat((l3_feature2.max(dim=-1)[0], f2), dim=1))
        out = self.seq4(torch.cat((l4_feature2.max(dim=-1)[0], f3), dim=1))

        hidden_flat = (f3, f2, f1,
                       l4_feature2.mean(dim=-1),
                       l3_feature2.mean(dim=-1),
                       l2_feature2.mean(dim=-1),
                       l1_feature2.mean(dim=-1))

        return out, hidden_flat


class FlowNet3DDiscriminatorP(nn.Module):
    def __init__(self, feature_dim=3, n_points=1024):
        super(FlowNet3DDiscriminatorP, self).__init__()

        self.sa1 = PointNetSetAbstraction(npoint=n_points, radius=0.5, nsample=16,
                                          in_channel=feature_dim, mlp=[32, 32, 64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=256, radius=1.0, nsample=16,
                                          in_channel=64, mlp=[64, 64, 128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=64, radius=2.0, nsample=8,
                                          in_channel=128, mlp=[128, 128, 256], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=16, radius=4.0, nsample=8,
                                          in_channel=256, mlp=[256, 256, 512], group_all=False)

        self.fe_layer = FlowEmbedding(radius=10.0, nsample=64, in_channel=128, mlp=[128, 128, 128], pooling='max',
                                      corr_func='concat')

        self.su1 = PointNetSetUpConv(nsample=8, radius=2.4, f1_channel=256, f2_channel=512, mlp=[], mlp2=[256, 256])
        self.su2 = PointNetSetUpConv(nsample=8, radius=1.2, f1_channel=128 + 128, f2_channel=256, mlp=[128, 128, 256],
                                     mlp2=[256])
        self.su3 = PointNetSetUpConv(nsample=8, radius=0.6, f1_channel=64, f2_channel=256, mlp=[128, 128, 256],
                                     mlp2=[256])
        self.fp = PointNetFeaturePropogation(in_channel=256 + feature_dim, mlp=[256, 256])

        self.conv1 = nn.Conv1d(256, 128, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=1, bias=True)

    def forward(self, pc1, pc2):
        pc2_nograd = pc2.clone().detach()

        # pdb.set_trace()
        l1_pc1, l1_feature1 = self.sa1(pc1, pc1)
        l2_pc1, l2_feature1 = self.sa2(l1_pc1, l1_feature1)

        # embedd anchor
        l1_pc2, l1_feature2 = self.sa1(pc2_nograd, pc2)
        l2_pc2, l2_feature2 = self.sa2(l1_pc2, l1_feature2)

        _, l2_feature1_new = self.fe_layer(l2_pc1, l2_pc2, l2_feature1, l2_feature2)

        l3_pc1, l3_feature1 = self.sa3(l2_pc1, l2_feature1_new)
        l4_pc1, l4_feature1 = self.sa4(l3_pc1, l3_feature1)

        l3_fnew1 = self.su1(l3_pc1, l4_pc1, l3_feature1, l4_feature1)
        l2_fnew1 = self.su2(l2_pc1, l3_pc1, torch.cat([l2_feature1, l2_feature1_new], dim=1), l3_fnew1)
        l1_fnew1 = self.su3(l1_pc1, l2_pc1, l1_feature1, l2_fnew1)
        l0_fnew1 = self.fp(pc1, l1_pc1, pc1, l1_fnew1)

        x = F.relu(self.bn1(self.conv1(l0_fnew1)))
        sf = self.conv2(x)

        hidden_flat = (l0_fnew1.mean(dim=-1),
                       l1_fnew1.mean(dim=-1),
                       l2_fnew1.mean(dim=-1),
                       l3_fnew1.mean(dim=-1),
                       l4_feature1.mean(dim=-1),
                       l3_feature1.mean(dim=-1),
                       l2_feature2.mean(dim=-1),
                       l1_feature2.mean(dim=-1))

        return sf, hidden_flat


if __name__ == '__main__':
    import os
    import torch

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # input = torch.randn((8,3,2048))
    # label = torch.randn(8,16)

    pc1 = torch.zeros(2, 3, 8192)
    l1_pc1 = torch.zeros(2, 3, 8192)
    feature1 = None
    l1_fnew1 = torch.zeros(2, 256, 8192)

    layer = PointNetFeaturePropogation(in_channel=256, mlp=[256, 256])
    out = layer(pc1, l1_pc1, feature1, l1_fnew1)
    print(out.shape)

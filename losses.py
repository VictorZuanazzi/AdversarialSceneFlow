"""Implemented by Victor Zuanazzi
Implementation of different Loss functions for use in sets. In specif, the losses are designed to help the models to
learn dense motion (flow) in point clouds."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def inverse_triplet(dim=1, norm=2, max_margin=1.0):
    """Initialize triplet loss with dynamic margin, used in combination with triplet loss in adversarial triplet
        learning
    input:
        dim: int, the dimension to take the norm;
        norm: int, the norm type;
        max_marging: int, max dynamic margin
    return:
        funcction that calculates inverse triplet loss
    """

    def func(anchor, positive, negative):
        """implements inverse triplet margin, with dynamic margin.
        input:
            anchor: torch.tensor(B, Z), the anchor of the triplet.
            positive: torch.tensor(B, Z), the positive example.
            negative: torch.tensor(B, Z), the negative example.
        return:
            triplet distance, torch.tensor(B)"""

        d_p = (anchor - positive).norm(dim=dim, p=norm).mean()
        d_n = (anchor - negative).norm(dim=dim, p=norm).mean()

        return F.relu(d_p - d_n.clamp(min=0.0, max=max_margin))

    return func


def l2_loss(dim=1, norm=2):
    """l2 loss, or any other norm loss you want!"""

    def loss_func(p, t):
        return (p - t).norm(dim=dim, p=norm).mean()

    return loss_func


def l_inv_loss(dim=1, norm=2, pow=2, eps=1e-3):
    """ loss on the inverse of the inputs, or any other norm loss you want!"""

    def loss_func(p, t):
        t_norm_inv = 1 / (t.norm(dim=dim, p=pow, keepdim=True) + eps)
        return ((p - t) * t_norm_inv).norm(dim=dim, p=norm).mean()

    return loss_func


def cycons_func_(cycle_type):
    """cycle consistency loss
    cycle_type: str containing  any combination of 'cos', 'mse', 'l1', 'l2', 'lmax';
    return function that takes the cycle consistency loss"""
    loss_dict = {}
    if "cos" in cycle_type:
        cos_sim = torch.nn.CosineSimilarity()
        loss_dict["cos"] = lambda x, y: cos_sim(x, -y).mean()
    if "mse" in cycle_type:
        loss_dict["mse"] = torch.nn.MSELoss()
    if "l2" in cycle_type:
        loss_dict["l2"] = l2_loss(dim=1, norm=2)
    if "l1" in cycle_type:
        loss_dict["l1"] = l2_loss(dim=1, norm=1)
    if "lmax" in cycle_type:
        loss_dict["lmax"] = l2_loss(dim=1, norm=np.inf)

    def func(f, bf):
        loss = 0
        for l_type in loss_dict:
            loss = loss_dict[l_type](f, -bf) + loss
        return loss

    return func


def triplet_divergence(device, triplet=True, eps=1e-5):
    """JN divergence between multivariate gaussians with a diagonal covariance matrix"""

    def kl_divergence(mean_1, mean_2, std_1, std_2):
        return torch.log(std_2 / std_1) + ((std_1 ** 2) + ((mean_1 - mean_2) ** 2)) / (2 * (std_2 ** 2)) - (1 / 2)

    eps = torch.tensor([eps], device=device)

    def div_func_swap(anchor, positive, negative):

        B, S = anchor.shape
        half = S // 2
        mean_a, std_a = anchor[:, :half], torch.max(anchor[:, half:], eps)
        mean_p, std_p = positive[:, :half], torch.max(positive[:, half:], eps)
        mean_n, std_n = negative[:, :half], torch.max((negative[:, half:]), eps)

        jn_p = (kl_divergence(mean_a, mean_p, std_a, std_p) + kl_divergence(mean_p, mean_a, std_p, std_a)).sum(dim=1)
        jn_n1 = (kl_divergence(mean_a, mean_n, std_a, std_n) + kl_divergence(mean_n, mean_a, std_n, std_a)).sum(dim=1)
        jn_n2 = (kl_divergence(mean_p, mean_n, std_p, std_n) + kl_divergence(mean_n, mean_p, std_n, std_p)).sum(dim=1)

        return (jn_p - (jn_n1 + jn_n2) / 2).mean()

    def div_func_bi(anchor, positive, negative):

        B, S = anchor.shape
        half = S // 2
        mean_a, mean_p = anchor[:, :half], positive[:, :half]
        std_a, std_p = torch.max(anchor[:, half:], eps), torch.max(positive[:, half:], eps)

        jn_p = (kl_divergence(mean_a, mean_p, std_a, std_p) + kl_divergence(mean_p, mean_a, std_p, std_a)).sum(dim=1)

        return jn_p.mean()

    if triplet:
        return div_func_swap
    else:
        return div_func_bi


def deep_loss_(triplet_loss, use_deep_loss, device, scale_type=""):
    """implementation of the deep loss using triplet loss.
    input: triplet_loss(), or a function that takes 3 inputs torch.tensor(B, :) in the order
        (anchor, positive example, negative example)
    output: function that retuns the loss."""

    if scale_type == "shape":
        scale_func = lambda i, shape: 1. / shape
    elif scale_type == "shape_sub":
        scale_func = lambda i, shape: 1. / np.sqrt(shape)
    elif scale_type == "depth_sub":
        scale_func = lambda i, shape: 1. / np.sqrt(i + 2)
    elif scale_type == "depth_lin":
        scale_func = lambda i, shape: 1. / (i + 2)
    elif scale_type == "depth_sup":
        scale_func = lambda i, shape: 1. / ((i + 2) ** 2)
    elif scale_type == "":
        scale_func = lambda i, shape: 1.

    if use_deep_loss:
        def deep_loss_func(hidden_feats_0, hidden_feats_p, hidden_feats_n):
            loss_hidden_feats = torch.zeros(1, device=device)
            B = hidden_feats_0[0].shape[0]
            for i, (hf0, hfp, hfn) in enumerate(zip(hidden_feats_0, hidden_feats_p, hidden_feats_n)):
                shape = hf0.view(B, -1).shape[-1]
                loss_hidden_feats += (triplet_loss(hf0.view(B, -1),
                                                   hfp.view(B, -1),
                                                   hfn.view(B, -1)) * scale_func(i, shape))

            return loss_hidden_feats

    else:
        def deep_loss_func(hidden_feats_0, hidden_feats_p, hidden_feats_n):
            return torch.zeros(1, device=device)

    return deep_loss_func


def local_flow_consistency(pc1, flow_pred, n_samples=4, radius=0.0125):
    """calculate l2 distance between flow vectors of points inside a ball.
    input:
        pc1: torch.tensor(B, 3, N), point cloud which flow vectors were estimated.
        flow_pred: torch.tensor(B, 3, N), estimated flow vectors.
        n_samples: int number of points to query for each point in point cloud pc1.
        radius: max radius for the query, zeros will be used to pad in case not all n_samples were found in the ball.
    return:
        torch.tensor(1), l2 local consistency"""

    B, D, N1 = pc1.shape
    pc_pred = pc1 + flow_pred
    pc_pred_e = pc_pred.unsqueeze(-1).expand(B, D, N1, N1)

    pc_pred_diff = pc_pred_e - pc_pred_e.transpose(-2, -1)
    dists_pp = pc_pred_diff.norm(dim=1)
    mask_diag = 1 - torch.eye(N1, device=pc1.device).unsqueeze(0).expand(B, N1, N1)
    mask_pred = ((dists_pp / dists_pp.max(dim=-1, keepdim=True)[0]) < radius) * mask_diag

    flow_diffs = (flow_pred.unsqueeze(-1).expand(B, D, N1, N1) - flow_pred.unsqueeze(-2).expand(B, D, N1, N1)).norm(dim=1)
    flow_diffs_mean = (mask_pred * flow_diffs).sum(dim=-1) / mask_pred.sum(dim=-1).clamp(min=1)
    local_consistency_loss = flow_diffs_mean.sum(dim=-1)
    # local_consistency_loss = flow_diffs_mean.mean(dim=-1)

    return local_consistency_loss.mean()


def knn_loss(pc_pred, pc_target, n_samples=1, radius=1.0):
    """1-nearest neighbor loss"""

    B, D, N1 = pc_pred.shape
    N2 = pc_target.shape[-1]
    pc_pred_e = pc_pred.unsqueeze(-1).expand(B, D, N1, N2)
    pc_target_e = pc_target.unsqueeze(-2).expand(B, D, N1, N2)

    # calculate distances
    pc_pred_target_diff = pc_pred_e - pc_target_e
    dists_pt = pc_pred_target_diff.norm(dim=1)

    dist_pt = dists_pt.topk(k=1, dim=-1, largest=False)[0].squeeze()
    loss = dist_pt.mean()

    return loss


def chamfer_distance(pc_pred, pc_target):
    """chamfer distance between two point clouds"""

    B, D, N1 = pc_pred.shape
    N2 = pc_target.shape[-1]
    pc_pred_e = pc_pred.unsqueeze(-1).expand(B, D, N1, N2)
    pc_target_e = pc_target.unsqueeze(-2).expand(B, D, N1, N2)

    # calculate distances
    pc_pred_target_diff = pc_pred_e - pc_target_e
    dists_pt = pc_pred_target_diff.norm(dim=1)

    # chamfer distace:
    dist_pt = dists_pt.topk(k=1, dim=-1, largest=False)[0].squeeze()
    dist_tp = dists_pt.topk(k=1, dim=-2, largest=False)[0].squeeze()
    dist_chamfer = dist_pt.sum(dim=-1) + dist_tp.sum(dim=-1)

    return dist_chamfer.mean()


def laplacian_regularization(pc_pred, pc_target, treshold=0.0125):
    """As discribed in PointPWC paper"""
    B, D, N1 = pc_pred.shape
    N2 = pc_target.shape[-1]
    pc_pred_e = pc_pred.unsqueeze(-1).expand(B, D, N1, N2)
    pc_target_e = pc_target.unsqueeze(-2).expand(B, D, N1, N2)

    pc_pred_diff = pc_pred_e - pc_pred_e.transpose(-2, -1)
    pc_pred_target_diff = pc_pred_e - pc_target_e

    dists_pp = pc_pred_diff.norm(dim=1)
    dists_pt = pc_pred_target_diff.norm(dim=1)

    mask_diag = 1 - torch.eye(N1, device=pc_pred.device).unsqueeze(0).expand(B, N1, N1)
    mask_pred = ((dists_pp / dists_pp.max(dim=-1, keepdim=True)[0]) < treshold) * mask_diag
    mask_target = ((dists_pt / dists_pt.max(dim=-1, keepdim=True)[0]) < treshold) * mask_diag

    lp = (pc_pred_diff * mask_pred.unsqueeze(1)).sum(dim=-1) / mask_pred.sum(dim=-2, keepdim=True).clamp(min=1)
    lt = (pc_pred_target_diff * mask_target.unsqueeze(1)).sum(dim=-1) / mask_target.sum(dim=-2, keepdim=True).clamp(
        min=1)
    laplace_error = (lp - lt).norm(dim=1).sum(dim=-1)

    return laplace_error.mean()
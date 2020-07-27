# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Model imports
from Models.cloud_embedder import PointNetFlowFeature, PointNetFlowFeature2c, FlowNet3DFeature, \
    FlowNet3DiscriminatorHCat, FlowNet3DiscriminatorDiff, FlowNet3DFeatureCat, FlowNet3DFeatureCat4l, \
    FlowNet3DFeatureCatFat, FlowNet3DDiscriminatorP
# Other imports
from losses import inverse_triplet, triplet_divergence, triplet_margin, triplet_norm_margin


# ### Functions to work with Point Cloud
def initialize_cloud_embedder(args, params, device):
    """initialize the loss module accordingly to the choice of flow_extractor"""
    # initialize triplet loss module
    if params["loss_module"] == "segdisc":
        loss_module = PointNetFlowFeature2c(use_batch_norm=not args.no_BN,
                                            feat_dim=args.loss_feat_dim)

    elif params["loss_module"] == "segfeat":
        loss_module = PointNetFlowFeature(use_batch_norm=not args.no_BN,
                                          feat_dim=args.loss_feat_dim)

    elif params["loss_module"] == "fn3ddisc":
        loss_module = FlowNet3DDiscriminatorP()

    elif params["loss_module"] == "fn3dfeat":
        loss_module = FlowNet3DFeature(feat_dim=args.loss_feat_dim,
                                       n_points=params["n_points"])

    elif params["loss_module"] == "fn3dfeatcat":
        loss_module = FlowNet3DFeatureCat(feat_dim=args.loss_feat_dim,
                                          n_points=params["n_points"])

    elif params["loss_module"] == "fn3ddiscdiff":
        loss_module = FlowNet3DiscriminatorDiff(feat_dim=args.loss_feat_dim,
                                                n_points=params["n_points"])

    elif params["loss_module"] == "fn3ddischcat":
        loss_module = FlowNet3DiscriminatorHCat(feat_dim=args.loss_feat_dim,
                                                n_points=params["n_points"])

    elif params["loss_module"] == "fn3dfeatcat4l":
        loss_module = FlowNet3DFeatureCat4l(feat_dim=args.loss_feat_dim,
                                            n_points=params["n_points"])

    elif params["loss_module"] == "fn3dfeatcatfat":
        loss_module = FlowNet3DFeatureCatFat(feat_dim=args.loss_feat_dim,
                                             n_points=params["n_points"])

    else:
        raise NotImplementedError

    loss_module = nn.DataParallel(loss_module)
    loss_module = loss_module.to(device)

    if args.load_model:
        model_dict = torch.load(args.load_model)
        if "loss_module" in model_dict.keys():
            loss_module.load_state_dict(model_dict["loss_module"])

    return loss_module


def initialize_optimizers(params, flow_extractor, loss_module=None):
    # initialize optimizers
    if params["opt"] == "sgd":
        opt_flow = optim.SGD(flow_extractor.parameters(), lr=params["lr"], momentum=0.9, weight_decay=1e-4)
        opt_loss_module = optim.SGD(loss_module.parameters(), lr=params["lr_LM"], momentum=0.9, weight_decay=1e-4)
    elif params["opt"] == "adam":
        opt_flow = optim.Adam(flow_extractor.parameters(), lr=params["lr"], betas=(0.0, 0.99), weight_decay=1e-4)
        opt_loss_module = optim.Adam(loss_module.parameters(), lr=params["lr_LM"], betas=(0.0, 0.99), weight_decay=1e-4)

    class DumbScheduler:
        @staticmethod
        def step(self):
            pass

    if params["lr_patience"] > 0:
        lr_update_flow = ReduceLROnPlateau(optimizer=opt_flow, mode='min', factor=params["lr_factor"],
                                           patience=params["lr_patience"], verbose=True, cooldown=0, min_lr=1e-6)
    else:
        lr_update_flow = DumbScheduler()

    if params["lr_patience_LM"] > 0:
        lr_update_loss = ReduceLROnPlateau(optimizer=opt_loss_module, mode='min', factor=params["lr_factor"],
                                           patience=params["lr_patience_LM"], verbose=True, cooldown=0, min_lr=1e-6)
    else:
        lr_update_loss = DumbScheduler()

    return opt_flow, opt_loss_module, lr_update_flow, lr_update_loss


def initialize_flow_extractor_loss(loss_type, params):
    # ### Losses for triplet adversarial training.
    # whole point cloud mapped to one vector
    if loss_type == "triplet_l":
        # the fastest to run, usually gets the best results and is more stable.
        flow_extractor_loss_func = lambda p, t, _: (p - t).norm(dim=1, p=params["norm_FE"]).mean()
    elif loss_type == "triplet_hinge":
        # less stable training and metrics don't score as well as triplet_l
        flow_extractor_loss_func = lambda p, t, _: F.relu((p - t).norm(dim=1, p=params["norm_FE"]).mean() - 1)
    elif loss_type == "triplet_inverse":
        # scores about as well as triplet_l, but takes longer to run
        flow_extractor_loss_func = inverse_triplet(norm=params["norm_FE"])
    elif loss_type == "triplet":
        # Not recomended for theoretical reasons. In practice it performs slightly worse than triplet_l
        flow_extractor_loss_func = nn.TripletMarginLoss(swap=True, p=params["norm_FE"])
    elif loss_type == "js":
        flow_extractor_loss_func = triplet_divergence(device=params["device"], triplet=False)
    elif loss_type == "dist":
        flow_extractor_loss_func = lambda x, y, z: x.mean()
    elif loss_type == "emb_dist":
        flow_extractor_loss_func = lambda p, t, _: (p - t).norm(dim=1, p=params["norm_FE"]).mean()

    return flow_extractor_loss_func


def initialize_cloud_embedder_loss(loss_type, params):
    """"""
    # ### Losses for triplet learning:
    if "triplet" in loss_type:
        loss_module_loss_func = nn.TripletMarginLoss(swap=True, p=params["norm_LM"], margin=params["margin"])
    elif loss_type == "js":
        loss_module_loss_func = triplet_divergence(device=params["device"])
    elif loss_type == "emb_dist":
        loss_module_loss_func = triplet_norm_margin(norm=params["norm_LM"], margin=params["margin"])
    elif loss_type == "dist":
        loss_module_loss_func = triplet_margin(params["margin"])

    return loss_module_loss_func

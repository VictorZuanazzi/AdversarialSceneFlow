# torch imports
import torch
import torch.nn.functional as F

# python miscellaneous
from collections import defaultdict
import pdb

# my imports
from aml_utils import select_inputs_, calculate_eucledian_metrics
from losses import l2_loss, deep_loss_, cycons_func_, chamfer_distance, local_flow_consistency, knn_loss,\
    laplacian_regularization
from initializations import initialize_flow_extractor_loss


def train_step_flow_extractor(args, params, use_flow_signal=False, supervised=False, n_points=None):
    """implements functions to perform ONE train step of the flow extractor"""

    # initialize some recurrent parameters
    n_points = params["n_points"] if n_points is None else n_points
    use_deep_loss = not args.use_shallow_loss
    device = params["device"]

    # parse input for the loss module
    # use_half_clouds = False if args.loss_type in ["triplet_l", "triplet_hinge", "js"] else True
    select_inputs = select_inputs_(use_flow_signal, n_points, half_cloud=True)

    out_acc_dict = defaultdict(lambda: 0.0)
    out_error_dict = defaultdict(lambda: 0.0)

    # initialize loss functions
    if not supervised:
            loss_func_shallow = initialize_flow_extractor_loss(loss_type=args.loss_type, params=params)
            loss_func_deep = initialize_flow_extractor_loss(loss_type=params["loss_type_deep"], params=params)
            deep_loss = deep_loss_(loss_func_deep, use_deep_loss, device, scale_type=args.deep_loss_scale)
    else:
        if args.train_type == 'knn':
            if args.loss_type == 'knn':
                loss_func_supervised = knn_loss
        else:
            loss_func_supervised = l2_loss(dim=1, norm=params["norm_FE"])

    if (not supervised) and (params["cycle_consistency"] is not None):
        cycons_func = cycons_func_(params["cycle_consistency"])
    elif params["cycle_consistency_sup"] is not None:
        cycons_func = cycons_func_(params["cycle_consistency_sup"])

    def train_step_FE_func_uns(flow_extractor, cloud_embedder, c1, c2, flow_t1=None):
        # train the flow extractor
        flow_extractor.train()
        cloud_embedder.eval()

        flow_pred = flow_extractor(c1, c2)
        c2_pred = c1 + flow_pred

        c1_, c_anchor, c_negative, c_positive = select_inputs(c1, c2, c2_pred, flow_t1)

        f_0, hidden_feats_0 = cloud_embedder(c1_, c_anchor)
        f_p, hidden_feats_p = cloud_embedder(c1_, c_positive)
        f_n, hidden_feats_n = cloud_embedder(c1_, c_negative)

        loss_hidden_feats = deep_loss(hidden_feats_0, hidden_feats_p, hidden_feats_n)
        loss_feat = loss_func_shallow(f_0, f_p, f_n)

        loss_FE = loss_feat + loss_hidden_feats
        out_error_dict["loss_feat"], out_error_dict["loss_hidden_feats"] = loss_feat.item(), loss_hidden_feats.item()

        if params["cycle_consistency"] is not None:
            c2_pred = c2_pred.detach()
            if args.cycon_aug:
                c2_pred = torch.cat((c2_pred, c2), dim=-1)
            flow_pred_backwards = flow_extractor(c2_pred, c1)[..., :flow_pred.shape[-1]]
            loss_cycons = cycons_func(flow_pred, flow_pred_backwards)
            loss_FE = loss_cycons * args.cycon_contribution + loss_FE
            out_error_dict["loss_cycons"] = loss_cycons.item()
            c2_pred = c2_pred[..., :flow_pred.shape[-1]]

        if params["local_consistency"]:
            # loss_loccons = local_flow_consistency(pc1=c1, flow_pred=flow_pred)
            loss_loccons = local_flow_consistency(c1, flow_pred)
            loss_FE = loss_loccons * params["local_consistency"] + loss_FE
            out_error_dict["loss_loccons"] = loss_loccons.item()

        if params["chamfer"] > 0:
            chamfer_dist = chamfer_distance(pc_pred=c2_pred, pc_target=c2)
            loss_FE += params["chamfer"] * chamfer_dist
            out_error_dict["chamfer"] = chamfer_dist.item()

        if params["laplace"] > 0:
            laplace_loss = laplacian_regularization(pc_pred=c2_pred, pc_target=c2)
            loss_FE += params["laplace"] * laplace_loss
            out_error_dict["laplace"] = laplace_loss.item()

        if params["static_penalty"] > 0:
            static_penalty = (c2_pred[..., :c1.shape[-1]] - c1).norm(dim=1).clamp(max=1).mean()
            loss_FE += params["static_penalty"] * static_penalty
            out_error_dict["static"] = static_penalty.item()

        if flow_t1 is not None:
            flow_errors_dict, flow_accs_dict = calculate_eucledian_metrics(flow_pred, flow_t1)
            out_error_dict.update(flow_errors_dict)
            out_acc_dict.update(flow_accs_dict)

        out_dict = {"error": out_error_dict, "acc": out_acc_dict}

        return loss_FE, out_dict

    def train_step_FE_func_sup(flow_extractor, c1, c2, flow_t1=None):
        # train the flow extractor
        flow_extractor.train()

        flow_pred = flow_extractor(c1, c2)
        c2_pred = c1 + flow_pred
        loss_FE = 0
        # pdb.set_trace()

        loss_FE += args.sup_scale * loss_func_supervised(flow_pred, flow_t1)

        if params["cycle_consistency_sup"] is not None:
            c2_pred_ = c2_pred.detach()
            flow_pred_backwards = flow_extractor(c2_pred_, c1)
            loss_cycons = cycons_func(flow_pred, flow_pred_backwards)
            loss_FE = loss_cycons * args.cycon_contribution + loss_FE
            out_error_dict["loss_cycons"] = loss_cycons.item()

        if params["local_consistency"]:
            loss_loccons = local_flow_consistency(c1, flow_pred)
            loss_FE = loss_loccons * params["local_consistency"] + loss_FE
            out_error_dict["loss_loccons"] = loss_loccons.item()

        if params["chamfer"] > 0:
            chamfer_dist = chamfer_distance(pc_pred=c2_pred, pc_target=c2)
            loss_FE += params["chamfer"] * chamfer_dist
            out_error_dict["chamfer"] = chamfer_dist.item()

        if params["laplace"] > 0:
            laplace_loss = laplacian_regularization(pc_pred=c2_pred, pc_target=c2)
            loss_FE += params["laplace"] * laplace_loss
            out_error_dict["laplace"] = laplace_loss.item()

        if params["static_penalty"] > 0:
            static_penalty = (c2_pred[..., :c1.shape[-1]] - c1).norm(dim=1).clamp(max=1).mean()
            loss_FE += params["static_penalty"] * static_penalty
            out_error_dict["static"] = static_penalty.item()

        flow_errors_dict, flow_accs_dict = calculate_eucledian_metrics(flow_pred, flow_t1)
        out_error_dict.update(flow_errors_dict)
        out_acc_dict.update(flow_accs_dict)

        out_dict = {"error": out_error_dict, "acc": out_acc_dict}

        return loss_FE, out_dict

    def train_step_FE_func_knn_sup(flow_extractor, c1, c2, flow_t1=None):
        # train the flow extractor using knn loss
        flow_extractor.train()

        flow_pred = flow_extractor(c1, c2)
        c2_pred = c1 + flow_pred
        loss_FE = loss_func_supervised(c1 + flow_pred, c2)

        if params["cycle_consistency_sup"] is not None:
            c2_pred_ = c2_pred.detach()
            flow_pred_backwards = flow_extractor(c2_pred_, c1)
            loss_cycons = cycons_func(flow_pred, flow_pred_backwards)
            loss_FE = loss_cycons * args.cycon_contribution + loss_FE
            out_error_dict["loss_cycons"] = loss_cycons.item()

        if params["local_consistency"]:
            loss_loccons = local_flow_consistency(c1, flow_pred)
            loss_FE = loss_loccons * params["local_consistency"] + loss_FE
            out_error_dict["loss_loccons"] = loss_loccons.item()

        if params["chamfer"] > 0:
            chamfer_dist = chamfer_distance(pc_pred=c2_pred, pc_target=c2)
            loss_FE += params["chamfer"] * chamfer_dist
            out_error_dict["chamfer"] = chamfer_dist.item()

        if params["laplace"] > 0:
            laplace_loss = laplacian_regularization(pc_pred=c2_pred, pc_target=c2)
            loss_FE += params["laplace"] * laplace_loss
            out_error_dict["laplace"] = laplace_loss.item()

        if params["static_penalty"] > 0:
            static_penalty = (c2_pred[..., :c1.shape[-1]] - c1).norm(dim=1).clamp(max=1).mean()
            loss_FE += params["static_penalty"] * static_penalty
            out_error_dict["static"] = static_penalty.item()

        flow_errors_dict, flow_accs_dict = calculate_eucledian_metrics(flow_pred, flow_t1)
        out_error_dict.update(flow_errors_dict)
        out_acc_dict.update(flow_accs_dict)

        out_dict = {"error": out_error_dict, "acc": out_acc_dict}

        return loss_FE, out_dict

    if supervised:
        if args.train_type == "knn":
            return train_step_FE_func_knn_sup

        return train_step_FE_func_sup

    else:
        return train_step_FE_func_uns

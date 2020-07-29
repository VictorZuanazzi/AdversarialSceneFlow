from collections import defaultdict

# my imports
from initializations import initialize_cloud_embedder_loss
from losses import deep_loss_
from aml_utils import select_inputs_
import pdb


def train_step_cloud_embedder(args, params, use_flow_signal=False, n_points=None):
    """implements the training step of the Loss Module"""

    # initialize some recurrent parameters
    n_points = params["n_points"] if n_points is None else n_points
    n_sweeps = params["n_sweeps"]
    norm = params["norm_LM"]
    device = params["device"]
    use_deep_loss = not args.use_shallow_loss
    out_error_dict = defaultdict(lambda: 0)
    out_acc_dict = defaultdict(lambda: 0)

    # initialize loss functions
    loss_func_shallow = initialize_cloud_embedder_loss(loss_type=args.loss_type, params=params)
    loss_func_deep = initialize_cloud_embedder_loss(loss_type=params["loss_type_deep"], params=params)
    deep_loss = deep_loss_(loss_func_deep, use_deep_loss, device, scale_type=args.deep_loss_scale)

    # initializes how to parse the input of the loss module
    select_inputs = select_inputs_(use_flow_signal, n_points)

    def train_step_func(flow_extractor, cloud_embedder, c1, c2, flow_t1=None):
        flow_extractor.eval()
        cloud_embedder.train()

        flow_pred = flow_extractor(c1, c2)
        c2_pred = c1 + flow_pred

        c1_, c_anchor, c_positive, c_negative = select_inputs(c1, c2, c2_pred, flow_t1)

        f_0, hidden_feats_0 = cloud_embedder(c1_, c_anchor)
        f_p, hidden_feats_p = cloud_embedder(c1_, c_positive)
        f_n, hidden_feats_n = cloud_embedder(c1_, c_negative)

        loss_hidden_feats = deep_loss(hidden_feats_0, hidden_feats_p, hidden_feats_n)

        if len(f_0.shape) > 2:
            N = f_0.shape[1]
            f_0 = f_0.transpose(1, -1).reshape(-1, N)
            f_p = f_p.transpose(1, -1).reshape(-1, N)
            f_n = f_n.transpose(1, -1).reshape(-1, N)
        loss_feat = loss_func_shallow(f_0, f_p, f_n)

        loss_LM = loss_feat + loss_hidden_feats
        out_error_dict["loss_feat"], out_error_dict["loss_hidden_feats"] = loss_feat.item(), loss_hidden_feats.item()

        # calculate the acc of the loss module
        out_acc_dict["loss_acc"] = (
                (f_0 - f_p).norm(dim=-1, p=norm) < (f_0 - f_n).norm(dim=-1, p=norm)).float().mean().item()

        out_dict = {"error": out_error_dict, "acc": out_acc_dict}

        return loss_LM, out_dict

    return train_step_func


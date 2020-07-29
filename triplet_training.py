from __future__ import print_function

import sys

sys.path.insert(0, '.')
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.nn as nn

# my imports
from aml_utils import calculate_eucledian_metrics, plot_3d_clouds, plot_2d_clouds, print_and_log, save_model, update_running_mean
from initializations import initialize_cloud_embedder, initialize_optimizers, initialize_cloud_embedder_loss
from train_flow_extractor import train_step_flow_extractor
from train_cloud_embedder import train_step_cloud_embedder


def eval_triplet(args, params, flow_extractor, cloud_embedder, dataloader_eval, writer, epoch):
    """Performs evaluation of the models.
    Input:
        args: args from main;
        params: dict, with training parameters;
        flow_extractor: nn.Module, flow model;
        cloud_embedder: nn.Module or None, the cloud_embedder, if None the evalution is only done on the flow_extractor;
        data_loader_eval: pytorch dataloader, with the evaluation data;
        writer: tensorboard writer;
        epoch: int, the current epoch;
    Output:
        writes evaluation statistics to tensorboard writer;
    Return:
        dictionary with statistics from training."""

    n_points = params["n_points_eval"]
    device = params["device"]

    # initialize the cloud embedder losses
    if cloud_embedder is not None:
        loss_func_loss_module_shallow = initialize_cloud_embedder_loss(args.loss_type, params=params)
        loss_func_loss_module_deep = initialize_cloud_embedder_loss(loss_type=params["loss_type_deep"], params=params)

    # odd and even indeces
    i_odds = torch.arange(start=1, end=n_points, step=2)
    i_evens = torch.arange(start=0, end=n_points, step=2)

    # initialize stats
    stats_FE = defaultdict(lambda: 0)
    acc_FE = defaultdict(lambda: 0)
    stats_LM = defaultdict(lambda: 0)
    acc_LM = defaultdict(lambda: 0)

    counter = 0
    with torch.no_grad():
        for i, (clouds, extra_dict) in enumerate(tqdm(dataloader_eval, desc=f'{epoch}) Eval')):
            clouds = clouds.float().to(device)

            c1 = clouds[:, :, :n_points].contiguous()
            c2 = clouds[:, :, n_points: 2 * n_points].contiguous()

            flow_target = extra_dict["flow"].float().to(device)
            flow_t1 = flow_target[:, :, :n_points]
            flow_extractor.eval()

            flow_pred = flow_extractor(c1, c2)
            i_out = flow_pred.shape[2]
            clouds_pred = clouds[:, :, :i_out] + flow_pred
            c2_pred = clouds_pred[:, :, :n_points]
            flow_p1 = flow_pred[:, :, :n_points]

            counter += 1

            # keep stats for flow extractor
            flow_errors_dict, flow_accs_dict = calculate_eucledian_metrics(flow_pred[:, :, :n_points],
                                                                           flow_target[:, :, :n_points])
            stats_FE = update_running_mean(stats_FE, flow_errors_dict, counter=counter)
            acc_FE = update_running_mean(acc_FE, flow_accs_dict, counter=counter)

            if args.save_all_scenes:
                # pdb.set_trace()
                save_dict = {"c1": c1.cpu().numpy(),
                             "c2": c2.cpu().numpy(),
                             "flow": flow_t1.cpu().numpy(),
                             "c2_pred": c2_pred.detach().cpu().numpy(),
                             "flow_pred": flow_pred.detach().cpu().numpy()}
                torch.save(save_dict, args.exp_name + f"/scenes_{i}.pt")

            if cloud_embedder is not None:
                # evaluates the cloud_embedder if it was given
                cloud_embedder.eval()

                # stats of the loss module
                fs_0, hiddens_feats_0 = cloud_embedder(c1.contiguous(), c2.contiguous())
                fs_p, hiddens_feats_p = cloud_embedder(c1.contiguous(), c1 + flow_t1)
                fs_n, hiddens_feats_n = cloud_embedder(c1.contiguous(), c2_pred)

                loss_hidden_feats = 0
                B = fs_0.shape[0]
                for hf0, hfp, hfn in zip(hiddens_feats_0, hiddens_feats_p, hiddens_feats_n):
                    loss_hidden_feats += loss_func_loss_module_deep(hf0.view(B, -1), hfp.view(B, -1), hfn.view(B, -1))

                if len(fs_0.shape) > 2:
                    N = fs_0.shape[1]
                    fs_0 = fs_0.transpose(1, -1).reshape(-1, N)
                    fs_p = fs_p.transpose(1, -1).reshape(-1, N)
                    fs_n = fs_n.transpose(1, -1).reshape(-1, N)

                loss_feat = loss_func_loss_module_shallow(fs_0, fs_p, fs_n)
                loss_module_supervised = loss_feat + loss_hidden_feats

                stats_LM["sup_feat"] += (loss_feat.item()
                                         - stats_LM["sup_feat"]) / counter
                stats_LM["sup_hidden"] += (loss_hidden_feats.item()
                                           - stats_LM["sup_hidden"]) / counter

                loss_module_supervised_acc = (
                        (fs_0 - fs_p).norm(dim=-1) < (fs_0 - fs_n).norm(dim=-1)).float().mean().item()

                fu_0, hiddenu_feats_0 = cloud_embedder(c1[:, :, i_odds], c2[:, :, i_odds])
                fu_p, hiddenu_feats_p = cloud_embedder(c1[:, :, i_evens], c2[:, :, i_evens])
                fu_n, hiddenu_feats_n = cloud_embedder(c1[:, :, i_evens], c2_pred[:, :, i_evens])

                loss_hidden_feats = 0
                B = fu_0.shape[0]
                for hf0, hfp, hfn in zip(hiddenu_feats_0, hiddenu_feats_p, hiddenu_feats_n):
                    loss_hidden_feats += loss_func_loss_module_deep(hf0.view(B, -1), hfp.view(B, -1), hfn.view(B, -1))

                if len(fu_0.shape) > 2:
                    N = fu_0.shape[1]
                    fu_0 = fu_0.transpose(1, -1).reshape(-1, N)
                    fu_p = fu_p.transpose(1, -1).reshape(-1, N)
                    fu_n = fu_n.transpose(1, -1).reshape(-1, N)

                loss_feat = loss_func_loss_module_shallow(fu_0, fu_p, fu_n)
                loss_module_unsupervised = loss_feat + loss_hidden_feats

                stats_LM["uns_feat"] += (loss_feat.item()
                                         - stats_LM["uns_feat"]) / counter
                stats_LM["uns_hidden"] += (loss_hidden_feats.item()
                                           - stats_LM["uns_hidden"]) / counter
                loss_module_unsupervised_acc = (
                        (fu_0 - fu_p).norm(dim=-1) < (fu_0 - fu_n).norm(dim=-1)).float().mean().item()

                stats_LM["supervised"] += (loss_module_supervised.item()
                                           - stats_LM["supervised"]) / counter

                stats_LM["unsupervised"] += (loss_module_unsupervised.item()
                                             - stats_LM["unsupervised"]) / counter

                acc_LM["supervised_acc"] += (loss_module_supervised_acc
                                             - acc_LM["supervised_acc"]) / counter

                acc_LM["unsupervised_acc"] += (loss_module_unsupervised_acc
                                               - acc_LM["unsupervised_acc"]) / counter

            if args.overfit:
                break

        writer.add_scalars(main_tag="FE_eval/stats", tag_scalar_dict=stats_FE, global_step=epoch)
        writer.add_scalars(main_tag="FE_eval/acc", tag_scalar_dict=acc_FE, global_step=epoch)
        print("flow stats eval: ", stats_FE)
        print("acc_FE eval:", acc_FE)

        if cloud_embedder is not None:
            writer.add_scalars(main_tag="LM_eval/stats", tag_scalar_dict=stats_LM, global_step=epoch)
            writer.add_scalars(main_tag="LM_eval/acc", tag_scalar_dict=acc_LM, global_step=epoch)
            print("stats_LM eval:", stats_LM)
            print("acc_LM eval:", acc_LM)

            embeddings = torch.cat((fs_0, fs_p, fs_n, fu_0, fu_p, fu_n), dim=0)
            labels = [f"C2_{i}" for i in range(fs_0.shape[0])] \
                     + [f"C1+f_{i}" for i in range(fs_p.shape[0])] \
                     + [f"C2spred_{i}" for i in range(fs_n.shape[0])] \
                     + [f"C2e_{i}" for i in range(fu_0.shape[0])] \
                     + [f"C2o_{i}" for i in range(fu_p.shape[0])] \
                     + [f"C2upred_{i}" for i in range(fu_n.shape[0])]

            writer.add_embedding(tag="feature", metadata=labels, mat=embeddings, global_step=epoch)

        class ArgsGambi():
            def __init__(self, epoch, n_sweeps, n_points):
                self.n_sweeps = n_sweeps
                self.n_points = n_points
                self.global_step = epoch
                self.save_dir = args.exp_name + "/"

        args_ = ArgsGambi(epoch=epoch, n_sweeps=params["n_sweeps"], n_points=n_points)
        plot_2d = plot_2d_clouds(dataloader_eval.dataset, args_, clouds=clouds, flow_target=flow_target,
                                 tokens=None, cloud_pred=c2_pred, flow_pred=flow_p1, dot_size=0.5)

        writer.add_figure("2D clouds", plot_2d, global_step=epoch)

        plot_3d = plot_3d_clouds(dataloader_eval.dataset, args_, clouds=clouds, flow_target=flow_target,
                                 tokens=None, cloud_pred=c2_pred, flow_pred=flow_p1, dot_size=0.5)

        writer.add_figure("3D clouds", plot_3d, global_step=epoch)

        flow_norms = flow_t1.norm(dim=2).view(-1)
        flow_pred_norms = flow_p1.norm(dim=2).view(-1)
        writer.add_histogram(tag="flow/target", values=flow_norms, global_step=epoch)
        writer.add_histogram(tag="flow/pred", values=flow_pred_norms, global_step=epoch)

        cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)(flow_t1, flow_p1).view(-1)
        writer.add_histogram(tag="cos_sim", values=cos_sim, global_step=epoch)

    return {**stats_LM, **acc_LM, **stats_FE, **acc_FE}


def train_triplet(args, params, flow_extractor, dataloader_train, dataloader_eval, writer):
    """Triplet training
    Input:
        args: args from main;
        params: dict, with training parameters;
        flow_extractor: nn.Module, flow model;
        data_loader_train: pytorch dataloader, with the training data;
        data_loader_eval: pytorch dataloader, with the evaluation data;
        writer: tensorboard writer;
    Output:
        Writes training statistics to tensorboard writer;
        Saves best performing model to disc;
    """
    epochs = args.epochs
    n_points = params["n_points"]
    use_flow_signal = args.use_flow_signal
    device = params["device"]

    # initialize triplet loss module
    cloud_embedder = initialize_cloud_embedder(args, params, device)

    opt_flow, opt_loss_module, lr_update_flow, lr_update_loss = initialize_optimizers(params,
                                                                                      flow_extractor,
                                                                                      cloud_embedder)

    skip_rate = 1  # 2
    skip_rate_loss_module = 1

    best_eval_epe = np.inf
    finish = False
    epoch = 0
    counter_bad_epochs = 0

    train_LM = train_step_cloud_embedder(args, params, args.use_flow_signal)
    train_FE = train_step_flow_extractor(args, params, use_flow_signal, supervised=False)

    while not finish:
        epoch += 1

        # initialize stats
        stats_FE = defaultdict(lambda: 0)
        stats_LM = defaultdict(lambda: 0)
        acc_LM = defaultdict(lambda: 0)
        acc_FE = defaultdict(lambda: 0)

        counter_flow_extractor = 0
        counter_loss_module = 0
        for i, (clouds, extra_dict) in enumerate(tqdm(dataloader_train, desc=f'{epoch}) Train')):

            clouds = clouds.float().to(device)
            flow_target = extra_dict.get("flow", None)
            if flow_target is not None:
                flow_target = extra_dict["flow"].float().to(device)
                flow_t1 = flow_target[:, :, :n_points].contiguous()
            else:
                flow_t1 = None

            c1 = clouds[:, :, :n_points].contiguous()
            c2 = clouds[:, :, n_points: 2 * n_points].contiguous()

            # Train Cloud Embedder
            if not (i % skip_rate_loss_module):
                counter_loss_module += 1
                opt_loss_module.zero_grad()

                loss_LM, train_dict = train_LM(flow_extractor,
                                               cloud_embedder,
                                               c1, c2,
                                               flow_t1=flow_t1)

                if args.reverse_LM:
                    loss_reverse, _ = train_LM(flow_extractor,
                                               cloud_embedder,
                                               c2, c1)
                    loss_LM += loss_reverse
                    stats_LM["rev"] += (loss_reverse.item() - stats_LM["rev"]) / counter_loss_module

                loss_LM.backward()
                torch.nn.utils.clip_grad_norm_(cloud_embedder.parameters(), max_norm=5.0)
                opt_loss_module.step()

                stats_LM["loss"] += (loss_LM.item() - stats_LM["loss"]) / counter_loss_module
                acc_LM = update_running_mean(acc_LM, train_dict["acc"], counter=counter_loss_module)
                stats_LM = update_running_mean(stats_LM, train_dict["error"], counter=counter_loss_module)

            # gives the loss module some time to adjust to what is coming
            if i % skip_rate:
                continue

            # Train Flow Extractor
            counter_flow_extractor += 1
            opt_flow.zero_grad()

            loss_FE, train_dict = train_FE(flow_extractor, cloud_embedder, c1, c2, flow_t1=flow_t1)

            if args.reverse_FE:
                loss_reverse, _ = train_FE(flow_extractor, cloud_embedder, c2, c1, flow_t1=None)
                loss_FE += loss_reverse
                stats_FE["rev"] += (loss_reverse.item() - stats_FE["rev"]) / counter_flow_extractor

            loss_FE.backward()
            torch.nn.utils.clip_grad_norm_(flow_extractor.parameters(), max_norm=5.0)
            opt_flow.step()

            # keep stats
            stats_FE["loss"] += (loss_FE.item() - stats_FE["loss"]) / counter_flow_extractor
            acc_FE = update_running_mean(acc_FE, train_dict["acc"], counter=counter_flow_extractor)
            stats_FE = update_running_mean(stats_FE, train_dict["error"], counter=counter_flow_extractor)

        # Evaluate model
        stats_eval = eval_triplet(args, params, flow_extractor, cloud_embedder, dataloader_eval, writer, epoch)

        # log training stats to tensorboard
        writer.add_scalars(main_tag="train_LM/stats", tag_scalar_dict=stats_LM, global_step=epoch)
        writer.add_scalars(main_tag="train_LM/acc", tag_scalar_dict=acc_LM, global_step=epoch)
        print("train loss module stats: ", stats_LM)
        print("train loss module acc: ", acc_LM)

        writer.add_scalars(main_tag="train_FE/stats", tag_scalar_dict=stats_FE, global_step=epoch)
        writer.add_scalars(main_tag="train_FE/acc", tag_scalar_dict=acc_FE, global_step=epoch)
        print("train flow stats: ", stats_FE)
        print("train flow acc: ", acc_FE)

        # skip rate of the loss module
        if acc_LM["acc"] > args.skip_loss_module:
            skip_rate_loss_module += int(acc_LM["acc"] * 10)
        else:
            skip_rate_loss_module = max(1, skip_rate_loss_module // 2)

        writer.add_scalars(main_tag="other/skip_loss_module",
                           tag_scalar_dict={"skip_rate": skip_rate_loss_module},
                           global_step=epoch)

        # save best model
        if stats_eval["epe"] < best_eval_epe:
            counter_bad_epochs = 0
            best_eval_epe = stats_eval["epe"]
            dict_wraper = {"flow_extractor": flow_extractor.state_dict(),
                           "cloud_embedder": cloud_embedder.state_dict(),
                           "opt_flow": opt_flow.state_dict(),
                           "opt_loss": opt_loss_module.state_dict(),
                           **stats_eval,
                           **stats_FE,
                           **acc_FE,
                           **stats_LM,
                           **acc_LM}
            save_model(dict_to_be_saved=dict_wraper, file_name=args.exp_name + "/model_best.pt", retry=True)
            print_and_log(message=f"{epoch}) BEST EPE. \n {stats_eval}",
                          verbose=True,
                          add_timestamp=True,
                          global_step=epoch,
                          tensorboard_writer=writer)

        else:
            counter_bad_epochs += 1

        if args.retrieve_old_loss_module and (counter_loss_module > 2 * params["lr_patience"]):
            # retrieve the old loss module
            dict_wraper = torch.load(args.exp_name + "/model_best.pt")
            cloud_embedder.load_state_dict(dict_wraper["cloud_embedder"])

        else:
            # update parameters that depend on the evaluation stats
            # lr scheduler
            lr_update_flow.step(stats_eval["epe"])
            lr_update_loss.step(stats_eval["epe"])

        writer.add_scalars(main_tag="other/lr",
                           tag_scalar_dict={"flow": opt_flow.state_dict()["param_groups"][0]["lr"],
                                            "cloud_embedder": opt_loss_module.state_dict()["param_groups"][0]["lr"]},
                           global_step=epoch)

        if epoch > args.min_epochs:
            if (counter_bad_epochs > 4 * params["lr_patience"]) or (epoch > epochs):
                # end training.
                finish = True

    # End of training stuff
    dict_wraper = torch.load(args.exp_name + "/model_best.pt")
    flow_extractor.load_state_dict(dict_wraper["flow_extractor"])
    cloud_embedder.load_state_dict(dict_wraper["cloud_embedder"])

    stats_eval = eval_triplet(args, params, flow_extractor, cloud_embedder, dataloader_eval, writer, epoch)

    print_and_log(message=f"{epoch}) End of Training. \n {stats_eval}",
                  verbose=True,
                  add_timestamp=True,
                  global_step=epoch,
                  tensorboard_writer=writer)
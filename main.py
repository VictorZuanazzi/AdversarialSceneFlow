"""Implemented by Victor Zuanazzi
Code for Adversarial Metric Learning """

# stanard imports
from __future__ import print_function
import sys

sys.path.insert(0, '.')
import argparse
import numpy as np
from tqdm import tqdm
import pdb
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# my imports
from Sandbox.toy_data import SquaresToyDataset
from Sandbox.singleshapenet import SingleShapeNet
from Sandbox.multishapenet import MultiShapeNet
from Sandbox.flyingthings3d import FlyingThings3D
from Sandbox.kitti import KITTI, KittiNoGround
from Sandbox.lyft import TorchLyftDataset
from aml_utils import calculate_eucledian_metrics, plot_3d_clouds, plot_2d_clouds, parsedtime, log_args, \
    print_and_log, save_model, update_running_mean
from Models.flow_extractor import Segmenter, weights_init, ZeroFlow, AvgFlow, knnFlow, PPWC, FlowNet3D
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
        dictionary with statistics from traioning."""

    n_points = params["n_points_eval"]
    device = params["device"]

    if cloud_embedder is not None:
        loss_func_loss_module_shallow = initialize_cloud_embedder_loss(args.loss_type, params=params)
        loss_func_loss_module_deep = initialize_cloud_embedder_loss(loss_type=params["loss_type_deep"], params=params)

    # a bunch of indexing!
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


def train_supervised(args, params, flow_extractor, dataloader_train, dataloader_eval, writer):
    """Supervised Training"""
    epochs = args.epochs
    n_points = params["n_points"]
    device = params["device"]

    # initialize optimizers
    if params["opt"] == "sgd":
        opt_flow = optim.SGD(flow_extractor.parameters(), lr=params["lr"], momentum=0.9, weight_decay=1e-4)
    elif params["opt"] == "adam":
        opt_flow = optim.Adam(flow_extractor.parameters(), lr=params["lr"], betas=(0.9, 0.999), weight_decay=1e-4)

    lr_update_flow = ReduceLROnPlateau(optimizer=opt_flow, mode='min', factor=params["lr_factor"],
                                       patience=params["lr_patience"], verbose=True, cooldown=0, min_lr=1e-6)

    best_eval_epe = np.inf
    finish = False
    epoch = 0
    counter_bad_epochs = 0

    train_FE = train_step_flow_extractor(args, params, use_flow_signal=False, supervised=True)

    while not finish:
        epoch += 1

        # initialize stats
        stats_FE = defaultdict(lambda: 0)
        acc_FE = defaultdict(lambda: 0)

        counter = 0
        for i, (clouds, extra_dict) in enumerate(tqdm(dataloader_train, desc=f'{epoch}) Train')):
            counter += 1

            clouds = clouds.float().to(device)
            flow_target = extra_dict.get("flow", torch.zeros_like(clouds))
            flow_target = flow_target.float().to(device)

            c1 = clouds[:, :, :n_points].contiguous()
            c2 = clouds[:, :, n_points: 2 * n_points].contiguous()
            flow_t1 = flow_target[:, :, :n_points].contiguous()

            loss_FE, train_dict = train_FE(flow_extractor, c1, c2, flow_t1)

            opt_flow.zero_grad()
            loss_FE.backward()
            torch.nn.utils.clip_grad_norm_(flow_extractor.parameters(), max_norm=5.0)
            opt_flow.step()
            # keep stats
            stats_FE["loss"] += (loss_FE.item() - stats_FE["loss"]) / counter
            acc_FE = update_running_mean(acc_FE, train_dict["acc"], counter=counter)
            stats_FE = update_running_mean(stats_FE, train_dict["error"], counter=counter)

        stats_eval = eval_triplet(args, params, flow_extractor, None, dataloader_eval, writer, epoch)

        # lr scheduler
        if not args.overfit:
            lr_update_flow.step(stats_eval["epe"])
        else:
            lr_update_flow.step(stats_FE["epe"])

        writer.add_scalars(main_tag="lr",
                           tag_scalar_dict={"flow": opt_flow.state_dict()["param_groups"][0]["lr"]},
                           global_step=epoch)

        writer.add_scalars(main_tag="train_FE/stats", tag_scalar_dict=stats_FE, global_step=epoch)
        writer.add_scalars(main_tag="train_FE/acc", tag_scalar_dict=acc_FE, global_step=epoch)
        print("stats_FE train:", stats_FE)
        print("acc_FE train", acc_FE)

        # save best model
        if stats_eval["epe"] < best_eval_epe:
            counter_bad_epochs = 0
            best_eval_epe = stats_eval["epe"]
            dict_wraper = {"flow_extractor": flow_extractor.state_dict(),
                           "opt_flow": opt_flow.state_dict(),
                           **stats_eval,
                           **stats_FE,
                           **acc_FE}
            save_model(dict_to_be_saved=dict_wraper, file_name=args.exp_name + "/model_best.pt", retry=True)
            print_and_log(message=f"{epoch}) BEST EPE. \n {stats_eval}",
                          verbose=True,
                          add_timestamp=True,
                          global_step=epoch,
                          tensorboard_writer=writer)

        else:
            counter_bad_epochs += 1
        if epoch > args.min_epochs:
            if (counter_bad_epochs > 4 * params["lr_patience"]) or (epoch > epochs):
                # end training.
                finish = True

    # End of training stuff
    dict_wraper = torch.load(args.exp_name + "/model_best.pt")
    flow_extractor.load_state_dict(dict_wraper["flow_extractor"])

    stats_eval = eval_triplet(args, params, flow_extractor, None, dataloader_eval, writer, epoch)

    print_and_log(message=f"{epoch}) End of Training. \n {stats_eval}",
                  verbose=True,
                  add_timestamp=True,
                  global_step=epoch,
                  tensorboard_writer=writer)


def train_triplet(args, params, flow_extractor, dataloader_train, dataloader_eval, writer):
    """Triplet training """
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

            # Train Loss Module
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


def train_semi_supervised(args, params, flow_extractor, dataloader_sup, dataloader_uns, dataloader_eval, writer):
    epochs = args.epochs
    n_points = params["n_points"]
    n_points_uns = params["n_points_uns"]
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

    # initialize training functions
    train_LM_sup = train_step_cloud_embedder(args, params, use_flow_signal=True, n_points=n_points)
    train_LM_uns = train_step_cloud_embedder(args, params, use_flow_signal=False, n_points=n_points_uns)

    train_FE_sup = train_step_flow_extractor(args, params, supervised=True, n_points=n_points)
    train_FE_uns = train_step_flow_extractor(args, params, use_flow_signal=False, supervised=False,
                                             n_points=n_points_uns)

    num_iterations = min(len(dataloader_sup), len(dataloader_uns))
    fade_uns_loss = args.fade_uns_loss if args.fade_uns_loss > 0 else 0.1

    while not finish:
        epoch += 1

        # initialize stats
        stats_FE = defaultdict(lambda: 0)
        stats_FE_epe = defaultdict(lambda: 0)
        stats_FE_sec = defaultdict(lambda: 0)
        stats_LM = defaultdict(lambda: 0)
        acc_LM = defaultdict(lambda: 0)
        acc_FE = defaultdict(lambda: 0)
        acc_FE_sec = defaultdict(lambda: 0)

        counter_flow_extractor = 0
        counter_loss_module = 0

        generator_sup = iter(dataloader_sup)
        generator_uns = iter(dataloader_uns)
        fade_factor = min(epoch / fade_uns_loss, args.max_fade_uns_loss)

        for i in tqdm(range(num_iterations), desc=f'{epoch}) Train'):

            # input for supervised training
            clouds, extra_dict = next(generator_sup)
            clouds = clouds.float()
            flow_target = extra_dict["flow"].float()

            c1_sup = clouds[:, :, :n_points].contiguous()
            c2_sup = clouds[:, :, n_points: 2 * n_points].contiguous()
            flow_sup = flow_target[:, :, :n_points].contiguous()

            # input for unsupervised training
            clouds, extra_dict = next(generator_uns)
            clouds = clouds.float()
            flow_target = extra_dict.get("flow", None)
            c1_uns = clouds[:, :, :n_points_uns].contiguous()
            c2_uns = clouds[:, :, n_points_uns: 2 * n_points_uns].contiguous()
            flow_uns = flow_target[:, :, :n_points_uns].float().contiguous() if flow_target is not None else None

            if args.normalize_input:
                r_sup = c1_sup.norm(dim=1, keepdim=True).max(dim=-1, keepdim=True)[0]
                r_uns = c1_uns.norm(dim=1, keepdim=True).max(dim=-1, keepdim=True)[0]
                rate = r_uns / r_sup
                c1_sup, c2_sup, flow_sup = rate * c1_sup, rate * c2_sup, rate * flow_sup
            else:
                rate = 1.0

            # train loss module
            if not (i % skip_rate_loss_module):
                counter_loss_module += 1

                # supervised trainnig
                loss_sup, train_dict_sup = train_LM_sup(flow_extractor, cloud_embedder,
                                                        c1_sup.to(device), c2_sup.to(device),
                                                        flow_t1=flow_sup.to(device))

                # unsupervised training
                loss_uns, train_dict_uns = train_LM_uns(flow_extractor, cloud_embedder,
                                                        c1_uns.to(device), c2_uns.to(device), flow_t1=flow_uns)

                loss_LM = loss_sup + loss_uns

                # one step for both supervised and unsupervised training.
                opt_loss_module.zero_grad()
                loss_LM.backward()
                torch.nn.utils.clip_grad_norm_(cloud_embedder.parameters(), max_norm=5.0)
                opt_loss_module.step()

                # keep stats
                stats_LM["loss"] += (loss_LM.item() - stats_LM["loss"]) / counter_loss_module
                stats_LM["loss_sup"] += (loss_sup.item() - stats_LM["loss_sup"]) / counter_loss_module
                stats_LM["loss_uns"] += (loss_uns.item() - stats_LM["loss_uns"]) / counter_loss_module

                stats_LM = update_running_mean(stats_LM, train_dict_sup["error"], counter_loss_module,
                                               pos_string="_sup")
                stats_LM = update_running_mean(stats_LM, train_dict_uns["error"], counter_loss_module,
                                               pos_string="_uns")
                acc_LM = update_running_mean(acc_LM, train_dict_sup["acc"], counter_loss_module, pos_string="_sup")
                acc_LM = update_running_mean(acc_LM, train_dict_uns["acc"], counter_loss_module, pos_string="_uns")

                common_errors = {}
                for metric in train_dict_sup["error"].keys():
                    if metric in list(train_dict_sup["error"].keys()):
                        common_errors[metric + "_all"] = train_dict_uns["error"][metric] + train_dict_sup["error"][
                            metric]
                stats_LM = update_running_mean(stats_LM, common_errors, counter_loss_module)

                common_accs = {}
                for metric in train_dict_sup["acc"].keys():
                    if metric in list(train_dict_sup["acc"].keys()):
                        common_accs[metric + "_all"] = train_dict_uns["acc"][metric] + train_dict_sup["acc"][metric]

                acc_LM = update_running_mean(acc_LM, common_accs, counter_loss_module)

            # train flow extractor
            # gives the loss module some time to adjust to what is coming
            if i % skip_rate:
                continue

            # Trian Flow Extractor
            counter_flow_extractor += 1

            # train the flow extractor
            loss_sup, train_dict_sup = train_FE_sup(flow_extractor,
                                                    c1_sup.to(device), c2_sup.to(device), flow_t1=flow_sup.to(device))

            loss_uns, train_dict_uns = train_FE_uns(flow_extractor, cloud_embedder,
                                                    c1_uns.to(device), c2_uns.to(device),
                                                    flow_t1=flow_uns.to(device) if flow_uns is not None else None)

            loss_FE = loss_sup + (loss_uns * fade_factor)

            opt_flow.zero_grad()
            loss_FE.backward()
            torch.nn.utils.clip_grad_norm_(flow_extractor.parameters(), max_norm=5.0)
            opt_flow.step()

            # keep stats
            # main loss metrics are kept in stats_FE
            stats_FE["loss"] += (loss_FE.item() - stats_FE["loss"]) / counter_flow_extractor
            stats_FE["loss_sup"] += (loss_sup.item() - stats_FE["loss_sup"]) / counter_flow_extractor
            stats_FE["loss_uns"] += (loss_uns.item() - stats_FE["loss_uns"]) / counter_flow_extractor

            acc_FE_sec = update_running_mean(acc_FE_sec, train_dict_sup["acc"], counter_flow_extractor,
                                             pos_string="_sup")
            acc_FE_sec = update_running_mean(acc_FE_sec, train_dict_uns["acc"], counter_flow_extractor,
                                             pos_string="_uns")

            common_accs = {}
            for metric in train_dict_sup["acc"]:
                if metric in train_dict_uns["acc"]:
                    common_accs[metric + "_all"] = (train_dict_sup["acc"][metric] + train_dict_uns["acc"][metric]) / 2
            acc_FE = update_running_mean(acc_FE, common_accs, counter_flow_extractor, pos_string="_mean")

            metrics_sup = list(train_dict_sup["error"].keys())
            metrics_uns = list(train_dict_uns["error"].keys())
            metrics_all = set(metrics_sup + metrics_uns)

            # ugly, but necessary
            for metric in metrics_all:

                if (metric in metrics_sup) and (metric in metrics_uns):
                    m_sup = metric + "_sup"
                    m_uns = metric + "_uns"
                    sum_sup_uns = train_dict_sup["error"][metric] + train_dict_uns["error"][metric]

                    if "epe" in metric:
                        sum_sup_uns /= 2
                        stats_FE_epe[m_sup] += (train_dict_sup["error"][metric] - stats_FE_epe[
                            m_sup]) / counter_flow_extractor
                        stats_FE_epe[m_uns] += (train_dict_uns["error"][metric] - stats_FE_epe[
                            m_uns]) / counter_flow_extractor
                        stats_FE_epe[metric + "_mean"] += (sum_sup_uns - stats_FE_epe[
                            metric + "_mean"]) / counter_flow_extractor

                    else:
                        stats_FE_sec[m_sup] += (train_dict_sup["error"][metric] - stats_FE_sec[
                            m_sup]) / counter_flow_extractor
                        stats_FE_sec[m_uns] += (train_dict_uns["error"][metric] - stats_FE_sec[
                            m_uns]) / counter_flow_extractor
                        stats_FE_sec[metric + "_all"] += (sum_sup_uns - stats_FE_sec[
                            metric + "_all"]) / counter_flow_extractor
                else:
                    m_sup = metric if metric in metrics_sup else None
                    m_uns = metric if metric in metrics_uns else None

                    if "epe" in metric:
                        if m_sup is not None:
                            stats_FE_epe[m_sup] += (train_dict_sup["error"][metric] - stats_FE_epe[
                                m_sup]) / counter_flow_extractor
                        if m_uns is not None:
                            stats_FE_epe[m_uns] += (train_dict_uns["error"][metric] - stats_FE_epe[
                                m_uns]) / counter_flow_extractor
                    else:
                        if m_sup is not None:
                            stats_FE_sec[m_sup] += (train_dict_sup["error"][metric] - stats_FE_sec[
                                m_sup]) / counter_flow_extractor
                        if m_uns is not None:
                            stats_FE_sec[m_uns] += (train_dict_uns["error"][metric] - stats_FE_sec[
                                m_uns]) / counter_flow_extractor

        # Evaluate model
        stats_eval = eval_triplet(args, params, flow_extractor, cloud_embedder, dataloader_eval, writer, epoch)

        # log training stats to tensorboard
        writer.add_scalars(main_tag="LM_train/stats", tag_scalar_dict=stats_LM, global_step=epoch)
        writer.add_scalars(main_tag="LM_train/acc", tag_scalar_dict=acc_LM, global_step=epoch)
        print("train loss module stats: ", stats_LM)
        print("train loss module acc: ", acc_LM)

        writer.add_scalars(main_tag="FE_train_stats/stats", tag_scalar_dict=stats_FE, global_step=epoch)
        writer.add_scalars(main_tag="FE_train_stats/epe", tag_scalar_dict=stats_FE_epe, global_step=epoch)
        writer.add_scalars(main_tag="FE_train_stats/sec", tag_scalar_dict=stats_FE_sec, global_step=epoch)
        writer.add_scalars(main_tag="FE_train_acc/acc", tag_scalar_dict=acc_FE, global_step=epoch)
        writer.add_scalars(main_tag="FE_train_acc/acc_sec", tag_scalar_dict=acc_FE_sec, global_step=epoch)
        print("train flow stats: ", stats_FE)
        print("train flow acc: ", acc_FE)

        # skip rate of the loss module
        if acc_LM["all"] > args.skip_loss_module:
            skip_rate_loss_module += int(acc_LM["all"] * 10)
        else:
            skip_rate_loss_module = max(1, skip_rate_loss_module // 2)

        writer.add_scalars(main_tag="other/skip_loss_module",
                           tag_scalar_dict={"skip_rate": skip_rate_loss_module},
                           global_step=epoch)

        writer.add_scalars(main_tag="other/fade_factor",
                           tag_scalar_dict={"fade_factor": fade_factor},
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


def main(args):
    random_seed = args.seed
    input_dims = 3
    flow_dim = 3
    test_step = args.test_step

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(device)
    print("There are ", torch.cuda.device_count(), " GPUs available.")

    # initialize default params
    params = {"device": device,
              "lr_factor": {"supervised": 0.5}.get(args.train_type, 0.75) if args.lr_factor < 0 else args.lr_factor,
              # decreases the lr by factor when stuck
              "lr_patience": {"supervised": 5}.get(args.train_type, 10) if args.lr_patience < 0 else args.lr_patience,
              "lr_patience_LM": {"supervised": 5
                                 }.get(args.train_type, 10) if args.lr_patience_LM < 0 else args.lr_patience_LM,
              "opt": {"supervised": "adam",
                      "semi_supervised": "adam"}.get(args.train_type, "adam") if not args.opt else args.opt,
              "n_sweeps": 2 if args.train_type == "lidarbugsup" else 2,
              "width": 32 if args.width < 0 else args.width,
              "height": 32 if args.height < 0 else args.height,
              "norm_FE": args.norm_FE if args.norm_FE > 0 else np.inf,
              "norm_LM": args.norm_LM if args.norm_LM > 0 else np.inf,
              "dataset_uns": args.dataset if not args.dataset_uns else args.dataset_uns,
              "dataset_eval": args.dataset if not args.dataset_eval else args.dataset_eval,
              "cycle_consistency": {
                  "supervised": args.cycle_consistency_sup if args.cycle_consistency_sup == '' else None,
                  "triplet": "cos_l2" if args.cycle_consistency == '' else None,
                  "semi_supervised": "cos_l2" if args.cycle_consistency == '' else None,
              }.get(args.train_type, "cos_l2") if args.cycle_consistency in (
                  "", "none") else args.cycle_consistency,
              "cycle_consistency_sup": None if args.cycle_consistency_sup in (
                  "", "none") else args.cycle_consistency_sup,
              "local_consistency": args.local_consistency,
              "loss_type_deep": {"triplet_hinge": "triplet_hinge",
                                 "triplet_inverse": "triplet_inverse",
                                 "triplet_margin": "triplet_margin",
                                 "triplet": "triplet",
                                 "js": "js",
                                 "dist": "dist",
                                 "emb_dist": "emb_dist"}.get(args.loss_type,
                                                             "triplet_l") if args.loss_type_deep == "" else args.loss_type_deep,
              "margin": args.margin,
              "chamfer": args.chamfer,
              "laplace": args.laplace,
              "static_penalty": args.static_penalty,
              }

    # initialize training dataloaders
    # in case of semi-supervised training, it initializes the supervised dataloader
    if args.dataset == "squares":
        # default parameters for training with this Sandbox
        params["batch_size"] = 32 if args.batch_size <= 0 else args.batch_size
        params["n_points"] = 64 if args.n_points <= 0 else args.n_points
        params["flow_extractor"] = "segmenter" if not args.flow_extractor else args.flow_extractor
        params["lr"] = {"supervised": 0.001}.get(args.train_type, 0.005) if args.lr < 0 else args.lr
        params["lr_LM"] = params["lr"] / 10 if args.lr_LM < 0 else args.lr_LM

        dataset_train = SquaresToyDataset(n_sweeps=params["n_sweeps"], n_points=params["n_points"],
                                          input_dims=input_dims, flow_dim=flow_dim, random_transition=True)

    elif args.dataset == "fly":

        # default parameters for training with this Sandbox
        params["batch_size"] = {"semi_supervised": 16}.get(args.train_type,
                                                           32) if args.batch_size <= 0 else args.batch_size
        params["n_points"] = 2048 if args.n_points <= 0 else args.n_points

        params["lr"] = {"supervised": 0.0005,
                        "semi_supervised": 0.005}.get(args.train_type, 0.005) if args.lr < 0 else args.lr
        params["lr_LM"] = params["lr"] / 10 if args.lr_LM < 0 else args.lr_LM

        params["flow_extractor"] = "flownet3d" if not args.flow_extractor else args.flow_extractor
        dataset_train = FlyingThings3D(n_sweeps=params["n_sweeps"], n_points=params["n_points"],
                                       input_dims=input_dims,
                                       verbose=True, partition="train", load_data_in_memory=True)

    elif args.dataset == "shapenet":
        # default parameters for training with this Sandbox
        params["batch_size"] = {"segmenter": 128,
                                "flownet3d": 64 if "semi" not in args.train_type else 32
                                }.get(args.flow_extractor, 32) if args.batch_size <= 0 else args.batch_size
        params["n_points"] = 512 if args.n_points <= 0 else args.n_points
        params["flow_extractor"] = "flownet3d" if not args.flow_extractor else args.flow_extractor
        params["lr"] = {"supervised": 0.001,
                        "semi_supervised": 0.0005}.get(args.train_type, 0.005) if args.lr < 0 else args.lr
        params["lr_LM"] = params["lr"] / 10 if args.lr_LM < 0 else args.lr_LM

        dataset_train = SingleShapeNet(n_sweeps=params["n_sweeps"], n_points=params["n_points"],
                                       rotate=not args.freeze_rotation,
                                       overfit=args.overfit, data_size=None if args.data_size < 0 else args.data_size)

    elif args.dataset == "mult_shape":
        # default parameters for training with this Sandbox
        params["batch_size"] = {"segmenter": 128,
                                "flownet3d": 64 if "semi" not in args.train_type else 32
                                }.get(args.flow_extractor, 32) if args.batch_size <= 0 else args.batch_size
        params["n_points"] = 512 if args.n_points <= 0 else args.n_points
        params["flow_extractor"] = "flownet3d" if not args.flow_extractor else args.flow_extractor
        params["lr"] = {"supervised": 0.001,
                        "semi_supervised": 0.0005}.get(args.train_type, 0.005) if args.lr < 0 else args.lr
        params["lr_LM"] = params["lr"] / 10 if args.lr_LM < 0 else args.lr_LM

        dataset_train = MultiShapeNet(n_sweeps=params["n_sweeps"], n_points=params["n_points"],
                                      rotate=not args.freeze_rotation,
                                      overfit=args.overfit,
                                      data_size=None if args.data_size < 0 else args.data_size,
                                      occlusion=args.occlusion)

    elif args.dataset == "kitti":
        # default parameters for training with this Sandbox
        params["batch_size"] = {"semi_supervised": 16}.get(args.train_type,
                                                           32) if args.batch_size <= 0 else args.batch_size
        params["n_points"] = 2048 if args.n_points <= 0 else args.n_points
        params["flow_extractor"] = "flownet3d" if not args.flow_extractor else args.flow_extractor
        params["lr"] = {"supervised": 0.0005,
                        "semi_supervised": 0.005}.get(args.train_type, 0.005) if args.lr < 0 else args.lr
        params["lr_LM"] = params["lr"] / 10 if args.lr_LM < 0 else args.lr_LM

        dataset_train = KITTI(n_points=params["n_points"], input_dims=input_dims, remove_ground=args.remove_ground,
                              verbose=True, load_data_in_memory=True, partition="train")

    elif args.dataset == "kitting":
        # default parameters for training with this Sandbox
        params["batch_size"] = {"semi_supervised": 8}.get(args.train_type,
                                                          16) if args.batch_size <= 0 else args.batch_size
        params["n_points"] = 16384 if args.n_points <= 0 else args.n_points
        params["flow_extractor"] = "flownet3d" if not args.flow_extractor else args.flow_extractor
        params["lr"] = {"supervised": 0.0005,
                        "semi_supervised": 0.005}.get(args.train_type, 0.005) if args.lr < 0 else args.lr
        params["lr_LM"] = params["lr"] / 10 if args.lr_LM < 0 else args.lr_LM

        dataset_train = KittiNoGround(n_points=params["n_points"], input_dims=input_dims,
                                      remove_ground=args.remove_ground,
                                      verbose=True, load_data_in_memory=True, partition="train")

    elif args.dataset == "lyft":
        params["batch_size"] = {"semi_supervised": 8}.get(args.train_type,
                                                          16) if args.batch_size <= 0 else args.batch_size
        params["n_points"] = 16384 if args.n_points <= 0 else args.n_points
        params["flow_extractor"] = "flownet3d" if not args.flow_extractor else args.flow_extractor
        params["lr"] = {"supervised": 0.0005,
                        "semi_supervised": 0.005}.get(args.train_type, 0.005) if args.lr < 0 else args.lr
        params["lr_LM"] = params["lr"] / 10 if args.lr_LM < 0 else args.lr_LM

        dataset_train = TorchLyftDataset(n_points=params["n_points"], input_dims=input_dims,
                                         remove_ground=args.remove_ground,
                                         verbose=True, load_data_in_memory=True, partition="train")

    else:
        raise Exception(f'Dataset {args.dataset} Not implemented')

    # initialize evaluation dataloaders
    if params["dataset_eval"] == "squares":
        params["n_points_eval"] = 64 if args.n_points <= 0 else args.n_points
        dataset_eval = SquaresToyDataset(n_sweeps=params["n_sweeps"], n_points=params["n_points_eval"],
                                         input_dims=input_dims, flow_dim=flow_dim, random_transition=True)

    elif params["dataset_eval"] == "fly":
        params["n_points_eval"] = 2048 if args.n_points <= 0 else args.n_points
        dataset_eval = FlyingThings3D(n_sweeps=params["n_sweeps"], n_points=params["n_points_eval"],
                                      input_dims=input_dims,
                                      verbose=True, partition="eval", load_data_in_memory=True)

    elif params["dataset_eval"] == "shapenet":
        params["n_points_eval"] = 512 if args.n_points <= 0 else args.n_points
        dataset_eval = SingleShapeNet(n_sweeps=params["n_sweeps"], n_points=params["n_points_eval"],
                                      partition="val", rotate=not args.freeze_rotation,
                                      overfit=args.overfit, data_size=None if args.data_size < 0 else args.data_size)

    elif params["dataset_eval"] == "mult_shape":
        params["n_points_eval"] = 512 if args.n_points <= 0 else args.n_points
        dataset_eval = MultiShapeNet(n_sweeps=params["n_sweeps"], n_points=params["n_points_eval"],
                                     partition="val", rotate=not args.freeze_rotation,
                                     overfit=args.overfit,
                                     data_size=None if args.data_size < 0 else args.data_size,
                                     occlusion=args.occlusion)

    elif params["dataset_eval"] == "kitti":
        params["n_points_eval"] = 2048 if args.n_points <= 0 else args.n_points
        dataset_eval = KITTI(n_points=params["n_points"], input_dims=input_dims, remove_ground=args.remove_ground,
                             verbose=True, load_data_in_memory=True, partition="eval")

    elif params["dataset_eval"] == "kitting":
        params["n_points_eval"] = 16384 if args.n_points <= 0 else args.n_points
        dataset_eval = KittiNoGround(n_points=params["n_points"], input_dims=input_dims,
                                     remove_ground=args.remove_ground,
                                     verbose=True, load_data_in_memory=True, partition="eval")

    else:
        raise Exception(f'Dataset {params["dataset_eval"]} Not implemented')

    # create dataloaders
    torch.multiprocessing.set_sharing_strategy('file_system')  # this fix multi-thread issues
    dataloader_train = DataLoader(dataset_train, batch_size=params["batch_size"], shuffle=True, drop_last=True,
                                  num_workers=args.num_workers)
    dataloader_eval = DataLoader(dataset_eval, batch_size=params["batch_size"], shuffle=False, drop_last=True,
                                 num_workers=args.num_workers)

    if args.train_type == "semi_supervised":
        # initialize unsupervised dataloaders
        if params["dataset_uns"] == "squares":
            params["n_points_uns"] = 64 if args.n_points <= 0 else args.n_points
            dataset_uns = SquaresToyDataset(n_sweeps=params["n_sweeps"], n_points=params["n_points_uns"],
                                            input_dims=input_dims, flow_dim=flow_dim, random_transition=True)

        elif params["dataset_uns"] == "fly":
            params["n_points_uns"] = 2048 if args.n_points <= 0 else args.n_points
            dataset_uns = FlyingThings3D(n_sweeps=params["n_sweeps"], n_points=params["n_points_uns"],
                                         input_dims=input_dims,
                                         verbose=True, partition="train", load_data_in_memory=True)


        elif params["dataset_uns"] == "shapenet":
            params["n_points_uns"] = 512 if args.n_points <= 0 else args.n_points
            dataset_uns = SingleShapeNet(n_sweeps=params["n_sweeps"], n_poinfts=params["n_points_uns"],
                                         partition="train", rotate=not args.freeze_rotation,
                                         overfit=args.overfit,
                                         data_size=None if args.data_size < 0 else args.data_size)

        elif params["dataset_uns"] == "mult_shape":
            params["n_points_uns"] = 512 if args.n_points <= 0 else args.n_points
            dataset_uns = MultiShapeNet(n_sweeps=params["n_sweeps"], n_points=params["n_points_uns"],
                                        partition="train", rotate=not args.freeze_rotation,
                                        overfit=args.overfit,
                                        data_size=None if args.data_size < 0 else args.data_size,
                                        occlusion=args.occlusion)

        elif params["dataset_uns"] == "kitti":
            params["n_points_uns"] = 2048 if args.n_points <= 0 else args.n_points
            dataset_uns = KITTI(n_points=params["n_points"], input_dims=input_dims, remove_ground=args.remove_ground,
                                verbose=True, load_data_in_memory=True, partition="train")

        elif params["dataset_uns"] == "lyft":
            params["n_points_uns"] = 2048 if args.n_points <= 0 else args.n_points
            dataset_uns = TorchLyftDataset(n_points=params["n_points"], input_dims=input_dims,
                                           remove_ground=args.remove_ground,
                                           verbose=True, load_data_in_memory=True, partition="train")

        else:
            raise Exception(f'Dataset {params["dataset_uns"]} Not implemented')

        dataloader_uns = DataLoader(dataset_uns, batch_size=params["batch_size"], shuffle=True, drop_last=True,
                                    num_workers=args.num_workers)

    # initialize flow extractor
    if params["flow_extractor"] == "flownet3d":
        flow_extractor = FlowNet3D(feature_dim=0, n_points=min(max(256, params["n_points"] // 2), 512))
        flow_extractor.apply(weights_init)
    elif params["flow_extractor"] == "segmenter":
        flow_extractor = Segmenter(n_points=params["n_points"],
                                   n_sweeps=params["n_sweeps"],
                                   in_channels=3,
                                   feat_size=1024,
                                   num_classes=3,
                                   classifier_layer_dims=[1024, 512, 256],
                                   feat_layer_dims=[128, 128, 256],
                                   activation=F.relu,
                                   batchnorm=not args.no_BN,
                                   transposed_input=True)

    elif params["flow_extractor"] == "ppwc":
        flow_extractor = PPWC()
    elif params["flow_extractor"] == "zero":
        flow_extractor = ZeroFlow()
    elif params["flow_extractor"] == "avg":
        flow_extractor = AvgFlow()
    elif params["flow_extractor"] == "knn":
        flow_extractor = knnFlow()
    else:
        raise Exception(f'Flow extractor  {args.flow_extractor} Not implemented')

    if args.load_model:
        model_dict = torch.load(args.load_model)
        try:
            flow_extractor = flow_extractor.to(device)
            flow_extractor.load_state_dict(model_dict["flow_extractor"])
        except RuntimeError:
            flow_extractor = nn.DataParallel(flow_extractor)
            flow_extractor = flow_extractor.to(device)
            flow_extractor.load_state_dict(model_dict["flow_extractor"])

    else:
        # Multi GPU stuff
        flow_extractor = nn.DataParallel(flow_extractor)
        flow_extractor = flow_extractor.to(device)

    params["cloud_embedder"] = {"flownet3d": "fn3dfeatcat",
                                "segmenter": "segfeat"}.get(params["flow_extractor"],
                                                            "fn3dfeatcat") if not args.cloud_embedder else args.cloud_embedder

    print(args)
    print(params)

    writer = SummaryWriter(log_dir=args.exp_name)

    # Log in tensorboard the args and params of the training before it starts
    log_args(tag="args", args=args, writer=writer, step=test_step)
    log_args(tag="params", args=params, writer=writer, step=test_step)

    # Lets train this motherfucker
    if args.train_type == "evaluate":
        stats_eval = eval_triplet(args, params, flow_extractor, None, dataloader_eval, writer, 0)
        print(stats_eval)
    elif "triplet" in args.train_type:
        train_triplet(args, params, flow_extractor, dataloader_train, dataloader_eval, writer)
    elif args.train_type in ("supervised", "knn"):
        train_supervised(args, params, flow_extractor, dataloader_train, dataloader_eval, writer)
    elif args.train_type == "semi_supervised":
        train_semi_supervised(args, params, flow_extractor, dataloader_train, dataloader_uns, dataloader_eval, writer)
    else:
        raise Exception(f'Training method  {args.train_type} Not implemented')

    # recursevely perform the overfit test
    if args.do_overfit_test:
        args.test_step += 1
        args.data_size = args.data_size * 2
        if args.data_size > 1024:
            args.do_overfit_test = False
        else:
            main(args)

    print_and_log(message=f"Training is finished.",
                  verbose=True,
                  add_timestamp=True,
                  global_step=args.epochs,
                  tensorboard_writer=writer)

    print("Training finished.")
    print(args)
    print(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Flow Prediction')

    # ### Training arguments
    parser.add_argument('--exp_name', type=str, default="toy_test",
                        help='Name of the experiment')
    parser.add_argument('--use_flow_signal', action="store_true", default=False,
                        help="uses flow supervision to train the flow extractor and the loss module")
    parser.add_argument('--epochs', type=int, default=500,  # 150,
                        help='number of epochs')
    parser.add_argument('--min_epochs', type=int, default=50,
                        help="minimum number of epochs to train the model before early stopping kicks in.")
    parser.add_argument("--train_type", type=str, default="triplet",
                        choices=("supervised", "triplet", "semi_supervised", "evaluate", "knn"),
                        help="chooses the type of training")
    parser.add_argument("--loss_type", default="triplet_l",
                        choices=("triplet", "triplet_l", "triplet_hinge", "triplet_inverse", "js"
                                                                                             "knn", "knn_kl",),
                        help="chooses the type of triplet loss training. (Previously called 'triplet_type')")
    parser.add_argument("--use_shallow_loss", default=False, action="store_true",
                        help="this option turns off the use of the deep loss")
    parser.add_argument("--loss_type_deep", type=str, default="",
                        choices=(
                        "triplet", "triplet_l", "triplet_hinge", "triplet_inverse", "js", "", "none", "emb_dist",
                        "dist"),
                        help="defines which time of deep loss is used. The empty string uses the default parameter,"
                             " 'none' uses shallow loss")
    parser.add_argument("--deep_loss_scale", default="depth_sub", type=str,
                        choices=("shape", "shape_sub", "depth_sub", "depth_lin", "depth_sup", ""),
                        help="rescale the deep loss based on the depth or activation map size")
    parser.add_argument("--retrieve_old_loss_module", default=False, action="store_true",
                        help="retrieves old loss module if training is stuck.")
    parser.add_argument("--cycle_consistency", default="",
                        help="use cycle consistency loss to auxiliate training")
    parser.add_argument("--cycle_consistency_sup", default="",
                        help="use cycle consistency loss to auxiliate supervised training. This option is overuled by "
                             "cycle_consistency when supervised training is selected.")
    parser.add_argument("--cycon_contribution", default=1.0, type=float,
                        help="contribution of the cycle consistency loss")
    parser.add_argument("--cycon_aug", action="store_true", default=False,
                        help="use pc2 to augment pc_pred in the cycle consistency loss")
    parser.add_argument("--norm_FE", default=2, type=float,
                        help="norm to be used when computing distances in the latent space in triplet training,"
                             " for the Flow Extractor. Use -1 if max norm is to be used.")
    parser.add_argument("--norm_LM", default=2, type=float,
                        help="norm to be used when computing distances in the latent space in triplet training,"
                             " for the Loss Module. Use -1 if max norm is to be used.")
    parser.add_argument("--fade_uns_loss", default=0, type=int,
                        help="Number of epochs to fade in the unsupervised loss in the semi-supervised case.")
    parser.add_argument("--max_fade_uns_loss", default=1.0, type=float,
                        help="ceiling value for the fade factor. that will multiply the unsupervised loss of the semi-"
                             "supervised setting.")
    parser.add_argument("--local_consistency", default=0.0, type=float,
                        help="contribution of the local consistency")
    # parser.add_argument("--loss_module_grad_penalty", type=float, default=0,
    #                     help="apply gradient penalty penalty in the loss module using given the norm. For no penalty "
    #                          "select 0, for max norm use -1.")
    parser.add_argument("--margin", default=1., type=float,
                        help="margin size for emb loss")
    parser.add_argument("--chamfer", default=0., type=float,
                        help="use of chamfer distance, set to zero if not used.")
    parser.add_argument("--laplace", default=0.0, type=float,
                        help="use laplace regularization, set to zero if not used.")
    parser.add_argument("--static_penalty", default=0, type=float,
                        help="applies a penalty if the predicted point cloud is too close to the original point cloud.")
    parser.add_argument("--reverse_LM", action="store_true", default=False,
                        help="reverses order of point clouds for an extra training step.")
    parser.add_argument("--reverse_FE", action="store_true", default=False,
                        help="reverses order of point clouds for an extra training step.")
    parser.add_argument("--sup_scale", default=1.0, type=float,
                        help="scaling used on the supervised loss")
    parser.add_argument("--seed", default=1234, type=int,
                        help="seed for random number generators")
    parser.add_argument("--save_all_scenes", default=False, action="store_true",
                        help="saves scenes and predictions from evaluations as point clouds")

    # ### Model Argumens
    parser.add_argument("--flow_extractor", type=str, default="",
                        choices=("", "flownet3d", "fn3dmini", "fn3dkitti", "lidarbug", "pwc", "segmenter", "zero",
                                 "avg", "knn", "ppwc"),
                        help="chose the model for flow extraction, empty string uses the default model for the Sandbox")
    parser.add_argument("--no_BN", default=False, action="store_true",
                        help="whether to use Batch norm or not")
    parser.add_argument("--cloud_embedder", default="", type=str,
                        choices=("", "fn3dfeatcat", "fn3dfeatcatfat", "fn3dfeat", "fn3ddiscdiff", "fn3ddischcat",
                                 "fn3ddisc", "segfeat", "segdisc", "fn3dfeatcat4l", "fn3dlogist", "fn3ddisclogist",
                                 "fn3ddist", "fn3ddist2", "emb"),
                        help="choses the model for the loss module, if empty string is given, then the default is used")
    parser.add_argument("--loss_feat_dim", default=4096, type=int,
                        help="size of the feature vector of the loss module.")
    parser.add_argument("--load_model", default="", type=str,
                        help="the path to the model to be loaded.")

    # ### Data argumetns
    parser.add_argument('--Sandbox', type=str, default="shapenet",
                        choices=("toy", "fly", "squares", "shapenet", "mult_shape", "kitti", "kitting", "lyft"),
                        help='choses the Sandbox')
    parser.add_argument('--dataset_uns', type=str, default='',
                        choices=("toy", "fly", "squares", "shapenet", "mult_shape", "kitti", "lyft"),
                        help='choses the Sandbox for supervised training, if empty string is given it will use '
                             'args.Sandbox')
    parser.add_argument('--dataset_eval', type=str, default='',
                        choices=("toy", "fly", "squares", "shapenet", "mult_shape", "kitti", "kitting",),
                        help='choses the Sandbox for evaluation, if empty string is given it will use '
                             'args.Sandbox')
    parser.add_argument('--n_points', type=int, default=-1,
                        help='number of points per point cloud, -1 uses the default number for the Sandbox')
    parser.add_argument('--height', default=-1, type=int,
                        help='vertical resolution of xyz maps (lidarbug like)')
    parser.add_argument('--width', default=-1, type=int,
                        help='horizontal (azimuth) resolution of xyz maps (lidarbug like)')
    parser.add_argument('--n_sweeps', default=2, type=int,
                        help='number of frames (or lidar sweeps) to use in training.')
    parser.add_argument("--batch_size", default=-1, type=int,
                        help="define batch size")
    parser.add_argument("--overfit", default=False, action="store_true",
                        help="overfit in one sample of data.")
    parser.add_argument("--data_size", default=-1, type=int,
                        help="Size of data to train with when overfiting")
    parser.add_argument("--test_step", default=0, type=int,
                        help="intendent for use with the overfit test")
    parser.add_argument("--do_overfit_test", action="store_true", default=False,
                        help="perform an overfit test over different bach sizes")
    parser.add_argument("--normalize_input", type=bool, default=True,
                        help="normalize supervised input to the norm of the unsupervised input.")
    parser.add_argument("--remove_ground", action="store_true", default=False,
                        help="used in the KITTI Sandbox, removes all points y < 0.3")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of cpu cores used for dataloading")
    parser.add_argument("--occlusion", action="store_true", default=False,
                        help="Aplly occlusions in the mult_shape Sandbox")

    # ### Optimizer arguments
    parser.add_argument("--opt", default="", type=str,
                        choices=("sgd", "adam", ""))
    parser.add_argument("--lr", default=-1, type=float,
                        help="learning rate to be used to train the Flow Extractor, -1 uses the default parameter")
    parser.add_argument("--lr_LM", default=-1, type=float,
                        help="learning rate to be used in the to train the Loss module, -1 uses the default parameter")
    parser.add_argument("--lr_factor", default=-1, type=float,
                        help="factor to scale the learning rate during training when the eval metric stop improving.")
    parser.add_argument("--lr_patience", default=-1, type=int,
                        help="Number of epochs to keep lr of the Flow Extractor before decreasing it, use -1 for "
                             "default parameter and 0 to not use the lr scheduler")
    parser.add_argument("--lr_patience_LM", default=-1, type=int,
                        help="Number of epochs to keep lr of the Loss Module before decreasing it, use -1 for "
                             "default parameter and 0 to not use the lr scheduler")
    parser.add_argument("--freeze_rotation", action="store_true", default=False,
                        help="freeze rotation of toy datasets, only rigid translation is applied")
    parser.add_argument("--skip_loss_module", type=float, default=1.0,
                        help="skip the training of the loss module when the accuracy drops bellow the given treshold")

    args = parser.parse_args()

    # create experiment name
    if args.exp_name != "toy_test":
        args.exp_name = args.exp_name + "_" + args.train_type + "_" + args.dataset + "_" + args.flow_extractor \
                        + "_" + parsedtime()

    if args.do_overfit_test:
        assert args.do_overfit_test == args.overfit, f"When performing a overfit test you must also call --overfit"
        assert args.train_type == "supervised", f"The overfit test is only compatible with the supervised training."

    main(args)

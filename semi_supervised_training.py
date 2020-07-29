from __future__ import print_function
import sys

sys.path.insert(0, '.')
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch

# my imports
from aml_utils import print_and_log, save_model, update_running_mean
from initializations import initialize_cloud_embedder, initialize_optimizers
from train_flow_extractor import train_step_flow_extractor
from train_cloud_embedder import train_step_cloud_embedder
from triplet_training import eval_triplet


def train_semi_supervised(args, params, flow_extractor, dataloader_sup, dataloader_uns, dataloader_eval, writer):
    """Supervised and triplet Training
        Input:
            args: args from main;
            params: dict, with training parameters;
            flow_extractor: nn.Module, flow model;
            data_loader_sup: pytorch dataloader, with data for supervised training;
            data_loader_uns: pytorch dataloader, with data for unsupervised training;
            data_loader_eval: pytorch dataloader, with the evaluation data;
            writer: tensorboard writer;
        Output:
            Writes training statistics to tensorboard writer;
            Saves best performing model to disc;
        """

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

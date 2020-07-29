from __future__ import print_function

import sys

sys.path.insert(0, '.')
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# my imports
from aml_utils import print_and_log, save_model, update_running_mean
from train_flow_extractor import train_step_flow_extractor


def train_supervised(args, params, flow_extractor, dataloader_train, dataloader_eval, writer):
    """Supervised Training
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

    # initialize training function for the flow extractor
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
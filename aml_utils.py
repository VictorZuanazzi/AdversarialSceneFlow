"""Implemented by Victor Zuanazzi
Lose helper functions"""
import os
import matplotlib as mpl

if os.environ.get('DISPLAY', '') == '':
    print('(pcf_utils) No display found. Using non-interactive Agg backend.')
    mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from datetime import datetime
import time
import numpy as np
import torch
import torch.nn as nn


# ################### Functions related to time ################### #
def parsedtime(str_parse=None):
    """Function to output the parsed current time.
    str_parse: str, should contain the desired date format.
        Default: '%Y_%b_%d_%H_%M_%S'"""
    today = datetime.today()
    str_parse = "%Y_%b_%d_%H_%M_%S" if str_parse is None else str_parse
    return today.strftime(str_parse)


def elapsed_time(time_0):
    """Calculates elapsed time in minutes.
    Args:
        time_0: object time.time(), previos time stamp
    return:
        float, elapsed time in minutes"""
    return (time.time() - time_0) / 60


# ################### Functions related to stats and metrics ################### #
def calculate_stats(time_series: list, name: str = "loss", sma_window: int = 100, show_min: bool = True,
                    show_max: bool = False) -> dict:
    """Calculate all wanted statics from the given time series
    Args:
        name: str, name of the metric for which stats will be calculated.
        time_series: list, a list containing all the values (floats or ints) taken be the metric of interest.
        sma_window: int, the size of the window for the simple moving average.
        show_min: bool, if min should be included in the stats.
        show_max: bool, if max should be included in the stats.
    :return {'sma': float, 'mean': float, min: float, max: float name: float}
        """
    if not time_series:
        time_series = [0]

    stats_dict = {"sma": np.mean(time_series[-min(len(time_series), sma_window):]),
                  "mean": np.mean(time_series),
                  name: time_series[-1]}

    if show_min:
        stats_dict["min"] = np.min(time_series)

    if show_max:
        stats_dict["max"] = np.max(time_series)

    return stats_dict


def update_running_mean(main_dict, source_dict, counter, pre_string="", pos_string=""):
    """updates the running mean of main_dict with the incomming values of source dict
    input:
        main_dict: collections.defaultdict: a dictionary where the running means are stored.
        source_dict: dict, a dictionary with the new values.
        counter: int > 0, the number of the update.
        pre_string and pos_string: str, main_dict.keys()[i] = pre_string + source_dict.keys()[i] + pos_string.
    return:
        main_dict with updated values"""
    for key in source_dict:
        main_string = pre_string + key + pos_string
        main_dict[main_string] += (source_dict[key] - main_dict[main_string]) / counter

    return main_dict


def calculate_eucledian_metrics(pred_, target_, mask=None):
    """
    calculate the metrics in the eucledian space. End-point-error and accuracies 01 and 005.
    :param pred: torch.tensor() [B, C, N] the vectors to be compared.
    :param target: torch.tensor() [B, C, N] the verctors we want to compare to (the targets).
    :return: (dict, dict)
        error dict, dictionary containing metrics that can be interpreted as errors (lower the better),
        acc dict, dictionary containing metrics that can be interpreted as accuracies
            (higher the better and 0.0 <= acc <= 1.0).
    """

    if mask is None:
        mask = torch.tensor(1.)
        mask_factor = pred_.shape[2]
    else:
        mask = mask.any(dim=1, keepdim=True)
        mask_factor = mask.sum(dim=-1)
        mask_factor.squeeze_()

    pred = pred_ * mask
    target = target_ * mask

    # Caclualte the squared error
    pair_error = (target - pred).norm(dim=1)

    EPE = (pair_error.sum(dim=-1) / mask_factor).mean().item()

    target_norm = target.norm(dim=1)
    relative_error = pair_error / (target_norm + 1e-4)

    mask.squeeze_()

    # EPE < 0.1m or relative norms < 10%
    acc_01 = ((((pair_error < .1) + (relative_error < 0.1)) * mask).float().sum(dim=-1) / mask_factor).mean().item()
    # EPE < 0.05m or relative norms < 5%.
    acc_005 = ((((pair_error < .05) + (relative_error < 0.05)) * mask).float().sum(dim=-1) / mask_factor).mean().item()
    # EPE > 0.3 or relative norms > 0.1
    outlier = ((((pair_error > 0.3) + (relative_error > 0.1)) * mask).float().sum(dim=-1) / mask_factor).mean().item()

    cos = nn.CosineSimilarity(dim=1)(pred, target)
    cos_error = (((- cos + 1) * mask).sum(dim=-1) / mask_factor).mean().item()
    cos_limit = torch.tensor(2.5 * np.pi / 180).cos()
    cos_acc = (((cos > cos_limit) * mask).float().sum(dim=-1) / mask_factor).mean().item()

    error_dict = {"epe": EPE, "outlier": outlier, "cos_flow_error": cos_error}
    acc_dict = {"acc_01": acc_01, "acc_005": acc_005, "cos_flow_acc": cos_acc}

    return error_dict, acc_dict


def running_average(old_average: float, new_value: float, n: int) -> float:
    """
    Calculates the running average using the following formula:
        running_average = (old_average * (n - 1) + new_value) / n
    :param old_averare: float, the average so far.
    :param new_value: float, the new value to update the average with.
    :param n: int, number of values taken in the old average and the new_value.
    :return: float, the running average.
    """
    return (old_average * (n - 1) + new_value) / max(n, 1)


# ################### Functions related to printing and loging and saving ################### #
def print_and_log(message: str, verbose: bool = False, add_timestamp: bool = False,
                  global_step: int = 0, tensorboard_writer=None, file_name: str = None) -> None:
    """Print a message and save it in a log file.
    Args:
        file_name: str, the name (with path and extension) of the file.
        message: str, the string to be printed and saved.
        verbose: bool, whether to print the message or not.min(idx-1, 0)
        add_timestamp: boo, whether to add a time stamp on front of the message or not."""

    if add_timestamp:
        message = "\n[" + parsedtime() + "] " + message

    if verbose:
        print(message)

    if file_name is not None:
        with open(file_name, "a+") as file:
            file.write(message + '\n')

    if tensorboard_writer is not None:
        tag = file_name.split("/")[0] if file_name is not None else "log"
        tensorboard_writer.add_text(tag=tag,
                                    text_string=message,
                                    global_step=global_step)


def save_args(file_name: str, args) -> None:
    """Save arguments in the given file.
    Args:
        file_name: str, file_name with path and extension. If file does not exist, it will be created.
        args: Namespace, Namespace containing the arguments to be saved.
        """
    divider = '\n' + '#' * 30 + '\n'
    underline = '\n' + '-' * 20 + '\n'
    args_dict = vars(args)
    with open(file_name, "w+") as params_file:
        params_file.write(divider + 'Argument Parameters:' + underline)

        # write all parameters
        for param in args_dict.keys():
            params_file.write(str(param) + ": " + str(args_dict[param]) + '\n')

        params_file.write(divider)


def log_args(tag, args, writer, step=0) -> None:
    """Log args in tensorboard.
    Args:
        tag: str, the tag of what is being logged.
        args: Namespace or dict, containing the arguments to be saved.
        writer: tensorboard writer
        step: int, global step to be used in the tensorboard log.
        """
    args_dict = vars(args) if type(args) is not dict else args
    text = "##### #### ####"
    # write all parameters
    for param in args_dict.keys():
        text += f"\n{param}: {args_dict[param]} \n"

    text += "\n##### #### ####\n"
    writer.add_text(tag=tag, text_string=text, global_step=step)


def save_model(dict_to_be_saved, file_name, retry=True):
    """save dict using torch.
    Args:
        dict_to_be_saved: dict, to contain what has to be saved, it may include the model.state_dict and loss curves.
        file_name: str, file name containing the path and extension of the file.
        retry: bool, whether to try one more time in case the saving failed for any reason"""
    try:
        # save file
        torch.save(dict_to_be_saved, file_name)
        return True

    except:
        print_and_log(file_name=file_name + "_Save_failed.txt",
                      message=f"{file_name} could not be saved. Retry is {retry}",
                      verbose=True,
                      add_timestamp=True)
        if retry:
            return save_model(dict_to_be_saved, file_name, retry=False)

    return False


def magic_colors(img_cloud):
    """defines the colors to be used for plotting based on the number of channels
    Args:
        img_cloud: np.array(float)[xyz + rbg, n_points], with dimensions that encode color. It assumes the first 3
        coordinates to be dimensional coordinates and the following 3 to encode the rbg channels.
    output:
        np.array(float), in the range [0, 1]. The output will be 1 dimensional if the input is one dimensional or if
            channel == 4. Otherwise the output is two dimensional [n_points, rgb]"""

    if img_cloud.ndim == 1:
        # makes everything positive, starting at zero:
        color_cloud = img_cloud - img_cloud.min()
        # There are no channels
        return (color_cloud / max(color_cloud.max(), 1)).clip(min=0.0, max=0.9)

    # makes everything positive, starting at zero:
    color_cloud = img_cloud - img_cloud.min(axis=1, keepdims=True)

    if color_cloud.shape[0] >= 6:
        # assumes the color is encoded in the channels 4, 5 and 6
        colors = (color_cloud[3:6, :] / color_cloud[3:6, :].max(axis=1, keepdims=True)).clip(min=0.0, max=0.9)

    else:
        colors = 1 - ((color_cloud[2, :] - color_cloud[2, :].min()) / (color_cloud[2, :].max() - color_cloud[2, :].min())).clip(min=0.1, max=0.9)


    return colors.T


def plot_2d_clouds(dataset, args, clouds, flow_target, tokens, cloud_pred, flow_pred, dot_size):
    # TODO(5): clean this function
    """ Plot point clouds in the XY projection in a way that makes it more intuitive to understand what is happening.
    Code dump!
    :param flow_pred:
    :param flow_target:
    :param dataset:
    :param args:
    :param clouds:
    :param tokens:
    :param cloud_pred:
    :param dot_size:
    :return: plt.figure
    """

    # For qualitative analysis, save images with one sample.
    fig, axs = plt.subplots(2, args.n_sweeps + 1,
                            figsize=(16, 8),
                            sharex=True, sharey=True)
    ref_cloud = None
    flow = None
    dot_size = 1.0

    # if tokens is None:
    #     tokens = torch.zeros_like(flow_pred)

    for a in range(args.n_sweeps + 1):
        if a < args.n_sweeps:
            # Also, add a title and sub-titles.
            idxs = np.arange(a * args.n_points, (a + 1) * args.n_points)
            img_cloud = clouds[0, :, idxs].to(torch.device("cpu")).numpy()

            colors = magic_colors(img_cloud)
            axs[0, a].scatter(img_cloud[0, :], img_cloud[1, :], c=colors, s=dot_size)
            title_dict = {0: f"Input LiDAR sweep, t={a}",
                          1: f"Input LiDAR sweep, t={a}",
                          2: f"Target LiDAR sweep, t={a}"}
            axs[0, a].set_title(title_dict[a])

            # Plot available annotations
            dataset.plot_annotations(axs[1, a], ref_cloud, img_cloud,
                                     tokens=tokens if tokens is None else tokens[a][0],
                                     flow=flow)
            axs[1, a].set_title(f"Annotations: t={a}")
            ref_cloud = img_cloud.copy()

            # flow is relative to the a - 1 cloud
            if flow_target is not None:
                flow = flow_target[0, :, np.minimum(idxs,
                                                    flow_target.shape[2] - 1)].to(torch.device("cpu")).numpy()

        else:
            # plot predicted point cloud
            img_cloud = cloud_pred[0, :, :].detach().to(torch.device("cpu")).numpy()
            flow = flow_pred[0, :, :].detach().to(torch.device("cpu")).numpy()

            colors = magic_colors(img_cloud)
            axs[0, a].scatter(img_cloud[0, :], img_cloud[1, :], c=colors, s=dot_size)
            axs[0, a].set_title(f"Predicted LiDAR sweep")

            # Quiver plot (fancy arrows) of predicted transition
            (begin, end) = (- 2 * args.n_points, -args.n_points) if args.n_sweeps > 2 else (0, args.n_points)
            ref_cloud = clouds[0, :, begin:end].to(torch.device("cpu")).numpy()
            dataset.plot_quiver(axs[1, a], ref_cloud, img_cloud, flow=flow)
            axs[1, a].set_title(f"Estimated flow")

        # Show ego vehicle.
        axs[0, a].plot(0, 0, "x", color="red")
        axs[1, a].plot(0, 0, "x", color="red")

        # save image
        plt.savefig(args.save_dir + str(args.global_step))

    return fig


def plot_3d_clouds(dataset, args, clouds, flow_target, tokens, cloud_pred, flow_pred, dot_size):
    # For qualitative analysis, save images with one sample.

    plot_shape = [2, args.n_sweeps + 1]
    fig, axs = plt.subplots(plot_shape[0], plot_shape[1],
                            figsize=(16, 8),
                            sharex=True, sharey=True)
    ref_cloud = None
    flow = None

    # if tokens is None:
    #     tokens = torch.zeros_like(flow_pred)

    for a in range(args.n_sweeps + 1):
        if a < args.n_sweeps:
            # Also, add a title and sub-titles.
            # Also, make it a separate function.
            idxs = np.arange(a * args.n_points, (a + 1) * args.n_points)
            img_cloud = clouds[0, :, idxs].to(torch.device("cpu")).numpy()

            colors = magic_colors(img_cloud)
            axs[0, a] = fig.add_subplot(plot_shape[0], plot_shape[1], a + 1, projection="3d")
            axs[0, a].scatter(img_cloud[0, :], img_cloud[1, :], img_cloud[2, :], c=colors, s=dot_size)
            title_dict = {0: f"Input LiDAR sweep, t={a}",
                          1: f"Input LiDAR sweep, t={a}",
                          2: f"Target LiDAR sweep, t={a}"}
            axs[0, a].set_title(title_dict[a])

            # Plot available annotations
            axs[1, a] = fig.add_subplot(plot_shape[0], plot_shape[1], a + plot_shape[1] + 1, projection="3d")
            dataset.plot_annotations(axs[1, a], ref_cloud, img_cloud,
                                     tokens=tokens if tokens is None else tokens[a][0],
                                                   flow=flow)
            axs[1, a].set_title(f"Annotations: t={a}")
            ref_cloud = img_cloud.copy()

            # flow is relative to the a - 1 cloud
            if flow_target is not None:
                flow = flow_target[0, :, np.minimum(idxs,
                                                    flow_target.shape[2] - 1)].to(torch.device("cpu")).numpy()


        else:
            # plot predicted point cloud
            img_cloud = cloud_pred[0, :, :].detach().to(torch.device("cpu")).numpy()
            flow = flow_pred[0, :, :].detach().to(torch.device("cpu")).numpy()

            colors = magic_colors(img_cloud)
            axs[0, a] = fig.add_subplot(plot_shape[0], plot_shape[1], a + 1, projection="3d")
            axs[0, a].scatter(img_cloud[0, :], img_cloud[1, :], img_cloud[2, :], c=colors, s=dot_size)
            axs[0, a].set_title(f"Predicted LiDAR sweep")

            # Quiver plot (fancy arrows) of predicted transition
            (begin, end) = (- 2 * args.n_points, -args.n_points) if args.n_sweeps > 2 else (0, args.n_points)
            ref_cloud = clouds[0, :, begin:end].to(torch.device("cpu")).numpy()
            axs[1, a] = fig.add_subplot(plot_shape[0], plot_shape[1], a + 1 + plot_shape[1], projection="3d")
            dataset.plot_quiver(axs[1, a], ref_cloud, img_cloud, flow=flow)
            axs[1, a].set_title(f"Estimated flow")

        plt.savefig(args.save_dir + str(args.global_step) + "_3d")

        # Show ego vehicle.
        axs[0, a].plot([0], [0], [0], "x", color="red")
        axs[1, a].plot([0], [0], [0], "x", color="red")

        # save image
        plt.savefig(args.save_dir + str(args.global_step) + "_3d")

    return fig


def plot_from_dict(dict, save_dir, title="Losses"):
    # TODO(5): Make this plot look professional (seaborn)
    fig, ax = plt.subplots(figsize=(12, 4))
    for key in dict.keys():
        ax.plot(dict[key], label=key)

    ax.legend()
    ax.set_title(label=title)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Step")
    fig.tight_layout()
    plt.savefig(save_dir + title)


# ################### Others ################### #
def make_rotation_matrix(angle, stretch):
    """make a rotation matrix given the angle
    angle: torch.tensor, with shape (n, 3,) containing angle in rads
    strech: torch.tensor with shape (n, 3,) containing the stretch coeficient"""

    cos = angle.cos()
    sin = angle.sin()
    zeros = torch.zeros(size=(angle.shape[0], 1, 1))

    rx = torch.cat((torch.cat((stretch[..., 0:1], zeros, zeros), dim=2),
                    torch.cat((zeros, cos[..., 0:1], -sin[..., 0:1]), dim=2),
                    torch.cat((zeros, sin[..., 0:1], cos[..., 0:1]), dim=2)), dim=1)
    ry = torch.cat((torch.cat((cos[..., 1:2], zeros, sin[..., 1:2]), dim=2),
                    torch.cat((zeros, stretch[..., 1:2], zeros), dim=2),
                    torch.cat((-sin[..., 1:2], zeros, cos[..., 1:2]), dim=2)), dim=1)
    rz = torch.cat((torch.cat((cos[..., 2:3], -sin[..., 2:3], zeros), dim=2),
                    torch.cat((sin[..., 2:3], cos[..., 2:3], zeros), dim=2),
                    torch.cat((zeros, zeros, stretch[..., 2:3]), dim=2)), dim=1)

    rotation_matrix = rx @ ry @ rz

    return rotation_matrix.transpose(2, 1)


def divide_input_target(clouds, n_points):
    """"""
    # make explicit the distinction between input and target point clouds.
    # at least 2 point clouds must be used for flow extraction.
    # If only 2 point clouds are given, then the 2nd one is also used as target.
    n_sweeps = clouds.shape[2] // n_points
    if n_sweeps > 2:
        input_idx = (n_sweeps - 1) * n_points
        target_idx = input_idx
    else:
        input_idx = 2 * n_points
        target_idx = n_points

    input_clouds = clouds[:, :, :input_idx]
    target_cloud = clouds[:, :, target_idx:]

    return input_clouds, target_cloud


def random_idxs(input_size, output_size, with_replacement=False):
    """samples indexes with or without replacement.
    Args:
        input_size: (int), the size of the vector we want to sample from.
        output_size: (int), the number of samples.
        with_replacement: (int), True for sampling with replacement.
    Return:.
        unidimensonal np.array(int) of size output_size with the indexes for sampling.
    """

    if (input_size < output_size) or with_replacement:
        # If the input size < output_size, the sampling must be with replacement.
        return np.random.randint(low=0, high=input_size, size=output_size)

    # Sampling without replacement.
    # The sampling does not follow a normal distribution. It approximates it by dividing the range of the vector in
    # equally spaced windows and sampling one index per window.
    window = np.floor(input_size / output_size).astype(int)
    wiggle = input_size % output_size
    rnd_start = np.random.randint(low=0, high=(1 + wiggle))
    rnd_stop = rnd_start + input_size - wiggle
    base_idx = np.arange(start=rnd_start, stop=rnd_stop, step=window, dtype=int)
    rand_idx = np.random.randint(low=0, high=window, size=len(base_idx), dtype=int)
    out_idx = base_idx + rand_idx
    return out_idx


def select_inputs_(use_flow_signal, n_points, half_cloud=True):
    """creates a function that selects the input given to the loss module given depending if flow signal is to be
    used or not.
    input: bool
    return: function"""
    # if use signal is on, then the c1 + flow is used as positive example, otherwise c2 is used.
    if use_flow_signal:
        def select_inputs_func(c_cond, c_input, c_pred, flow_input):
            """The four inputs are related in the following way:
                c_cond: is the point cloud where the movement is conditioned on, usually c1;
                c_input: is the point cloud we want to compare with, usually c2;
                c_pred: is the predicted point cloud, usually c_pred = c_cond + flow_pred;
                flow_input: is the ground truth flow, with respect to c_cond, such that c_cond + flow_input ~ c_input.
            returns:
                (c_cond, c_input, c_cond + flow_input, c_pred)"""
            c_displaced = c_cond + flow_input
            return c_cond, c_input, c_displaced, c_pred
    else:
        if half_cloud:
            i_odds = torch.arange(start=1, end=n_points, step=2)
            i_evens = torch.arange(start=0, end=n_points, step=2)
        else:
            i_odds = torch.arange(start=0, end=n_points, step=1)
            i_evens = torch.arange(start=0, end=n_points, step=1)

        def select_inputs_func(c_cond, c_input, c_pred, flow_input=None):
            """The four inputs are related in the following way:
                c_cond: is the point cloud where the movement is conditioned on, usually c1;
                c_input: is the point cloud we want to compare with, usually c2;
                c_pred: is the predicted point cloud, usually c_pred = c_cond + flow_pred;
                flow_input: is the ground truth flow. It is not used, the argument is only taken for compatibility.
            returns:
                (c_cond[odds], c_input[odds], c_input[evens], c_pred[evens])"""
            return c_cond[:, :, i_odds], c_input[:, :, i_odds], c_input[:, :, i_evens], c_pred[:, :, i_evens]

    return select_inputs_func
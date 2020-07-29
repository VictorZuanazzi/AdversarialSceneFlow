"""Implemented by Victor Zuanazzi
Code for Adversarial Metric Learning """

from __future__ import print_function

import sys

sys.path.insert(0, '.')
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# my imports
from Sandbox.toy_data import SquaresToyDataset
from Sandbox.singleshapenet import SingleShapeNet
from Sandbox.multishapenet import MultiShapeNet
from Sandbox.flyingthings3d import FlyingThings3D
from Sandbox.kitti import KITTI, KittiNoGround
from Sandbox.lyft import TorchLyftDataset
from aml_utils import parsedtime, log_args, print_and_log
from Models.flow_extractor import Segmenter, weights_init, ZeroFlow, AvgFlow, knnFlow, PPWC, FlowNet3D
from triplet_training import eval_triplet, train_triplet
from supervised_training import train_supervised
from semi_supervised_training import train_semi_supervised


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

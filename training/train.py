import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from path import Path

import models
from global_attacks.perturb_model import PerturbationsModel, compute_epe
from models.utils_model import predict_flow
from training.evaluate import validate_chairs, validate_kitti, validate_sintel
from training.utils import (
    Logger,
    count_parameters,
    fetch_dataloader,
    fetch_optimizer,
    multiscale_epe,
    sequence_loss,
)

# pylint: disable=ungrouped-imports
try:
    from torch.cuda.amp import GradScaler
except ImportError:
    print("Use dummy GradScaler")
    # dummy GradScaler for PyTorch < 1.6

    class GradScaler:
        def __init__(self, **kwargs):
            pass

        def scale(self, loss):  # pylint: disable=no-self-use
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):  # pylint: disable=no-self-use
            optimizer.step()

        def update(self):
            pass


# global params for training
CLUSTER_MAX_TIME = 24 * 3600 - 1000
VAL_FREQ = 5000
INNER_ITERATION = 3  # to speed-up training


def get_model(args):
    if args.flowNetC:
        if args.flexible_larger_field:
            model = getattr(models, "FlowNetC_flexible_larger_field")(
                div_flow=args.div_flow,
                kernel_size=args.kernel_size,
                number_of_reps=args.numReps,
                dilation=args.dilation,
            )
        elif args.larger_field:
            model = getattr(models, "FlowNetC_larger_field")(div_flow=args.div_flow)
        # elif args.larger_field2:
        #     model = getattr(models, "FlowNetC_2xlarger_field")(div_flow=args.div_flow)
        # elif args.relu_wo_out:
        #     model = getattr(models, "FlowNetC_relu_wo_out")(div_flow=args.div_flow)
    # elif args.flowNetCFlexible:
    #     args.corr_depth = args.flowNetC_corrDepth
    #     args.separate_context = args.flowNetC_separateContext
    #     args.raft_encoder = args.flowNetC_raftEncoder
    #     if args.predict_bias:
    #         model = getattr(models, "FlowNetCFlexible_withOnlyPredictFlowBias")(
    #             args,
    #             div_flow=args.div_flow,
    #             raft_kernel_size=args.raft_kernel_size,
    #             raft_no_out_conv=args.raft_no_out_conv,
    #             raft_leaky_relu=args.raft_leaky_relu,
    #             raft_small=args.raft_small,
    #         )
    #     else:
    #         model = getattr(models, "FlowNetCFlexible")(args, div_flow=args.div_flow)
    elif args.pwc:
        model = getattr(models, "PWCDCNet")()
    elif args.pwcflex:
        pwc_correlations = list(map(int, args.pwc_correlations))
        model = models.PWCNetFlex.PWCDCNetFlex(correlations_on=pwc_correlations)
    else:
        model = models.raft.raft.RAFT(args)
    return model


def train(args):
    cluster_start_time = time.time()

    # load model
    model = get_model(args)
    model.cuda()
    model.train()
    print("Parameter Count: %d" % count_parameters(model))

    train_loader = fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(
        args=args, model=model, inner_iteration=INNER_ITERATION if args.adv_train else 1
    )
    total_steps = 0

    if args.restore_ckpt is not None:
        print(f"Restore checkpoint {args.restore_ckpt}")
        if args.fully_trained or args.DEBUG:
            weights = torch.load(args.restore_ckpt)
            try:
                model.load_state_dict(weights, strict=True)
            except Exception:
                model = nn.DataParallel(model, device_ids=args.gpus)
                model.load_state_dict(weights, strict=True)
        else:
            checkpoint = torch.load(args.restore_ckpt)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            total_steps = checkpoint["total_steps"] + 1

    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model, device_ids=args.gpus)
    model.cuda()
    model.train()

    if (
        args.stage != "chairs"
        and not args.flowNetCFlexible
        and not args.flowNetC
        and not args.pwc
    ):
        model.module.freeze_bn()

    scaler = GradScaler(enabled=args.mixed_precision)
    if not args.DEBUG:
        logger = Logger(model, scheduler, args.ckpt_dir, total_steps)

    if args.adv_train:
        VAL_FREQ = 1000
    elif args.finetune:
        VAL_FREQ = 100
    else:
        VAL_FREQ = 5000

    should_keep_training = True
    if args.arbitrary_gt:
        train_dataset = train_loader.dataset

    # Main training loop
    while should_keep_training:
        for _, data_blob in enumerate(train_loader):
            if len(data_blob) == 4:
                image1, image2, flow, valid = (x.cuda() for x in data_blob)
            elif len(data_blob) == 5:
                image1, image2, gt_full, flow, valid = (x.cuda() for x in data_blob)

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(
                    0.0, 255.0
                )
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(
                    0.0, 255.0
                )

            if args.adv_train:
                model.eval()
                perturb_model = PerturbationsModel(
                    perturb_method=args.perturb_method,
                    perturb_mode=args.perturb_mode,
                    output_norm=args.output_norm,
                    n_step=args.perturb_n_step,
                    learning_rate=args.perturb_learning_rate,
                    momentum=args.perturb_momentum,
                    probability_diverse_input=args.probability_diverse_input,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    disparity=False,
                    targeted=True if args.arbitrary_gt else False,
                    show_perturbation_evolution=None,
                    print_out=False,
                    args=args,
                )
                if args.arbitrary_gt:
                    arbitrary_gt_index = random.choice(range(len(train_dataset)))
                    arbitrary_sample_blob = train_dataset[arbitrary_gt_index]
                    if len(arbitrary_sample_blob) == 4:
                        arb_flow = arbitrary_sample_blob[2].cuda()
                        arb_valid = arbitrary_sample_blob[3].cuda()
                    elif len(arbitrary_sample_blob) == 5:
                        arb_flow = arbitrary_sample_blob[3][None, ...].cuda()
                        arb_valid = arbitrary_sample_blob[4][None, ...].cuda()
                    perturb_gt = torch.cat((arb_flow, arb_valid[:, None, ...]), dim=1)
                else:
                    perturb_gt = torch.cat((flow, valid[:, None, ...]), dim=1)

                (_, _, image1_output, image2_output,) = perturb_model.forward(
                    model=model,
                    image0=image1,
                    image1=image2,
                    ground_truth=perturb_gt,
                )
                flow_output = predict_flow(
                    model,
                    None,
                    image1_output,
                    image2_output,
                    args,
                    return_feat_maps=False,
                )
                epe_attacked = compute_epe(gt=gt_full, pred=flow_output)

                # prepare "batch" (clean + adv) for fine-tuning
                image1 = torch.cat((image1, image1_output))
                image2 = torch.cat((image2, image2_output))
                flow = torch.cat((flow, flow))
                valid = torch.cat((valid, valid))
                model.train()

            # To speed-up training, reuse batch multiple times during adv. train
            for _ in range(INNER_ITERATION if args.adv_train else 1):
                optimizer.zero_grad()

                if args.adv_train or args.finetune:
                    if args.flowNetC or args.flowNetCFlexible or args.pwc:
                        flow_predictions = model(image1, image2)
                    else:  # RAFT
                        flow_predictions = model(
                            image1 * 255.0,
                            image2 * 255.0,
                            iters=3 if args.DEBUG else args.iters,
                        )
                else:
                    if args.flowNetC or args.flowNetCFlexible:
                        flow_predictions = model(image1 / 255.0, image2 / 255.0)
                    else:  # RAFT
                        flow_predictions = model(
                            image1, image2, iters=3 if args.DEBUG else args.iters
                        )

                if args.multiscaleEPE:
                    loss, metrics = multiscale_epe(
                        flow_predictions,
                        flow,
                        valid,
                        args.gamma,
                        flowNetC=args.flowNetC or args.flowNetCFlexible,
                        not_excluding=args.no_excluding,
                        div_flow=args.div_flow,
                        flownetc_weighing=args.flownetc_weighing,
                        pwc=args.pwc,
                    )
                else:
                    loss, metrics = sequence_loss(
                        flow_predictions,
                        flow,
                        valid,
                        args.gamma,
                        flowNetC=args.flowNetC or args.flowNetCFlexible,
                        not_excluding=args.no_excluding,
                        div_flow=args.div_flow,
                        flownetc_weighing=args.flownetc_weighing,
                        pwc=args.pwc,
                    )

                if any(torch.isnan(torch.tensor([loss]))):
                    break

                if args.adv_train:
                    scaler.scale(loss).backward(retain_graph=True)
                else:
                    scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                scaler.step(optimizer)
                scheduler.step()
                scaler.update()

            if args.adv_train and not args.finetune:
                metrics["epe_attacked"] = epe_attacked
            if not args.DEBUG:
                logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1 and not args.DEBUG:
                PATH = args.ckpt_dir / "checkpoint.pth"
                torch.save(
                    {
                        "model_state_dict": model.module.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "total_steps": total_steps,
                    },
                    PATH,
                    _use_new_zipfile_serialization=False,
                )

                if args.validation is not None:
                    results = {}
                    for val_dataset in args.validation:
                        if args.flowNetC or args.flowNetCFlexible:
                            if val_dataset == "chairs":
                                results.update(
                                    validate_chairs(model.module, flowNetC=True)
                                )
                            elif val_dataset == "sintel":
                                results.update(
                                    validate_sintel(model.module, flowNetC=True)
                                )
                            elif val_dataset == "kitti":
                                results.update(
                                    validate_kitti(model.module, flowNetC=True)
                                )
                        else:
                            if val_dataset == "chairs":
                                results.update(validate_chairs(model.module, args.iters))
                            elif val_dataset == "sintel":
                                results.update(validate_sintel(model.module, args.iters))
                            elif val_dataset == "kitti":
                                results.update(validate_kitti(model.module, args.iters))

                    logger.write_dict(results)

                    model.train()

                    if (
                        args.stage != "chairs"
                        and not args.flowNetCFlexible
                        and not args.flowNetC
                        and not args.pwc
                    ):
                        model.module.freeze_bn()

            total_steps += 1

            if (
                total_steps > args.num_steps
                or time.time() - cluster_start_time > CLUSTER_MAX_TIME
            ):
                should_keep_training = False
                break

    logger.close()

    PATH = args.ckpt_dir / f"final_{total_steps-1}.pth"
    torch.save(model.module.state_dict(), PATH, _use_new_zipfile_serialization=False)

    return PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training pipelines")
    parser.add_argument("--name", type=str, default="raft", help="name your experiment")
    parser.add_argument(
        "--stage",
        type=str,
        help="determines which dataset to use for training. For adv. training corresponds to dataset",
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="checkpoint", help="Directory to save checkpoints"
    )
    parser.add_argument("--restore_ckpt", type=str, help="restore checkpoint")
    parser.add_argument("--flownet", type=str, default="RAFT", help="flownet to use")

    # Training args
    parser.add_argument("--lr", type=float, default=0.000125)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--image_size", type=int, nargs="+", default=[256, 640])
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--no_excluding", action="store_true", help="do not exclude max displacements"
    )
    parser.add_argument("--add_noise", action="store_true")
    parser.add_argument(
        "--trans_rot",
        action="store_true",
        help="Use additionally translation & rotation for augmentations",
    )
    parser.add_argument("--validation", type=str, nargs="+")
    parser.add_argument(
        "--multiscaleEPE", action="store_true", help="Use multiscale epe loss function"
    )

    # RAFT + variants args
    parser.add_argument("--corr_levels", type=int, default=4)
    parser.add_argument("--iters", type=int, default=12)
    parser.add_argument("--wdecay", type=float, default=0.0001)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.8, help="exponential weighting")
    parser.add_argument(
        "--fnorm",
        type=str,
        default="instance",
        choices=["none", "batch", "group", "instance"],
    )
    parser.add_argument(
        "--cnorm",
        type=str,
        default="batch",
        choices=["none", "batch", "group", "instance"],
    )
    parser.add_argument(
        "--raft_small", action="store_true", help="Use small RAFT Encoder"
    )
    parser.add_argument(
        "--raft_kernel_size", type=int, default=3, help="Use this RAFT kernel size"
    )
    parser.add_argument(
        "--update_no_motion_downsampling",
        action="store_true",
        help="Use RAFT Update without downsampling",
    )
    parser.add_argument("--net_relu", action="store_true", help="Use RAFT net relu")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument("--no_residuals", action="store_true", help="use no residuals")
    parser.add_argument(
        "--no_separate_context",
        action="store_true",
        help="use no separate context encoder",
    )
    parser.add_argument(
        "--no_sep_gru", action="store_true", help="use no separate GRU unit"
    )
    parser.add_argument("--flowNetCEnc", action="store_true", help="use FlowNetC Encoder")
    parser.add_argument(
        "--compute_spatial", action="store_true", help="use spatial correlation layer"
    )
    parser.add_argument(
        "--iterative", action="store_true", help="use iterative refinement in extractor"
    )
    parser.add_argument(
        "--single", action="store_true", help="use single layer extractor"
    )

    # FlowNetC + Variants Args
    parser.add_argument("--div_flow", type=int, default=1)
    parser.add_argument(
        "--flownetc_weighing", action="store_true", help="Use FlowNetC weighing scheme"
    )
    parser.add_argument(
        "--larger_field2",
        action="store_true",
        help="Train FlowNetC with extra 2x 5x5 convs",
    )
    parser.add_argument(
        "--relu_wo_out",
        action="store_true",
        help="Train FlowNetC with ReLUs and without outgoing ReLU",
    )
    parser.add_argument(
        "--flowNetCFlexible", action="store_true", help="Train FlowNetC Flexible"
    )
    parser.add_argument(
        "--flowNetC_corrDepth",
        type=int,
        default=3,
        help="Train FlowNetC with correlation at certain depth",
    )
    parser.add_argument(
        "--flowNetC_separateContext",
        action="store_true",
        help="Train FlowNetC with separate context",
    )
    parser.add_argument(
        "--flowNetC_raftEncoder",
        action="store_true",
        help="Train FlowNetC with raft encoder",
    )
    parser.add_argument(
        "--raft_leaky_relu",
        action="store_true",
        help="Train FlowNetC with raft encoder and leaky relu",
    )
    parser.add_argument(
        "--raft_no_out_conv",
        action="store_true",
        help="Train FlowNetC without Conv at end",
    )
    parser.add_argument("--flowNetC", action="store_true", help="Train FlowNetC")
    parser.add_argument("--pinard", action="store_true", help="Train FlowNetC as Pinard")
    parser.add_argument(
        "--predict_bias",
        action="store_true",
        help="Train FlowNetC only with biases in predict",
    )
    parser.add_argument(
        "--larger_field",
        action="store_true",
        help="Train FlowNetC with extra 1x 5x5 convs",
    )

    parser.add_argument(
        "--flexible_larger_field",
        action="store_true",
        help="Train FlowNetC with flex larger field",
    )
    parser.add_argument(
        "--kernel_size", type=int, default=5, help="Use this FlowNetC kernel size"
    )
    parser.add_argument(
        "--numReps", type=int, default=0, help="Use this FlowNetC repition of kernels"
    )
    parser.add_argument(
        "--relu",
        action="store_true",
        help="Train FlowNetC and use ReLUs before correlation layer",
    )
    parser.add_argument(
        "--corr_conv",
        action="store_true",
        help="Train FlowNetC with 1x1 conv before corr and conv redir",
    )
    parser.add_argument(
        "--first_conv_5",
        action="store_true",
        help="Train FlowNetC with 5x5 at each level beginning but first",
    )
    parser.add_argument("--dilation", type=int, default=1, help="Use this dilation rate")

    # PWC-Net + Variants Args
    parser.add_argument("--pwc", action="store_true", help="Train PWC-Net")
    parser.add_argument(
        "--pwcflex",
        action="store_true",
        help="Train PWC-Net Flexible number of correlations",
    )
    parser.add_argument(
        "--pwc_correlations",
        default=[2, 3, 4, 5, 6],
        nargs="+",
        help="Use correlations of the respective layers",
    )

    # Adv. training args
    parser.add_argument("--adv_train", action="store_true", help="adversarial training")
    parser.add_argument("--arbitrary_gt", action="store_true", help="use arbitrary GT")
    parser.add_argument(
        "--online_subset",
        type=int,
        nargs="+",
        default=None,
        help="Subset for online adv training",
    )

    # Perturb model settings
    parser.add_argument(
        "--perturb_method",
        type=str,
        default="ifgsm",
    )
    parser.add_argument(
        "--perturb_mode",
        type=str,
        default="both",
    )
    parser.add_argument(
        "--output_norm", default=0.02, type=float, help="Output norm of noise"
    )
    parser.add_argument(
        "--perturb_learning_rate",
        type=float,
        default=2e-3,
        help="Learning rate (alpha) to use for optimizing perturbations",
    )
    parser.add_argument(
        "--perturb_n_step",
        type=int,
        default=40,
        help="Perturbation optimization steps",
    )
    parser.add_argument(
        "--perturb_momentum",
        type=float,
        default=0.47,
        help="Momentum (beta) used for momentum iterative fast gradient sign method",
    )
    parser.add_argument(
        "--probability_diverse_input",
        type=float,
        default=0.00,
        help="Probability (p) to use diverse input",
    )
    parser.add_argument(
        "--flow_loss",
        type=str,
        default="l2",
        choices=["cossim", "l2", "l1", "corr"],
        help="What type of flow loss to use",
    )

    # Fine-tuning
    parser.add_argument("--finetune", action="store_true", help="finetune")

    parser.add_argument("--DEBUG", action="store_true", help="Debug mode")
    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    args.ckpt_dir = Path(args.ckpt_dir) / f"{args.name}"

    first_training = False
    fully_trained = False
    if not os.path.exists(Path(args.ckpt_dir)) and not args.DEBUG:
        args.ckpt_dir.makedirs_p()
        first_training = True
        fully_trained = True

    if not first_training and not args.DEBUG:
        if os.path.isfile(Path(args.ckpt_dir) / "checkpoint.pth"):
            checkpoint = torch.load(Path(args.ckpt_dir) / "checkpoint.pth")
            if checkpoint["total_steps"] + 1 >= args.num_steps:
                fully_trained = True
        else:
            first_training = True

        if not fully_trained and not first_training:
            args.restore_ckpt = Path(args.ckpt_dir) / "checkpoint.pth"

    if first_training or not fully_trained:
        print(torch.cuda.get_device_name(0))
        args.fully_trained = fully_trained
        if not os.path.isfile(args.ckpt_dir / "args.json") and not args.DEBUG:
            with open(args.ckpt_dir / "args.json", "w", encoding="utf-8") as f:
                json.dump(vars(args), f, indent=6)
        train(args)

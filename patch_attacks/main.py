import argparse
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
from path import Path
from scipy.ndimage.interpolation import zoom
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm

from dataset_utils import custom_transforms
from dataset_utils.sequence_folders import SequenceFolder
from dataset_utils.validation_flow import ValidationFlowKitti2012, ValidationFlowKitti2015
from flowutils.flowlib import flow_to_image
from models.utils_model import fetch_model, get_flownet_choices, predict_flow
from patch_attacks.logger import AverageMeter, TermLogger
from patch_attacks.losses import compute_cossim, compute_epe
from patch_attacks.utils import tensor2array, transpose_image
from patch_attacks.utils_patch import (
    circle_transform,
    init_patch_circle,
    init_patch_from_image,
    init_patch_square,
    square_transform,
)

epsilon = 1e-8

parser = argparse.ArgumentParser(
    description="Generating Adversarial Patches for Optical Flow Networks",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--valset",
    dest="valset",
    type=str,
    default="kitti2015",
    choices=["kitti2015", "kitti2012"],
    help="Optical flow validation dataset",
)
parser.add_argument(
    "--patch-path", dest="patch_path", default="", help="Initialize patch from here"
)
parser.add_argument(
    "--mask-path", dest="mask_path", default="", help="Initialize mask from here"
)
parser.add_argument("--DEBUG", action="store_true", help="DEBUG Mode")
parser.add_argument(
    "--name",
    dest="name",
    type=str,
    default="demo",
    required=True,
    help="name of the experiment, checpoints are stored in checpoints/name",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers",
)
parser.add_argument(
    "-b", "--batch-size", default=4, type=int, metavar="N", help="mini-batch size"
)
parser.add_argument(
    "--lr", "--learning-rate", default=1e3, type=float, help="initial learning rate"
)
parser.add_argument(
    "--epochs", default=40, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument("--max-count", default=2, type=int, help="max count")
parser.add_argument(
    "--epoch-size",
    default=100,
    type=int,
    metavar="N",
    help="manual epoch size (will match dataset size if not set)",
)
parser.add_argument(
    "--momentum",
    default=0.9,
    type=float,
    metavar="M",
    help="momentum for sgd, alpha parameter for adam",
)
parser.add_argument(
    "--beta", default=0.999, type=float, metavar="M", help="beta parameters for adam"
)
parser.add_argument(
    "--weight-decay", "--wd", default=0, type=float, metavar="W", help="weight decay"
)
parser.add_argument(
    "--print-freq", default=10, type=int, metavar="N", help="print frequency"
)
parser.add_argument(
    "-e",
    "--evaluate",
    dest="evaluate",
    action="store_true",
    help="evaluate model on validation set",
)
parser.add_argument(
    "--flownet",
    dest="flownet",
    type=str,
    default="FlowNetC",
    choices=get_flownet_choices(),
    help="flow network architecture. Options: FlowNetS | SpyNet",
)
parser.add_argument("--alpha", default=0.0, type=float, help="regularization weight")
parser.add_argument(
    "--image-size",
    type=int,
    default=384,
    help="the min(height, width) of the input image to network",
)
parser.add_argument(
    "--patch-type", type=str, default="circle", help="patch type: circle or square"
)
parser.add_argument(
    "--patch-size", type=float, default=0.01, help="patch size. E.g. 0.05 ~= 5% of image "
)
parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="seed for random functions, and network initialization",
)
parser.add_argument(
    "--log-summary",
    default="progress_log_summary.csv",
    metavar="PATH",
    help="csv where to save per-epoch train and valid stats",
)
parser.add_argument(
    "--log-full",
    default="progress_log_full.csv",
    metavar="PATH",
    help="csv where to save per-gradient descent train stats",
)
parser.add_argument(
    "--log-output",
    type=bool,
    default=True,
    help="will log dispnet outputs and warped imgs at validation step",
)
parser.add_argument(
    "--norotate", action="store_true", help="will not apply rotation augmentation"
)
parser.add_argument(
    "--log-terminal", action="store_true", help="will display progressbar at terminal"
)
parser.add_argument(
    "-f",
    "--training-output-freq",
    type=int,
    help="frequence for outputting dispnet outputs and warped imgs at training for all scales if 0 will not output",
    metavar="N",
    default=50,
)
parser.add_argument("--l2", action="store_true", help="Use l2 loss function")

best_error = -1
n_iter = 0


def main():
    global args, best_error, n_iter  # pylint: disable=global-variable-undefined,global-variable-not-assigned
    args = parser.parse_args()
    save_path = (
        Path(args.name) / "kitti2015" / args.flownet / f"ps_{int(args.patch_size*384)}"
    )

    if args.l2:
        args.save_path = save_path / f"lr1e{int(np.log10(args.lr))}_{args.seed}_l2"
    else:
        args.save_path = save_path / f"lr1e{int(np.log10(args.lr))}_{args.seed}"
    print(f"=> will save everything to {args.save_path}")
    args.save_path.makedirs_p()

    patch_save_path = args.save_path / "patches"
    patch_save_path.makedirs_p()

    torch.manual_seed(args.seed)

    training_writer = SummaryWriter(args.save_path / "train")
    output_writer = SummaryWriter(args.save_path / "valid_attack")

    # Data loading code
    flow_loader_h, flow_loader_w = 384, 1280

    train_transform = custom_transforms.Compose(
        [
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(
                h=384 if 153 == int(args.patch_size * 384) else 256,
                w=384 if 153 == int(args.patch_size * 384) else 256,
            ),
            custom_transforms.ArrayToTensor(),
        ]
    )

    valid_transform = custom_transforms.Compose(
        [
            custom_transforms.Scale(h=flow_loader_h, w=flow_loader_w),
            custom_transforms.ArrayToTensor(),
        ]
    )

    print("=> fetching scenes in datasets/KITTI/2012_prepared")
    train_set = SequenceFolder(
        root="datasets/KITTI/2012_prepared",
        transform=train_transform,
        seed=args.seed,
        train=True,
        sequence_length=3,
    )

    if args.valset == "kitti2015":
        val_set = ValidationFlowKitti2015(
            root="datasets/KITTI/2015", transform=valid_transform
        )
    elif args.valset == "kitti2012":
        val_set = ValidationFlowKitti2012(
            root="datasets/KITTI/2012", transform=valid_transform
        )
    else:
        raise NotImplementedError(f"Valset {args.valset} is not implemented!")

    if args.DEBUG:
        train_set_len = 8  # 32
        train_set.__len__ = train_set_len
        train_set.samples = train_set.samples[:train_set_len]

    print(f"{len(train_set)} samples found in {len(train_set.scenes)} train scenes")
    print(f"{len(val_set)} samples found in valid scenes")
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=1,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # batch size is 1 since images in kitti have different sizes
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    flow_net = fetch_model(args)

    pytorch_total_params = sum(p.numel() for p in flow_net.parameters())
    print(f"Number of model paramters: {pytorch_total_params}")

    flow_net = flow_net.cuda()

    cudnn.benchmark = True
    if args.patch_type == "circle":
        patch, mask, patch_shape = init_patch_circle(args.image_size, args.patch_size)
        patch_init = patch.copy()
    elif args.patch_type == "square":
        patch, patch_shape = init_patch_square(args.image_size, args.patch_size)
        patch_init = patch.copy()
        mask = np.ones(patch_shape)
    else:
        sys.exit("Please choose a square or circle patch")

    if args.patch_path:
        patch, mask, patch_shape = init_patch_from_image(
            args.patch_path, args.mask_path, args.image_size, args.patch_size
        )
        patch_init = patch.copy()

    if args.log_terminal:
        logger = TermLogger(
            n_epochs=args.epochs,
            train_size=min(len(train_loader), args.epoch_size),
            valid_size=len(val_loader),
            attack_size=args.max_count,
        )
        logger.epoch_bar.start()
    else:
        logger = None

    for epoch in tqdm(range(args.epochs)):

        if args.log_terminal:
            logger.epoch_bar.update(epoch)
            logger.reset_train_bar()

        # train for one epoch
        patch, mask, patch_init, patch_shape = train(
            patch,
            mask,
            patch_init,
            patch_shape,
            train_loader,
            flow_net,
            logger,
            training_writer,
        )

        # Validate
        errors, error_names = validate_flow_with_gt(
            patch, mask, patch_shape, val_loader, flow_net, epoch, logger, output_writer
        )

        error_string = ", ".join(
            f"{name} : {error:.3f}" for name, error in zip(error_names, errors)
        )
        #
        if args.log_terminal:
            logger.valid_writer.write(f" * Avg {error_string}")
        else:
            print(f"Epoch {epoch} completed")

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        torch.save(patch, patch_save_path / f"epoch_{str(epoch)}")

    if args.log_terminal:
        logger.epoch_bar.finish()


def train(
    patch,
    mask,
    patch_init,
    patch_shape,
    train_loader,
    flow_net,
    logger=None,
    train_writer=None,
):
    global args, n_iter  # pylint: disable=global-variable-not-assigned
    batch_time = AverageMeter()
    data_time = AverageMeter()
    flow_net.eval()

    end = time.time()

    patch_shape_orig = patch_shape
    for i, (tgt_img, ref_img) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img_var = Variable(tgt_img.cuda())
        ref_past_img_var = Variable(ref_img[0].cuda())
        ref_future_img_var = Variable(ref_img[1].cuda())

        flow_pred_var = predict_flow(
            flow_net, ref_past_img_var, tgt_img_var, ref_future_img_var, args
        )
        data_shape = tgt_img.cpu().numpy().shape

        if args.patch_type == "circle":
            patch, mask, patch_init, rx, ry, patch_shape = circle_transform(
                patch, mask, patch_init, data_shape, patch_shape, True
            )
            # patch, mask, patch_init, rx, ry, patch_shape = circle_transform(
            #     patch, mask, patch_init, data_shape, patch_shape
            # )
        elif args.patch_type == "square":
            patch, mask, patch_init, rx, ry = square_transform(
                patch, mask, patch_init, data_shape, patch_shape, norotate=args.norotate
            )
        patch, mask = torch.FloatTensor(patch), torch.FloatTensor(mask)
        patch_init = torch.FloatTensor(patch_init)

        patch, mask = patch.cuda(), mask.cuda()
        patch_init = patch_init.cuda()
        patch_var, mask_var = Variable(patch), Variable(mask)
        patch_init_var = Variable(patch_init).cuda()

        target_var = Variable(-1 * flow_pred_var.data.clone(), requires_grad=True).cuda()
        adv_tgt_img_var, adv_ref_past_img_var, adv_ref_future_img_var, patch_var = attack(
            flow_net,
            tgt_img_var,
            ref_past_img_var,
            ref_future_img_var,
            patch_var,
            mask_var,
            patch_init_var,
            target_var=target_var,
            logger=logger,
        )

        masked_patch_var = torch.mul(mask_var, patch_var)
        patch = masked_patch_var.data.cpu().numpy()
        mask = mask_var.data.cpu().numpy()
        patch_init = patch_init_var.data.cpu().numpy()
        new_patch = np.zeros(patch_shape)
        new_mask = np.zeros(patch_shape)
        new_patch_init = np.zeros(patch_shape)
        for x in range(new_patch.shape[0]):
            for y in range(new_patch.shape[1]):
                new_patch[x][y] = patch[x][y][
                    ry : ry + patch_shape[-2], rx : rx + patch_shape[-1]
                ]
                new_mask[x][y] = mask[x][y][
                    ry : ry + patch_shape[-2], rx : rx + patch_shape[-1]
                ]
                new_patch_init[x][y] = patch_init[x][y][
                    ry : ry + patch_shape[-2], rx : rx + patch_shape[-1]
                ]

        patch = new_patch
        mask = new_mask
        patch_init = new_patch_init

        patch = zoom(
            patch,
            zoom=(
                1,
                1,
                patch_shape_orig[2] / patch_shape[2],
                patch_shape_orig[3] / patch_shape[3],
            ),
            order=1,
        )
        mask = zoom(
            mask,
            zoom=(
                1,
                1,
                patch_shape_orig[2] / patch_shape[2],
                patch_shape_orig[3] / patch_shape[3],
            ),
            order=0,
        )
        patch_init = zoom(
            patch_init,
            zoom=(
                1,
                1,
                patch_shape_orig[2] / patch_shape[2],
                patch_shape_orig[3] / patch_shape[3],
            ),
            order=1,
        )
        patch_shape = patch.shape

        if args.training_output_freq > 0 and n_iter % args.training_output_freq == 0:
            train_writer.add_image(
                "train tgt image", transpose_image(tensor2array(tgt_img[0])), n_iter
            )
            train_writer.add_image(
                "train ref past image",
                transpose_image(tensor2array(ref_img[0][0])),
                n_iter,
            )
            train_writer.add_image(
                "train ref future image",
                transpose_image(tensor2array(ref_img[1][0])),
                n_iter,
            )
            train_writer.add_image(
                "train adv tgt image",
                transpose_image(tensor2array(adv_tgt_img_var.data.cpu()[0])),
                n_iter,
            )
            if type(flow_net).__name__ == "Back2Future":
                train_writer.add_image(
                    "train adv ref past image",
                    transpose_image(tensor2array(adv_ref_past_img_var.data.cpu()[0])),
                    n_iter,
                )
            train_writer.add_image(
                "train adv ref future image",
                transpose_image(tensor2array(adv_ref_future_img_var.data.cpu()[0])),
                n_iter,
            )
            train_writer.add_image(
                "train patch",
                transpose_image(tensor2array(patch_var.data.cpu()[0])),
                n_iter,
            )
            train_writer.add_image(
                "train patch init",
                transpose_image(tensor2array(patch_init_var.data.cpu()[0])),
                n_iter,
            )
            train_writer.add_image(
                "train mask",
                transpose_image(tensor2array(mask_var.data.cpu()[0])),
                n_iter,
            )

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.log_terminal:
            logger.train_bar.update(i + 1)
        if i >= args.epoch_size - 1:
            break

        n_iter += 1

    return patch, mask, patch_init, patch_shape


def attack(
    flow_net,
    tgt_img_var,
    ref_past_img_var,  # pylint: disable=unused-argument
    ref_future_img_var,
    patch_var,
    mask_var,
    patch_init_var,
    target_var,
    logger,
):
    global args  # pylint: disable=global-variable-not-assigned
    flow_net.eval()

    adv_tgt_img_var = torch.mul((1 - mask_var), tgt_img_var) + torch.mul(
        mask_var, patch_var
    )
    adv_ref_future_img_var = torch.mul((1 - mask_var), ref_future_img_var) + torch.mul(
        mask_var, patch_var
    )

    count = 0
    loss_scalar = 1
    while loss_scalar > 0.1:
        count += 1
        adv_tgt_img_var = Variable(adv_tgt_img_var.data, requires_grad=True)
        adv_ref_past_img_var = None
        adv_ref_future_img_var = Variable(adv_ref_future_img_var.data, requires_grad=True)
        just_the_patch = Variable(patch_var.data, requires_grad=True)

        adv_flow_out_var = predict_flow(
            flow_net, adv_ref_past_img_var, adv_tgt_img_var, adv_ref_future_img_var, args
        )

        if args.l2:
            loss_data = (
                (torch.sum((adv_flow_out_var - target_var) ** 2, dim=1) + 1e-8)
                .sqrt()
                .mean()
            )
        else:
            loss_data = (
                1 - nn.functional.cosine_similarity(adv_flow_out_var, target_var)
            ).mean()

        loss_reg = nn.functional.l1_loss(
            torch.mul(mask_var, just_the_patch), torch.mul(mask_var, patch_init_var)
        )
        loss = (1 - args.alpha) * loss_data + args.alpha * loss_reg

        loss.backward()

        adv_tgt_img_grad = adv_tgt_img_var.grad.clone()
        adv_ref_future_img_grad = adv_ref_future_img_var.grad.clone()

        adv_tgt_img_var.grad.data.zero_()
        adv_ref_future_img_var.grad.data.zero_()

        patch_var -= torch.clamp(
            0.5 * args.lr * (adv_tgt_img_grad + adv_ref_future_img_grad), -2, 2
        )

        adv_tgt_img_var = torch.mul((1 - mask_var), tgt_img_var) + torch.mul(
            mask_var, patch_var
        )
        adv_ref_future_img_var = torch.mul(
            (1 - mask_var), ref_future_img_var
        ) + torch.mul(mask_var, patch_var)

        if (
            "FlowNetC" in args.flownet
            or "FlowNetS" in args.flownet
            or "FlowNet2" in args.flownet
            or "RAFT" in args.flownet
            or "PWC" in args.flownet
        ):
            adv_tgt_img_var = torch.clamp(adv_tgt_img_var, 0, 1)
            adv_ref_future_img_var = torch.clamp(adv_ref_future_img_var, 0, 1)
        else:
            adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)
            adv_ref_future_img_var = torch.clamp(adv_ref_future_img_var, -1, 1)

        loss_scalar = loss.item()

        if args.log_terminal:
            logger.attack_bar.update(count)

        if count > args.max_count - 1:
            break

    return adv_tgt_img_var, adv_ref_past_img_var, adv_ref_future_img_var, patch_var


def validate_flow_with_gt(
    patch, mask, patch_shape, val_loader, flow_net, epoch, logger, output_writer
):
    global args  # pylint: disable=global-variable-not-assigned
    batch_time = AverageMeter()
    error_names = ["epe", "adv_epe", "cos_sim", "adv_cos_sim"]
    errors = AverageMeter(i=len(error_names))

    flow_net.eval()

    end = time.time()

    with torch.no_grad():
        for i, (ref_img_past, tgt_img, ref_img_future, flow_gt, _, _, _) in enumerate(
            val_loader
        ):
            tgt_img_var = Variable(tgt_img.cuda())
            ref_img_past_var = Variable(ref_img_past.cuda())
            ref_img_future_var = Variable(ref_img_future.cuda())
            flow_gt_var = Variable(flow_gt.cuda())

            flow_fwd = predict_flow(
                flow_net, ref_img_past_var, tgt_img_var, ref_img_future_var, args
            )

            data_shape = tgt_img.cpu().numpy().shape
            if args.patch_type == "circle":
                patch_full, mask_full, _, _, _, _ = circle_transform(
                    patch, mask, patch.copy(), data_shape, patch_shape
                )
            elif args.patch_type == "square":
                patch_full, mask_full, _, _, _ = square_transform(
                    patch,
                    mask,
                    patch.copy(),
                    data_shape,
                    patch_shape,
                    norotate=args.norotate,
                )
            patch_full, mask_full = (
                torch.FloatTensor(patch_full),
                torch.FloatTensor(mask_full),
            )

            patch_full, mask_full = patch_full.cuda(), mask_full.cuda()
            patch_var, mask_var = Variable(patch_full), Variable(mask_full)

            adv_tgt_img_var = torch.mul((1 - mask_var), tgt_img_var) + torch.mul(
                mask_var, patch_var
            )
            adv_ref_img_past_var = None
            adv_ref_img_future_var = torch.mul(
                (1 - mask_var), ref_img_future_var
            ) + torch.mul(mask_var, patch_var)

            if (
                "FlowNetC" in args.flownet
                or "FlowNetS" in args.flownet
                or "FlowNet2" in args.flownet
                or "RAFT" in args.flownet
                or "PWC" in args.flownet
            ):
                adv_tgt_img_var = torch.clamp(adv_tgt_img_var, 0, 1)
                adv_ref_img_future_var = torch.clamp(adv_ref_img_future_var, 0, 1)
            else:
                adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)
                adv_ref_img_future_var = torch.clamp(adv_ref_img_future_var, -1, 1)

            adv_flow_fwd = predict_flow(
                flow_net,
                adv_ref_img_past_var,
                adv_tgt_img_var,
                adv_ref_img_future_var,
                args,
            )

            epe = compute_epe(gt=flow_gt_var, pred=flow_fwd)
            adv_epe = compute_epe(gt=flow_gt_var, pred=adv_flow_fwd)
            cos_sim = compute_cossim(flow_gt_var, flow_fwd)
            adv_cos_sim = compute_cossim(flow_gt_var, adv_flow_fwd)

            errors.update([epe, adv_epe, cos_sim, adv_cos_sim])

            if args.log_output and i % 100 == 0:
                index = int(i // 10)
                if epoch == 0:
                    output_writer.add_image(
                        "val flow Input", transpose_image(tensor2array(tgt_img[0])), 0
                    )
                    flow_to_show = flow_gt[0][:2, :, :].cpu()
                    output_writer.add_image(
                        "val target Flow",
                        transpose_image(flow_to_image(tensor2array(flow_to_show))),
                        epoch,
                    )

                val_Flow_Output = (
                    transpose_image(flow_to_image(tensor2array(flow_fwd.data[0].cpu())))
                    / 255.0
                )
                val_adv_Flow_Output = (
                    transpose_image(
                        flow_to_image(tensor2array(adv_flow_fwd.data[0].cpu()))
                    )
                    / 255.0
                )
                val_Diff_Flow_Output = (
                    transpose_image(
                        flow_to_image(
                            tensor2array((adv_flow_fwd - flow_fwd).data[0].cpu())
                        )
                    )
                    / 255.0
                )
                val_adv_tgt_image = transpose_image(
                    tensor2array(adv_tgt_img_var.data.cpu()[0])
                )
                val_adv_ref_future_image = transpose_image(
                    tensor2array(adv_ref_img_future_var.data.cpu()[0])
                )
                # val_patch = transpose_image(tensor2array(patch_var.data.cpu()[0]))

                if "FlowNetS" in args.flownet:
                    val_Flow_Output = zoom(
                        val_Flow_Output,
                        zoom=(
                            1,
                            val_adv_tgt_image.shape[1] / val_Flow_Output.shape[1],
                            val_adv_tgt_image.shape[2] / val_Flow_Output.shape[2],
                        ),
                        order=1,
                    )
                    val_adv_Flow_Output = zoom(
                        val_adv_Flow_Output,
                        zoom=(
                            1,
                            val_adv_tgt_image.shape[1] / val_adv_Flow_Output.shape[1],
                            val_adv_tgt_image.shape[2] / val_adv_Flow_Output.shape[2],
                        ),
                        order=1,
                    )
                    val_Diff_Flow_Output = zoom(
                        val_Diff_Flow_Output,
                        zoom=(
                            1,
                            val_adv_tgt_image.shape[1] / val_Diff_Flow_Output.shape[1],
                            val_adv_tgt_image.shape[2] / val_Diff_Flow_Output.shape[2],
                        ),
                        order=1,
                    )

                val_output_viz = np.hstack(
                    (
                        val_Flow_Output,
                        val_adv_Flow_Output,
                        val_Diff_Flow_Output,
                        val_adv_tgt_image,
                        val_adv_ref_future_image,
                    )
                )
                output_writer.add_image(f"val Output viz {index}", val_output_viz, epoch)

            if args.log_terminal:
                logger.valid_bar.update(i)

            batch_time.update(time.time() - end)
            end = time.time()

    return errors.avg, error_names


if __name__ == "__main__":
    main()

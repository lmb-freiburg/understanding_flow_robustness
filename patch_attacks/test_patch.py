import argparse
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
from path import Path
from PIL import Image
from scipy.ndimage.interpolation import zoom
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from tqdm import tqdm

from dataset_utils import custom_transforms
from dataset_utils.utils import get_evaluation_set
from flowutils.flowlib import flow_to_image
from models.utils_model import fetch_model, get_flownet_choices, predict_flow
from patch_attacks.logger import AverageMeter
from patch_attacks.losses import compute_cossim, compute_epe
from patch_attacks.utils import tensor2array, transpose_image
from patch_attacks.utils_patch import (
    circle_transform,
    circle_transform_different,
    get_patch_and_mask,
    project_patch_3d_scene,
    square_transform,
)

epsilon = 1e-8

parser = argparse.ArgumentParser(
    description="Test adversarial patch attacks on Optical Flow Networks",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--name", dest="name", default="", required=True, help="Save path")
parser.add_argument("--instance", dest="instance", default="", help="Specific instance")
parser.add_argument("--patch_name", dest="patch_name", default="", help="path to patches")
parser.add_argument(
    "--random_patch",
    dest="random_patch",
    default="",
    help="Random patch",
    choices=["", "gaussian", "uniform", "black", "white", "gray", "self"],
)
parser.add_argument("--patch_size", dest="patch_size", type=int, help="Size of the patch")
parser.add_argument(
    "--self_correlated_patch",
    dest="self_correlated_patch",
    default="",
    help="Self-correlated patch",
    choices=["vstripes", "hstripes", "checkered", "sin", "circle"],
)
parser.add_argument("--HOMOGENUOUS", action="store_true", help="Homogenuous Image")
parser.add_argument(
    "--different_pos",
    action="store_true",
    help="Place patch at slightly different position and using different affine transformations",
)
parser.add_argument(
    "--compression",
    dest="compression",
    default=0.0,
    type=float,
    help="Test whole image attack",
)
parser.add_argument(
    "--example", dest="example", default=0, type=int, help="Test whole image attack"
)
parser.add_argument(
    "--fixed_loc_x",
    dest="fixed_loc_x",
    default=-1,
    type=int,
    help="Test whole image attack",
)
parser.add_argument(
    "--fixed_loc_y",
    dest="fixed_loc_y",
    default=-1,
    type=int,
    help="Test whole image attack",
)
parser.add_argument("--mask_path", dest="mask_path", default="", help="path to dataset")
parser.add_argument(
    "--ignore_mask_flow", action="store_true", help="ignore flow in mask region"
)
parser.add_argument(
    "--valset",
    dest="valset",
    type=str,
    default="kitti2015",
    choices=["kitti2015", "kitti2012", "sintel"],
    help="Optical flow validation dataset",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers",
)
# parser.add_argument('-b', '--batch_size', default=1, type=int,
#                     metavar='N', help='mini-batch size')
parser.add_argument(
    "--flownet",
    dest="flownet",
    type=str,
    default="FlowNetC",
    choices=get_flownet_choices(),
    help="flow network architecture. Options: FlowNetS | SpyNet",
)
# parser.add_argument('--image_size', type=int, default=384, help='the min(height, width) of the input image to network')
parser.add_argument(
    "--patch_type", type=str, default="circle", help="patch type: circle or square"
)
parser.add_argument(
    "--norotate", action="store_true", help="will display progressbar at terminal"
)
parser.add_argument(
    "--true_motion",
    action="store_true",
    help="use the true motion according to static scene if intrinsics and depth are available",
)


def main():
    global args  # pylint: disable=global-variable-undefined
    args = parser.parse_args()
    assert not (args.random_patch and args.self_correlated_patch), "Cannot be both"
    if args.random_patch or args.self_correlated_patch:
        assert args.patch_size is not None, "Missing patch size"

    save_path = Path(args.name) / args.valset
    if args.random_patch:
        args.save_path = (
            save_path
            / args.flownet
            / "random_patch"
            / args.random_patch
            / f"ps_{args.patch_size}"
        )
    elif args.self_correlated_patch:
        args.save_path = (
            save_path
            / args.flownet
            / "self_correlated_patch"
            / args.self_correlated_patch
            / f"ps_{args.patch_size}"
        )
    else:
        args.save_path = save_path / args.flownet / args.instance

    if not args.random_patch and not args.self_correlated_patch:
        assert args.patch_name != "", "Missing patch name"
        patch_path = args.save_path / "patches" / args.patch_name
        args.patch_path = patch_path

    if args.HOMOGENUOUS:
        args.save_path = args.save_path / "homogenuous_img"

    print(f"=> will save everything to {args.save_path}")
    args.save_path.makedirs_p()
    if args.HOMOGENUOUS:
        output_vis_dir = args.save_path
    else:
        if args.different_pos:
            output_vis_dir = args.save_path / "images_test_diff_pos"
        elif args.true_motion:
            output_vis_dir = args.save_path / "images_test_true_motion"
        else:
            output_vis_dir = args.save_path / "images_test"
    output_vis_dir.makedirs_p()

    args.batch_size = 1

    if args.different_pos:
        output_writer = SummaryWriter(args.save_path / "valid_test_diff_pos")
    elif args.true_motion:
        output_writer = SummaryWriter(args.save_path / "valid_test_true_motion")
    else:
        output_writer = SummaryWriter(args.save_path / "valid_test")

    # Data loading code
    flow_loader_h, flow_loader_w = 384, 1280
    val_set = get_evaluation_set(flow_loader_h, flow_loader_w, args)
    print(f"{len(val_set)} samples found in valid scenes")

    # batch size is 1 since images in kitti have different sizes
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    if args.different_pos:
        result_file = os.path.join(args.save_path, "test_results_diff_pos.csv")
        result_scene_file = os.path.join(
            args.save_path, "test_result_scenes_diff_pos.csv"
        )
    elif args.true_motion:
        result_file = os.path.join(args.save_path, "test_results_true_motion.csv")
        result_scene_file = os.path.join(
            args.save_path, "test_result_scenes_true_motion.csv"
        )
    else:
        result_file = os.path.join(args.save_path, "test_results.csv")
        result_scene_file = os.path.join(args.save_path, "test_result_scenes.csv")

    # create model
    print("=> fetching model")
    flow_net = fetch_model(args)

    flow_net = flow_net.cuda()

    cudnn.benchmark = True
    patch, patch_shape, mask = get_patch_and_mask(args)

    error_names = ["epe", "adv_epe", "cos_sim", "adv_cos_sim"]
    errors = AverageMeter(i=len(error_names))

    # header
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("{:>10}, {:>10}, {:>10}, {:>10}\n".format(*error_names))
    with open(result_scene_file, "w", encoding="utf-8") as f:
        f.write(
            "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n".format(*(["scene"] + error_names))
        )

    flow_net.eval()

    # set seed for reproductivity
    np.random.seed(1337)

    with torch.no_grad():
        for (
            i,
            (
                ref_img_past,
                tgt_img,
                ref_img,
                flow_gt,
                disp_gt,
                calib,
                poses,
            ),
        ) in enumerate(tqdm(val_loader)):
            if args.HOMOGENUOUS:
                ref_img_past = torch.from_numpy(np.ones_like(ref_img_past)) * 0.5
                tgt_img = torch.from_numpy(np.ones_like(tgt_img)) * 0.5
                ref_img = torch.from_numpy(np.ones_like(ref_img)) * 0.5
                flow_gt = torch.from_numpy(np.zeros_like(flow_gt))

            tgt_img_var = Variable(tgt_img.cuda())
            ref_past_img_var = Variable(ref_img_past.cuda())
            ref_img_var = Variable(ref_img.cuda())
            flow_gt_var = Variable(flow_gt.cuda())

            flow_fwd = predict_flow(
                flow_net, ref_past_img_var, tgt_img_var, ref_img_var, args
            )

            data_shape = tgt_img.cpu().numpy().shape

            margin = 0
            if len(calib) > 0:
                margin = int(disp_gt.max())

            random_x = args.fixed_loc_x
            random_y = args.fixed_loc_y

            if args.different_pos:
                (
                    patch_full,
                    mask_full,
                    flow_full,
                    _,
                    random_x,
                    random_y,
                    _,
                ) = circle_transform_different(
                    patch,
                    mask,
                    patch.copy(),
                    data_shape,
                    patch_shape,
                    margin,
                    norotate=args.norotate,
                    fixed_loc=(random_x, random_y),
                )
                patch_full_tgt, patch_full_ref = patch_full
                mask_full_tgt, mask_full_ref = mask_full
                (
                    patch_full_tgt,
                    patch_full_ref,
                    mask_full_tgt,
                    mask_full_ref,
                    flow_full,
                ) = (
                    torch.FloatTensor(patch_full_tgt),
                    torch.FloatTensor(patch_full_ref),
                    torch.FloatTensor(mask_full_tgt),
                    torch.FloatTensor(mask_full_ref),
                    torch.FloatTensor(flow_full),
                )
                patch_full_tgt, patch_full_ref, mask_full_tgt, mask_full_ref = (
                    patch_full_tgt.cuda(),
                    patch_full_ref.cuda(),
                    mask_full_tgt.cuda(),
                    mask_full_ref.cuda(),
                )
                patch_var_tgt, patch_var_ref, mask_var_tgt, mask_var_ref = (
                    Variable(patch_full_tgt),
                    Variable(patch_full_ref),
                    Variable(mask_full_tgt),
                    Variable(mask_full_ref),
                )

                patch_var_future = patch_var_ref
                patch_var = patch_var_past = patch_var_tgt
                mask_var_future = mask_var_ref
                mask_var_past = mask_var = mask_var_tgt
            else:
                if args.patch_type == "circle":
                    patch_full, mask_full, _, random_x, random_y, _ = circle_transform(
                        patch,
                        mask,
                        patch.copy(),
                        data_shape,
                        patch_shape,
                        margin,
                        norotate=args.norotate,
                        fixed_loc=(random_x, random_y),
                    )
                elif args.patch_type == "square":
                    patch_full, mask_full, _, _, _ = square_transform(
                        patch,
                        mask,
                        patch.clone(),
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

                patch_var_future = patch_var_past = patch_var
                mask_var_future = mask_var_past = mask_var

            # adversarial flow
            bt, _, h_gt, w_gt = flow_gt_var.shape
            forward_patch_flow = Variable(
                torch.cat(
                    (torch.zeros((bt, 2, h_gt, w_gt)), torch.ones((bt, 1, h_gt, w_gt))), 1
                ).cuda()
            )

            # project patch into 3D scene
            if len(calib) > 0 and not args.HOMOGENUOUS and not args.different_pos:
                (
                    patch_var_future,
                    mask_var_future,
                    patch_var_past,
                    mask_var_past,
                ) = project_patch_3d_scene(
                    calib=calib,
                    poses=poses,
                    disp_gt=disp_gt,
                    patch_var=patch_var,
                    mask_var=mask_var,
                    random_x=random_x,
                    random_y=random_y,
                    patch_shape=patch_shape,
                    flow_loader_w=flow_loader_w,
                    flow_loader_h=flow_loader_h,
                    forward_patch_flow=forward_patch_flow,
                    args=args,
                )

            adv_tgt_img_var = torch.mul((1 - mask_var), tgt_img_var) + torch.mul(
                mask_var, patch_var
            )
            adv_ref_past_img_var = torch.mul(
                (1 - mask_var_past), ref_past_img_var
            ) + torch.mul(mask_var_past, patch_var_past)
            adv_ref_img_var = torch.mul((1 - mask_var_future), ref_img_var) + torch.mul(
                mask_var_future, patch_var_future
            )

            # adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)
            # adv_ref_past_img_var = torch.clamp(adv_ref_past_img_var, -1, 1)
            # adv_ref_img_var = torch.clamp(adv_ref_img_var, -1, 1)
            adv_tgt_img_var = torch.clamp(adv_tgt_img_var, 0, 1)
            adv_ref_past_img_var = torch.clamp(adv_ref_past_img_var, 0, 1)
            adv_ref_img_var = torch.clamp(adv_ref_img_var, 0, 1)

            adv_flow_fwd = predict_flow(
                flow_net, adv_ref_past_img_var, adv_tgt_img_var, adv_ref_img_var, args
            )

            # set patch to zero flow!
            mask_var_res = nn.functional.upsample(
                mask_var, size=(h_gt, w_gt), mode="bilinear"
            )

            # Ignore patch motion if set!
            if args.ignore_mask_flow:
                forward_patch_flow = Variable(
                    torch.cat(
                        (
                            torch.zeros((bt, 2, h_gt, w_gt)),
                            torch.zeros((bt, 1, h_gt, w_gt)),
                        ),
                        1,
                    ).cuda()
                )
            if args.different_pos:
                # remove occluded pixels from computation
                mask_var_ref = nn.functional.upsample(
                    mask_var_ref, size=(h_gt, w_gt), mode="bilinear"
                )
                flow_gt_var = torch.mul((1 - mask_var_ref), flow_gt_var) + torch.mul(
                    mask_var_ref,
                    Variable(
                        torch.cat(
                            (
                                torch.zeros((bt, 2, h_gt, w_gt)),
                                torch.zeros((bt, 1, h_gt, w_gt)),
                            ),
                            1,
                        ).cuda()
                    ),
                )

                # add new optical flow information for patch displacement
                scale_y = flow_gt_var.shape[2] / flow_full.shape[2]
                scale_x = flow_gt_var.shape[3] / flow_full.shape[3]
                forward_patch_flow = torch.from_numpy(
                    zoom(flow_full, zoom=(1, 1, scale_y, scale_x), order=1)
                ).cuda()

            flow_gt_var_adv = torch.mul((1 - mask_var_res), flow_gt_var) + torch.mul(
                mask_var_res, forward_patch_flow
            )

            # import pdb; pdb.set_trace()
            epe = compute_epe(gt=flow_gt_var, pred=flow_fwd)
            adv_epe = compute_epe(gt=flow_gt_var_adv, pred=adv_flow_fwd)
            print(epe, adv_epe)
            cos_sim = compute_cossim(flow_gt_var, flow_fwd)
            adv_cos_sim = compute_cossim(flow_gt_var_adv, adv_flow_fwd)

            errors.update([epe, adv_epe, cos_sim, adv_cos_sim])

            if i % 1 == 0:
                index = i  # int(i//10)
                normalize = custom_transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                )
                if args.HOMOGENUOUS:
                    imgs = [tgt_img] + [ref_img_past] + [ref_img]
                else:
                    imgs = normalize([tgt_img] + [ref_img_past] + [ref_img])
                norm_tgt_img = imgs[0]
                # norm_ref_img_past = imgs[1]
                # norm_ref_img = imgs[2]

                adv_norm_tgt_img = normalize(
                    adv_tgt_img_var.data.cpu()
                )  # torch.mul((1-mask_cpu), norm_tgt_img) + torch.mul(mask_cpu, patch_cpu)
                # adv_norm_ref_img_past = normalize(
                #     adv_ref_past_img_var.data.cpu()
                # )  # torch.mul((1-mask_cpu), norm_ref_img_past) + torch.mul(mask_cpu, patch_cpu)
                adv_norm_ref_img = normalize(
                    adv_ref_img_var.data.cpu()
                )  # torch.mul((1-mask_cpu), norm_ref_img) + torch.mul(mask_cpu, patch_cpu)

                output_writer.add_image(
                    "val flow Input", transpose_image(tensor2array(norm_tgt_img[0])), 0
                )
                flow_to_show = flow_gt[0][:2, :, :].cpu()
                output_writer.add_image(
                    "val target Flow",
                    transpose_image(flow_to_image(tensor2array(flow_to_show))),
                    0,
                )

                # set flow to zero
                # zero_flow = Variable(torch.zeros(flow_fwd.shape).cuda(), volatile=True)
                # flow_fwd_masked = torch.mul((1-mask_var[:,:2,:,:]), flow_fwd) + torch.mul(mask_var[:,:2,:,:], zero_flow)
                flow_fwd_masked = flow_fwd

                # get ground truth flow
                val_GT_adv = flow_gt_var_adv.data[0].cpu().numpy().transpose(1, 2, 0)
                # val_GT_adv = interp_gt_flow(val_GT_adv[:,:,:2], val_GT_adv[:,:,2])
                val_GT_adv = cv2.resize(
                    val_GT_adv,
                    (flow_loader_w, flow_loader_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                val_GT_adv[:, :, 0] = val_GT_adv[:, :, 0] * (flow_loader_w / w_gt)
                val_GT_adv[:, :, 1] = val_GT_adv[:, :, 1] * (flow_loader_h / h_gt)

                # gt normalization for visualization
                u = val_GT_adv[:, :, 0]
                v = val_GT_adv[:, :, 1]
                idxUnknow = (abs(u) > 1e7) | (abs(v) > 1e7)
                u[idxUnknow] = 0
                v[idxUnknow] = 0
                rad = np.sqrt(u**2 + v**2)
                maxrad = np.max(rad)

                val_GT_adv_Output = flow_to_image(val_GT_adv, maxrad)
                val_GT_adv_Output = cv2.erode(
                    val_GT_adv_Output, np.ones((3, 3), np.uint8), iterations=1
                )  # make points thicker
                val_GT_adv_Output = transpose_image(val_GT_adv_Output) / 255.0

                val_Flow_Output = (
                    transpose_image(
                        flow_to_image(tensor2array(flow_fwd.data[0].cpu()), maxrad)
                    )
                    / 255.0
                )
                val_adv_Flow_Output = (
                    transpose_image(
                        flow_to_image(tensor2array(adv_flow_fwd.data[0].cpu()), maxrad)
                    )
                    / 255.0
                )
                val_Diff_Flow_Output = (
                    transpose_image(
                        flow_to_image(
                            tensor2array((adv_flow_fwd - flow_fwd_masked).data[0].cpu()),
                            maxrad,
                        )
                    )
                    / 255.0
                )

                # val_tgt_image = transpose_image(tensor2array(norm_tgt_img[0]))
                # val_ref_image = transpose_image(tensor2array(norm_ref_img[0]))
                val_adv_tgt_image = transpose_image(tensor2array(adv_norm_tgt_img[0]))
                # val_adv_ref_image_past = transpose_image(
                #     tensor2array(adv_norm_ref_img_past[0])
                # )
                val_adv_ref_image = transpose_image(tensor2array(adv_norm_ref_img[0]))
                # val_patch = transpose_image(tensor2array(patch_var.data.cpu()[0]))

                if "FlowNetS" in args.flownet or "FlowNet2S" in args.flownet:
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

                if args.different_pos or args.true_motion:
                    val_output_viz = np.concatenate(
                        (
                            val_adv_tgt_image,
                            val_adv_ref_image,
                            val_Flow_Output,
                            val_adv_Flow_Output,
                            val_Diff_Flow_Output,
                            val_GT_adv_Output,
                        ),
                        2,
                    )
                else:
                    val_output_viz = np.concatenate(
                        (
                            val_adv_tgt_image,
                            val_adv_ref_image,
                            val_Flow_Output,
                            val_adv_Flow_Output,
                            val_Diff_Flow_Output,
                            val_GT_adv_Output,
                        ),
                        2,
                    )
                # np.save(output_vis_dir / 'viz'+str(i).zfill(3)+'.npy', val_output_viz)
                val_output_viz_im = Image.fromarray(
                    (255 * val_output_viz.transpose(1, 2, 0)).astype("uint8")
                )
                val_output_viz_im.save(output_vis_dir / "viz" + str(i).zfill(3) + ".jpg")
                output_writer.add_image(f"val Output viz {index}", val_output_viz, 0)

                with open(result_scene_file, "a", encoding="utf-8") as f:
                    f.write(
                        "{:10d}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n".format(
                            i, epe, adv_epe, cos_sim, adv_cos_sim
                        )
                    )

            if args.HOMOGENUOUS:
                break

    print("{:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*errors.avg))
    # result_file.write("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n".format(*errors.avg))
    with open(result_file, "a", encoding="utf-8") as f:
        f.write("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n".format(*errors.avg))
    with open(result_scene_file, "a", encoding="utf-8") as f:
        f.write(
            "{:>10}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n".format(
                *(["avg"] + errors.avg)
            )
        )


if __name__ == "__main__":
    main()

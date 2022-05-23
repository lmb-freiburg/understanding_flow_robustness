import argparse
import json
import os
from copy import deepcopy

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
from models.utils_model import (
    fetch_model,
    get_feature_map_keys,
    get_flownet_choices,
    predict_flow,
    setup_hooks,
)
from patch_attacks.logger import AverageMeter
from patch_attacks.losses import compute_cossim, compute_epe
from patch_attacks.utils import tensor2array, transpose_image
from patch_attacks.utils_patch import (
    circle_transform_two_patches,
    get_patch_and_mask,
    project_patch_3d_scene,
)

epsilon = 1e-8

parser = argparse.ArgumentParser(
    description="Replace Attacked Features with Unattacked Ones",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--name", dest="name", default="", required=True, help="path to dataset"
)
parser.add_argument(
    "-fk",
    "--feature_keys",
    nargs="+",
    help="List of feature (keys) to replace",
    required=True,
)
parser.add_argument("--instance", dest="instance", default="", help="Specific instance")
parser.add_argument("--patch_name", dest="patch_name", default="", help="patch name")
parser.add_argument(
    "--random_patch",
    dest="random_patch",
    default="",
    help="Random patch",
    choices=["", "gaussian", "uniform", "gray", "white", "black"],
)
parser.add_argument("--patch_size", dest="patch_size", type=int, help="Size of the patch")
parser.add_argument(
    "--self_correlated_patch",
    dest="self_correlated_patch",
    default="",
    help="Self-correlated patch",
    choices=["vstripes", "hstripes", "checkered", "sin", "circle"],
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
    choices=["kitti2015", "kitti2012"],
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
    global activation  # pylint: disable=global-variable-not-assigned
    global args  # pylint: disable=global-variable-undefined
    args = parser.parse_args()

    assert not (args.random_patch and args.self_correlated_patch), "Cannot be both"
    if args.random_patch or args.self_correlated_patch:
        assert not args.patch_size is None, "Missing patch size"

    save_path = Path(args.name) / args.valset / args.flownet
    if args.random_patch:
        args.save_path = (
            save_path / "random_patch" / args.random_patch / f"ps_{args.patch_size}"
        )
    elif args.self_correlated_patch:
        args.save_path = (
            save_path
            / "self_correlated_patch"
            / args.self_correlated_patch
            / f"ps_{args.patch_size}"
        )
    else:
        args.save_path = save_path / args.instance
    output_dir = args.save_path / "overwrite_features"
    output_dir.makedirs_p()

    if not args.random_patch and not args.self_correlated_patch:
        assert args.patch_name != "", "Missing patch name"
        patch_path = args.save_path / "patches" / args.patch_name
        args.patch_path = patch_path

    feature_keys = args.feature_keys
    for fk in feature_keys:
        assert fk in get_feature_map_keys(args)

    # Check if already exists
    overview_json_filename = output_dir / "folder_structure.json"
    json_data = {}
    if not os.path.isfile(overview_json_filename):
        json_data = {"1": [fk for fk in feature_keys]}
        output_dir = output_dir / "1"
    else:
        with open(overview_json_filename, encoding="utf-8") as json_file:
            json_data = json.load(json_file)
        already_exists = -1
        for k, v in json_data.items():
            if set(v) == set(feature_keys):
                already_exists = int(k)
                break
        if already_exists == -1:
            json_keys = [int(k) for k in json_data.keys()]
            json_data[str(max(json_keys) + 1)] = [fk for fk in feature_keys]
            output_dir = output_dir / str(max(json_keys) + 1)
        else:
            output_dir = output_dir / str(already_exists)
    with open(overview_json_filename, "w", encoding="utf-8") as outfile:
        json.dump(json_data, outfile, indent=4)
    output_dir.makedirs_p()
    args.save_path = output_dir
    output_vis_dir = args.save_path / "images_test"
    output_vis_dir.makedirs_p()
    output_writer = SummaryWriter(args.save_path / "valid_test")

    args.batch_size = 1
    # set seed for reproductivity
    np.random.seed(1337)

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

    # create model
    print("=> fetching model")
    flow_net = fetch_model(args=args, return_feat_maps=True)

    # set hooks
    setup_hooks(flow_net, args)

    flow_net = flow_net.cuda()

    cudnn.benchmark = True
    test_patch, test_patch_shape, test_mask = get_patch_and_mask(args)
    args.self_correlated_patch = ""
    args.random_patch = "uniform"
    args.patch_size = test_patch_shape[-1]
    uniform_patch, uniform_patch_shape, uniform_mask = get_patch_and_mask(args)

    error_names = ["epe", "adv_epe", "cos_sim", "adv_cos_sim"]
    errors = AverageMeter(i=len(error_names))

    result_file = open(  # pylint: disable=consider-using-with
        os.path.join(args.save_path, "test_results.csv"), "w", encoding="utf-8"
    )
    result_scene_file = open(  # pylint: disable=consider-using-with
        os.path.join(args.save_path, "test_result_scenes.csv"), "w", encoding="utf-8"
    )

    # header
    result_file.write("{:>10}, {:>10}, {:>10}, {:>10}\n".format(*error_names))
    result_scene_file.write(
        "{:>10}, {:>10}, {:>10}, {:>10}, {:>10}\n".format(*(["scene"] + error_names))
    )

    flow_net.eval()

    with torch.no_grad():
        assert np.all(uniform_mask == test_mask)
        assert np.all(uniform_patch_shape == test_patch_shape)
        mask = uniform_mask
        patch_shape = uniform_patch_shape
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
            overwrite_feat_maps = dict()
            random_x = args.fixed_loc_x
            random_y = args.fixed_loc_y

            data_shape = tgt_img.cpu().numpy().shape

            margin = 0
            if len(calib) > 0:
                margin = int(disp_gt.max())

            if args.patch_type == "circle":
                (
                    patch_full,
                    mask_f,
                    _,
                    random_x,
                    random_y,
                    _,
                ) = circle_transform_two_patches(
                    [uniform_patch, test_patch],
                    mask,
                    [uniform_patch.copy(), test_patch.copy()],
                    data_shape,
                    patch_shape,
                    margin,
                    norotate=args.norotate,
                    fixed_loc=(random_x, random_y),
                )
            elif args.patch_type == "square":
                raise NotImplementedError

            for j, patch in enumerate(patch_full):
                tgt_img_var = Variable(tgt_img.cuda())
                ref_past_img_var = Variable(ref_img_past.cuda())
                ref_img_var = Variable(ref_img.cuda())
                flow_gt_var = Variable(flow_gt.cuda())

                patch_full, mask_full = (
                    torch.FloatTensor(patch),
                    torch.FloatTensor(mask_f),
                )

                patch_full, mask_full = patch_full.cuda(), mask_full.cuda()
                patch_var, mask_var = Variable(patch_full), Variable(mask_full)

                patch_var_future = patch_var_past = patch_var
                mask_var_future = mask_var_past = mask_var

                # adversarial flow
                bt, _, h_gt, w_gt = flow_gt_var.shape
                forward_patch_flow = Variable(
                    torch.cat(
                        (
                            torch.zeros((bt, 2, h_gt, w_gt)),
                            torch.ones((bt, 1, h_gt, w_gt)),
                        ),
                        1,
                    ).cuda()
                )

                # project patch into 3D scene
                if len(calib) > 0:
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
                adv_ref_img_var = torch.mul(
                    (1 - mask_var_future), ref_img_var
                ) + torch.mul(mask_var_future, patch_var_future)

                # adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)
                # adv_ref_past_img_var = torch.clamp(adv_ref_past_img_var, -1, 1)
                # adv_ref_img_var = torch.clamp(adv_ref_img_var, -1, 1)

                adv_tgt_img_var = torch.clamp(adv_tgt_img_var, 0, 1)
                adv_ref_past_img_var = torch.clamp(adv_ref_past_img_var, 0, 1)
                adv_ref_img_var = torch.clamp(adv_ref_img_var, 0, 1)

                if j == 0:
                    flow_fwd, feat_maps = predict_flow(
                        flow_net,
                        adv_ref_past_img_var,
                        adv_tgt_img_var,
                        adv_ref_img_var,
                        args,
                        return_feat_maps=True,
                    )
                    for fk in feature_keys:
                        overwrite_feat_maps[fk] = feat_maps[fk]
                    uniform_tgt_img = deepcopy(adv_tgt_img_var.detach().cpu())
                    uniform_ref_img_past = deepcopy(adv_ref_past_img_var.detach().cpu())
                    uniform_ref_img = deepcopy(adv_ref_img_var.detach().cpu())
                else:
                    adv_flow_fwd, _ = predict_flow(
                        flow_net,
                        adv_ref_past_img_var,
                        adv_tgt_img_var,
                        adv_ref_img_var,
                        args,
                        return_feat_maps=True,
                        overwrite_feat_maps=overwrite_feat_maps,
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
                    ).cuda(),
                    volatile=True,
                )

            flow_gt_var_adv = torch.mul((1 - mask_var_res), flow_gt_var) + torch.mul(
                mask_var_res, forward_patch_flow
            )

            # import pdb; pdb.set_trace()
            epe = compute_epe(gt=flow_gt_var_adv, pred=flow_fwd)
            adv_epe = compute_epe(gt=flow_gt_var_adv, pred=adv_flow_fwd)
            cos_sim = compute_cossim(flow_gt_var_adv, flow_fwd)
            adv_cos_sim = compute_cossim(flow_gt_var_adv, adv_flow_fwd)

            errors.update([epe, adv_epe, cos_sim, adv_cos_sim])

            if i % 1 == 0:
                index = i  # int(i//10)
                normalize = custom_transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                )
                imgs = normalize(
                    [uniform_tgt_img] + [uniform_ref_img_past] + [uniform_ref_img]
                )
                norm_tgt_img = imgs[0]
                # norm_ref_img_past = imgs[1]
                norm_ref_img = imgs[2]

                # patch_cpu = patch_var.data[0].cpu()
                # mask_cpu = mask_var.data[0].cpu()

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
                val_ref_image = transpose_image(tensor2array(norm_ref_img[0]))
                val_adv_tgt_image = transpose_image(tensor2array(adv_norm_tgt_img[0]))
                # val_adv_ref_image_past = transpose_image(
                #     tensor2array(adv_norm_ref_img_past[0])
                # )
                val_adv_ref_image = transpose_image(tensor2array(adv_norm_ref_img[0]))
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

                # print(adv_norm_tgt_img.shape)
                # print(flow_fwd.data[0].cpu().shape)

                # if type(flow_net).__name__ == 'Back2Future':
                #     val_output_viz = np.concatenate((val_adv_ref_image_past, val_adv_tgt_image, val_adv_ref_image, val_Flow_Output, val_adv_Flow_Output, val_Diff_Flow_Output), 2)
                # else:
                # val_output_viz = np.concatenate((val_adv_tgt_image, val_adv_ref_image, val_Flow_Output, val_adv_Flow_Output, val_Diff_Flow_Output, val_GT_adv_Output), 2)

                val_output_viz = np.concatenate(
                    (
                        val_ref_image,
                        val_adv_ref_image,
                        val_Flow_Output,
                        val_adv_Flow_Output,
                        val_Diff_Flow_Output,
                        val_GT_adv_Output,
                    ),
                    2,
                )
                val_output_viz_im = Image.fromarray(
                    (255 * val_output_viz.transpose(1, 2, 0)).astype("uint8")
                )
                val_output_viz_im.save(output_vis_dir / "viz" + str(i).zfill(3) + ".jpg")
                output_writer.add_image(f"val Output viz {index}", val_output_viz, 0)

                # val_output_viz = np.vstack((val_Flow_Output, val_adv_Flow_Output, val_Diff_Flow_Output, val_adv_tgt_image, val_adv_ref_image))
                # scipy.misc.imsave('outfile.jpg', os.path.join(output_vis_dir, 'vis_{}.png'.format(index)))

                result_scene_file.write(
                    "{:10d}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n".format(
                        i, epe, adv_epe, cos_sim, adv_cos_sim
                    )
                )

    print("{:>10}, {:>10}, {:>10}, {:>10}".format(*error_names))
    print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(*errors.avg))
    result_file.write("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n".format(*errors.avg))
    result_scene_file.write(
        "{:>10}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n".format(*(["avg"] + errors.avg))
    )

    result_file.close()
    result_scene_file.close()


if __name__ == "__main__":
    main()

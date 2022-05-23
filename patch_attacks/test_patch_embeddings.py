import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import umap
from path import Path
from sklearn.manifold import TSNE
from torch.autograd import Variable
from tqdm import tqdm

from dataset_utils.utils import get_evaluation_set
from models.utils_model import (
    compute_feature_map,
    fetch_model,
    get_feature_map_keys,
    get_flownet_choices,
    predict_flow,
    setup_hooks,
)
from patch_attacks.utils_patch import (
    circle_transform,
    get_patch_and_mask,
    project_patch_3d_scene,
    square_transform,
)

epsilon = 1e-8

parser = argparse.ArgumentParser(
    description="Feature Embeddings",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--name", dest="name", default="", required=True, help="path to dataset"
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


def get_activation_from_feature_map(feature_maps: dict, key: str) -> np.ndarray:
    return [fm[key] for fm in feature_maps]


def maximum_mean_discrepancy(source: np.ndarray, target: np.ndarray):
    # Implementation based on https://github.com/ZongxianLee/MMD_Loss.Pytorch
    def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1))
        )
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [
            torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list
        ]
        return sum(kernel_val)

    source = torch.from_numpy(source)
    target = torch.from_numpy(target)
    features_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target)
    XX = kernels[:features_size, :features_size]
    YY = kernels[features_size:, features_size:]
    XY = kernels[:features_size, features_size:]
    YX = kernels[features_size:, :features_size]
    mmd = torch.mean(XX + YY - XY - YX)
    return mmd.numpy().item()


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

    if not args.random_patch and not args.self_correlated_patch:
        assert args.patch_name != "", "Missing patch name"
        patch_path = args.save_path / "patches" / args.patch_name
        args.patch_path = patch_path

    print(f"=> will save everything to {args.save_path}")
    args.save_path.makedirs_p()
    umap_save_path = args.save_path / "umap"
    umap_save_path.makedirs_p()
    tsne_save_path = args.save_path / "tsne"
    tsne_save_path.makedirs_p()

    args.batch_size = 1

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
    patch, patch_shape, mask = get_patch_and_mask(args)

    flow_net.eval()

    # set seed for reproductivity
    np.random.seed(1337)

    without_adversarial_patch = list()
    with_adversarial_patch = list()
    with torch.no_grad():
        for (
            _,
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
            tgt_img_var = Variable(tgt_img.cuda())
            ref_past_img_var = Variable(ref_img_past.cuda())
            ref_img_var = Variable(ref_img.cuda())
            flow_gt_var = Variable(flow_gt.cuda())

            _, wo_feature_maps = predict_flow(
                flow_net,
                ref_past_img_var,
                tgt_img_var,
                ref_img_var,
                args,
                return_feat_maps=True,
            )
            for fm in wo_feature_maps.keys():
                wo_feature_maps[fm] = compute_feature_map(wo_feature_maps[fm], "mean")

            without_adversarial_patch.append(wo_feature_maps)

            data_shape = tgt_img.cpu().numpy().shape

            margin = 0
            if len(calib) > 0:
                margin = int(disp_gt.max())

            random_x = args.fixed_loc_x
            random_y = args.fixed_loc_y

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

            # adverserial flow
            bt, _, h_gt, w_gt = flow_gt_var.shape
            forward_patch_flow = Variable(
                torch.cat(
                    (torch.zeros((bt, 2, h_gt, w_gt)), torch.ones((bt, 1, h_gt, w_gt))),
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
            adv_ref_img_var = torch.mul((1 - mask_var_future), ref_img_var) + torch.mul(
                mask_var_future, patch_var_future
            )

            # adv_tgt_img_var = torch.clamp(adv_tgt_img_var, -1, 1)
            # adv_ref_past_img_var = torch.clamp(adv_ref_past_img_var, -1, 1)
            # adv_ref_img_var = torch.clamp(adv_ref_img_var, -1, 1)

            adv_tgt_img_var = torch.clamp(adv_tgt_img_var, 0, 1)
            adv_ref_past_img_var = torch.clamp(adv_ref_past_img_var, 0, 1)
            adv_ref_img_var = torch.clamp(adv_ref_img_var, 0, 1)

            _, w_feature_maps = predict_flow(
                flow_net,
                adv_ref_past_img_var,
                adv_tgt_img_var,
                adv_ref_img_var,
                args,
                return_feat_maps=True,
            )
            for fm in w_feature_maps.keys():
                w_feature_maps[fm] = compute_feature_map(w_feature_maps[fm], "mean")

            with_adversarial_patch.append(w_feature_maps)

    mmd_dict = dict()
    vis_features = get_feature_map_keys(args)

    for layer_counter, act in enumerate(vis_features):
        print(act)
        wo_feature_maps = get_activation_from_feature_map(without_adversarial_patch, act)
        w_feature_maps = get_activation_from_feature_map(with_adversarial_patch, act)

        wo_feature_maps = np.array(wo_feature_maps)
        w_feature_maps = np.array(w_feature_maps)

        mmd = maximum_mean_discrepancy(w_feature_maps, wo_feature_maps)
        if np.isnan(mmd):
            mmd = 0
        mmd_dict[act] = mmd

        all_data = np.concatenate([w_feature_maps, wo_feature_maps], axis=0)

        # UMAP magic
        umap_embedding = umap.UMAP().fit_transform(all_data)
        # umap.plot.points(umap_embedding, labels=targets)
        nof_samples = w_feature_maps.shape[0]
        plt.scatter(
            umap_embedding[:nof_samples, 0],
            umap_embedding[:nof_samples, 1],
            c="red",
            label="With patch",
            s=6,
        )
        plt.scatter(
            umap_embedding[nof_samples:, 0],
            umap_embedding[nof_samples:, 1],
            c="blue",
            label="Without patch",
            s=6,
        )
        plt.legend()
        plt.gca().set_aspect("equal", "datalim")
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("UMAP dimension 0")
        plt.ylabel("UMAP dimension 1")
        plt.tight_layout()
        plt.savefig(umap_save_path / f"{layer_counter}_{act}.pdf")
        plt.close()

        # t-SNE magic
        tsne_embedding = TSNE(n_components=2).fit_transform(all_data)
        nof_samples = w_feature_maps.shape[0]
        plt.scatter(
            tsne_embedding[:nof_samples, 0],
            tsne_embedding[:nof_samples, 1],
            c="red",
            label="With patch",
            s=6,
        )
        plt.scatter(
            tsne_embedding[nof_samples:, 0],
            tsne_embedding[nof_samples:, 1],
            c="blue",
            label="Without patch",
            s=6,
        )
        # plt.legend()
        plt.gca().set_aspect("equal", "datalim")
        plt.gcf().patch.set_visible(False)
        plt.axis("off")
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(
            tsne_save_path / f"{layer_counter}_{act}.pdf",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()

    with open(args.save_path / "embedding_mmds.json", "w", encoding="utf-8") as fp:
        json.dump(mmd_dict, fp, indent=4)


if __name__ == "__main__":
    main()

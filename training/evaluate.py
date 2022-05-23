import argparse
import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
from torch import nn

import dataset_utils.datasets as datasets
from dataset_utils import data_utils
from models.raft.raft import RAFT
from models.raft.utils.utils import InputPadder, forward_interpolate


def tensor2array(tensor, max_value=255, colormap: str = "rainbow"):
    if max_value is None:
        max_value = tensor.max()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            if cv2.__version__.startswith("3") or cv2.__version__.startswith("4"):
                color_cvt = cv2.COLOR_BGR2RGB
            else:  # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            if colormap == "rainbow":
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == "bone":
                colormap = cv2.COLORMAP_BONE
            array = (
                (255 * tensor.squeeze().numpy() / max_value).clip(0, 255).astype(np.uint8)
            )
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32) / 255
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            array = (
                tensor.expand(tensor.size(0), tensor.size(1), 3).numpy() / max_value
            ).clip(0, 1)

    elif tensor.ndimension() == 3:
        if tensor.size(0) == 3:
            if tensor.min() >= 0 and tensor.max() <= 1:
                array = tensor.numpy().transpose(1, 2, 0)
            else:
                array = 0.5 + tensor.numpy().transpose(1, 2, 0) * 0.5
        elif tensor.size(0) == 2:
            array = tensor.numpy().transpose(1, 2, 0)
    return array


def transpose_image(array):
    return array.transpose(2, 0, 1)


def flow_to_image(flow, maxr=-1):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """

    def make_color_wheel():
        """
        Generate color wheel according Middlebury color code
        :return: Color wheel
        """
        RY = 15
        YG = 6
        GC = 4
        CB = 11
        BM = 13
        MR = 6

        ncols = RY + YG + GC + CB + BM + MR

        colorwheel = np.zeros([ncols, 3])

        col = 0

        # RY
        colorwheel[0:RY, 0] = 255
        colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
        col += RY

        # YG
        colorwheel[col : col + YG, 0] = 255 - np.transpose(
            np.floor(255 * np.arange(0, YG) / YG)
        )
        colorwheel[col : col + YG, 1] = 255
        col += YG

        # GC
        colorwheel[col : col + GC, 1] = 255
        colorwheel[col : col + GC, 2] = np.transpose(
            np.floor(255 * np.arange(0, GC) / GC)
        )
        col += GC

        # CB
        colorwheel[col : col + CB, 1] = 255 - np.transpose(
            np.floor(255 * np.arange(0, CB) / CB)
        )
        colorwheel[col : col + CB, 2] = 255
        col += CB

        # BM
        colorwheel[col : col + BM, 2] = 255
        colorwheel[col : col + BM, 0] = np.transpose(
            np.floor(255 * np.arange(0, BM) / BM)
        )
        col += +BM

        # MR
        colorwheel[col : col + MR, 2] = 255 - np.transpose(
            np.floor(255 * np.arange(0, MR) / MR)
        )
        colorwheel[col : col + MR, 0] = 255

        return colorwheel

    def compute_color(u, v):
        """
        compute optical flow color map
        :param u: optical flow horizontal map
        :param v: optical flow vertical map
        :return: optical flow in color code
        """
        [h, w] = u.shape
        img = np.zeros([h, w, 3])
        nanIdx = np.isnan(u) | np.isnan(v)
        u[nanIdx] = 0
        v[nanIdx] = 0

        colorwheel = make_color_wheel()
        ncols = np.size(colorwheel, 0)

        rad = np.sqrt(u**2 + v**2)

        a = np.arctan2(-v, -u) / np.pi

        fk = (a + 1) / 2 * (ncols - 1) + 1

        k0 = np.floor(fk).astype(int)

        k1 = k0 + 1
        k1[k1 == ncols + 1] = 1
        f = fk - k0

        for i in range(0, np.size(colorwheel, 1)):
            tmp = colorwheel[:, i]
            col0 = tmp[k0 - 1] / 255
            col1 = tmp[k1 - 1] / 255
            col = (1 - f) * col0 + f * col1

            idx = rad <= 1
            col[idx] = 1 - rad[idx] * (1 - col[idx])
            notidx = np.logical_not(idx)

            col[notidx] *= 0.75
            img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

        return img

    UNKNOWN_FLOW_THRESH = 1e7
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.0
    maxv = -999.0
    minu = 999.0
    minv = 999.0

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u**2 + v**2)
    maxrad = max(maxr, np.max(rad))

    # print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


@torch.no_grad()
def create_sintel_submission(
    model: nn.Module,
    iters: int = 32,
    warm_start: bool = False,
    output_path: str = "sintel_submission",
):
    """Create submission for the Sintel leaderboard"""
    model.eval()
    for dstype in ["clean", "final"]:
        test_dataset = datasets.MpiSintel(split="test", aug_params=None, dstype=dstype)
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=1,
        )

        flow_prev, sequence_prev = None, None
        for test_data in test_loader:
            image1, image2, (sequence, frame) = test_data
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1.cuda(), image2.cuda())

            flow_low, flow_pr = model(
                image1, image2, iters=iters, flow_init=flow_prev, test_mode=True
            )
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, "frame%04d.flo" % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            data_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(
    model: nn.Module, iters: int = 24, output_path: str = "kitti_submission"
):
    """Create submission for the Sintel leaderboard"""
    model.eval()
    test_dataset = datasets.KITTI(split="testing", aug_params=None)
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=1,
    )

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_data in test_loader:
        image1, image2, (frame_id,) = test_data
        padder = InputPadder(image1.shape, mode="kitti")
        image1, image2 = padder.pad(image1.cuda(), image2.cuda())

        _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        data_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters: int = 24, flowNetC: bool = False):
    """Perform evaluation on the FlyingChairs (test) split"""
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split="validation")
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=1,
    )
    for test_data in val_loader:
        image1, image2, flow_gt, _ = test_data
        image1 = image1.cuda()
        image2 = image2.cuda()

        if flowNetC:
            flow_pr = model(image1 / 255.0, image2 / 255.0)
        else:
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt[0]) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print(f"Validation Chairs EPE: {epe:.3f}")
    return {"chairs": epe}


@torch.no_grad()
def validate_sintel(model: nn.Module, iters: int = 32, flowNetC: bool = False):
    """Peform validation using the Sintel (train) split"""
    model.eval()
    results = {}
    for dstype in ["clean", "final"]:
        val_dataset = datasets.MpiSintel(split="training", dstype=dstype)
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=1,
            pin_memory=False,
            shuffle=False,
            num_workers=4,
            drop_last=True,
        )
        epe_list = []

        for data_blob in val_loader:
            image1, image2, flow_gt, _ = (x.cuda() for x in data_blob)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            if flowNetC:
                flow_pr = model(image1 / 255.0, image2 / 255.0)[0]
            else:
                _, flow_pr = model(image1, image2, iters=iters, test_mode=True)

            flow = padder.unpad(flow_pr[0])  # .cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).cpu().numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print(
            "Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f"
            % (dstype, epe, px1, px3, px5)
        )
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def validate_kitti(model: nn.Module, iters: int = 24, flowNetC: bool = False):
    """Peform validation using the KITTI-2015 (train) split"""
    model.eval()
    val_dataset = datasets.KITTI(split="training")
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=1,
    )

    out_list, epe_list = [], []
    for val_data in val_loader:
        image1, image2, flow_gt, valid_gt = val_data
        image1 = image1.cuda()
        image2 = image2.cuda()

        padder = InputPadder(image1.shape, mode="kitti")
        image1, image2 = padder.pad(image1, image2)

        if flowNetC:
            if image1.shape[-2] != 384:
                continue
            flow_pr = model(image1 / 255.0, image2 / 255.0)
        else:
            _, flow_pr = model(image1, image2, iters=iters, test_mode=True)

        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt[0]) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt[0] ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt[0].view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print(f"Validation KITTI: {epe:f}, {f1:f}")
    return {"kitti-epe": epe, "kitti-f1": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="restore checkpoint")
    parser.add_argument("--dataset", help="dataset for evaluation")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )
    parser.add_argument(
        "--alternate_corr",
        action="store_true",
        help="use efficent correlation implementation",
    )
    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model.cuda()
    model.eval()

    # create_sintel_submission(model.module, warm_start=True)
    # create_kitti_submission(model.module)

    with torch.no_grad():
        if args.dataset == "chairs":
            validate_chairs(model)
            # validate_chairs(model.module, flowNetC=args.flowNetC)

        elif args.dataset == "sintel":
            validate_sintel(model.module)

        elif args.dataset == "kitti":
            validate_kitti(model)

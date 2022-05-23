# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import os.path as osp

import cv2
import numpy as np
import torch

from dataset_utils.data_utils import (
    flow_read_png,
    load_as_float,
    load_disparity,
    read_paths,
)

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)


class KITTI(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        n_height,
        n_width,
        transform=None,
        finetune: bool = False,
        disparity: bool = False,
    ):
        self.root = root
        self.n_height = n_height
        self.n_width = n_width
        self.transform = transform
        self.finetune = finetune
        self.disparity = disparity

    def __getitem__(self, index):
        # Load images
        image0 = load_as_float(self.image0_paths[index])
        image1 = load_as_float(self.image0_paths[index][:-6] + "11.png")

        # Load ground truth
        if self.disparity:
            shape_resize = (self.n_height, self.n_width)
            ground_truth = load_disparity(
                self.ground_truth_paths[index], shape=shape_resize
            )
        else:
            u, v, valid = flow_read_png(self.ground_truth_paths[index])
            gtFlow = np.dstack((u, v, valid))
            ground_truth = gtFlow.transpose(2, 0, 1).squeeze()
            ground_truth = torch.from_numpy(ground_truth)
            # ground_truth = torch.nn.functional.interpolate(
            #     torch.from_numpy(ground_truth)[None, ...],
            #     (self.n_height, self.n_width),
            #     mode="area",
            # )[0, ...]
            scale_x = self.n_width / u.shape[1]
            scale_y = self.n_height / u.shape[0]
            if all((valid == 1).reshape(-1)):
                flow = np.dstack((u, v))
                flow = cv2.resize(
                    flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST
                )
                flow = flow * [scale_x, scale_y]
                gtFlow = np.dstack((flow, np.ones(flow.shape[:2])))
            else:
                flow = np.dstack((u, v))
                ht, wd = flow.shape[:2]
                coords = np.meshgrid(np.arange(wd), np.arange(ht))
                coords = np.stack(coords, axis=-1)

                coords = coords.reshape(-1, 2).astype(np.float32)
                flow = flow.reshape(-1, 2).astype(np.float32)
                valid = valid.reshape(-1).astype(np.float32)

                coords0 = coords[valid >= 1]
                flow0 = flow[valid >= 1]

                ht1 = int(round(ht * scale_y))
                wd1 = int(round(wd * scale_x))

                coords1 = coords0 * [scale_x, scale_y]
                flow1 = flow0 * [scale_x, scale_y]

                xx = np.round(coords1[:, 0]).astype(np.int32)
                yy = np.round(coords1[:, 1]).astype(np.int32)

                v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
                xx = xx[v]
                yy = yy[v]
                flow1 = flow1[v]

                flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
                valid_img = np.zeros([ht1, wd1], dtype=np.int32)

                flow_img[yy, xx] = flow1
                valid_img[yy, xx] = 1
                gtFlow = np.dstack((flow_img, valid_img))

            ground_truth_downsampled = torch.from_numpy(
                gtFlow.transpose(2, 0, 1).squeeze()
            )

        if self.transform:
            imgs = self.transform([image0] + [image1])
            image0 = imgs[0]
            image1 = imgs[1]

        if self.finetune:
            return (
                image0,
                image1,
                ground_truth_downsampled[:2],
                ground_truth_downsampled[2:][0],
            )
        return (
            image0,
            image1,
            ground_truth,
            ground_truth_downsampled[:2],
            ground_truth_downsampled[2:][0],
        )

    def __len__(self):
        return len(self.image0_paths)


class KITTI2012(KITTI):
    def __init__(
        self,
        root="datasets/KITTI/global_attacks/training",
        n_height: int = 256,
        n_width: int = 640,
        transform=None,
        finetune: bool = False,
        disparity: bool = False,
    ):
        super().__init__(
            root=root,
            n_height=n_height,
            n_width=n_width,
            transform=transform,
            finetune=finetune,
            disparity=disparity,
        )

        self.image0_paths = read_paths(osp.join(root, "kitti_stereo_flow_all_image0.txt"))
        self.image1_paths = read_paths(osp.join(root, "kitti_stereo_flow_all_image1.txt"))

        if disparity:
            self.ground_truth_paths = read_paths(
                osp.join(root, "kitti_stereo_flow_all_disparity.txt")
            )
        else:
            self.ground_truth_paths = read_paths(
                osp.join(root, "kitti_stereo_flow_all_flow.txt")
            )

        assert len(self.image0_paths) == len(self.image1_paths)


class KITTI2015(KITTI):
    def __init__(
        self,
        root="datasets/KITTI/global_attacks/training",
        n_height: int = 256,
        n_width: int = 640,
        transform=None,
        finetune: bool = False,
        disparity: bool = False,
    ):
        super().__init__(
            root=root,
            n_height=n_height,
            n_width=n_width,
            transform=transform,
            finetune=finetune,
            disparity=disparity,
        )

        self.image0_paths = read_paths(osp.join(root, "kitti_scene_flow_all_image0.txt"))
        self.image1_paths = read_paths(osp.join(root, "kitti_scene_flow_all_image1.txt"))
        if disparity:
            self.ground_truth_paths = read_paths(
                osp.join(root, "kitti_scene_flow_all_disparity.txt")
            )
        else:
            self.ground_truth_paths = read_paths(
                osp.join(root, "kitti_scene_flow_all_flow.txt")
            )

        assert len(self.image0_paths) == len(self.image1_paths)

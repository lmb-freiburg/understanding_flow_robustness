import os
import os.path as osp
from glob import glob

import cv2
import numpy as np
import torch
import torch.utils.data as data
from path import Path

from dataset_utils.data_utils import read_gen

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)


class MpiSintel(data.Dataset):
    def __init__(
        self,
        root="datasets/Sintel",
        transform=None,
        split="training",
        dstype="clean",
    ):
        self.root = Path(root)
        self.transform = transform
        self.split = split
        self.dstype = dstype

        self.flow_root = self.root / self.split / "flow"
        self.image_root = self.root / self.split / self.dstype

        self.image_list = []
        self.flow_list = []

        for scene in os.listdir(self.image_root):
            image_list = sorted(glob(osp.join(self.image_root, scene, "*.png")))
            for i in range(len(image_list) - 1):
                self.image_list += [[image_list[i], image_list[i + 1]]]
                # self.extra_info += [(scene, i)]  # scene and frame_id

            self.flow_list += sorted(glob(osp.join(self.flow_root, scene, "*.flo")))

    def __getitem__(self, index: int):
        gtFlow = read_gen(self.flow_list[index])
        tgt_img = read_gen(self.image_list[index][0])
        ref_img_future = read_gen(self.image_list[index][1])

        gtFlow = np.array(gtFlow).astype(np.float32)
        tgt_img = np.array(tgt_img).astype(np.uint8)
        ref_img_future = np.array(ref_img_future).astype(np.uint8)

        gtFlow = torch.FloatTensor(gtFlow.transpose(2, 0, 1))
        gtFlow = torch.cat(
            [gtFlow, torch.ones((1, gtFlow.shape[1], gtFlow.shape[2]))], dim=0
        )

        if self.transform is not None:
            imgs = self.transform([tgt_img] + [ref_img_future])
            tgt_img = imgs[0]
            ref_img_future = imgs[1]

        return (
            torch.zeros_like(ref_img_future),
            tgt_img,
            ref_img_future,
            gtFlow,
            {},
            {},
            {},
        )

    def __len__(self) -> int:
        return len(self.image_list)

import random

import cv2
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
from torchvision.transforms import ColorJitter

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


class FlowAugmentor:
    def __init__(
        self,
        crop_size,
        min_scale=-0.2,
        max_scale=0.5,
        do_flip=True,
        do_trans_rot=False,
        translate=10,
        rot_angle=17,
        diff_angle=5,
    ):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        self.do_trans_rot = do_trans_rot
        self.translate = (int(translate), int(translate))
        self.trans_prob = 0.7
        self.angle = rot_angle
        self.diff_angle = diff_angle
        self.rot_prob = 0.7

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14
        )
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """Photometric augmentation"""

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(
                self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8
            )
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(  # pylint: disable=dangerous-default-value
        self, img1, img2, bounds=[50, 100]
    ):
        """Occlusion augmentation"""

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0 : y0 + dy, x0 : x0 + dx, :] = mean_color

        return img1, img2

    def rotation(self, img1: np.ndarray, img2: np.ndarray, flow: np.ndarray):
        """Implementation of https://github.com/ClementPinard/FlowNetPytorch/blob/master/flow_transforms.py

        Args:
            img1 (np.ndarry): [Ref img]
            img2 (np.ndarray): [Target img]
            flow (np.ndarray): [GT Flow]
        """
        applied_angle = random.uniform(-self.angle, self.angle)
        diff = random.uniform(-self.diff_angle, self.diff_angle)
        angle1 = applied_angle - diff / 2
        angle2 = applied_angle + diff / 2
        angle1_rad = angle1 * np.pi / 180
        diff_rad = diff * np.pi / 180

        h, w, _ = flow.shape

        warped_coords = np.mgrid[:w, :h].T + flow
        warped_coords -= np.array([w / 2, h / 2])

        warped_coords_rot = np.zeros_like(flow)
        warped_coords_rot[..., 0] = (np.cos(diff_rad) - 1) * warped_coords[
            ..., 0
        ] + np.sin(diff_rad) * warped_coords[..., 1]
        warped_coords_rot[..., 1] = (
            -np.sin(diff_rad) * warped_coords[..., 0]
            + (np.cos(diff_rad) - 1) * warped_coords[..., 1]
        )

        flow += warped_coords_rot

        # Warning: this is problematic with NaNs, see https://stackoverflow.com/questions/34701206/scipy-ndimage-rotate-doesnt-work-with-np-nan-value
        img1 = ndimage.interpolation.rotate(img1, angle1, reshape=False, order=2)
        img2 = ndimage.interpolation.rotate(img2, angle2, reshape=False, order=2)
        flow = ndimage.interpolation.rotate(flow, angle1, reshape=False, order=2)
        # flow vectors must be rotated too! careful about Y flow which is upside down
        flow_ = np.copy(flow)
        flow[:, :, 0] = (
            np.cos(angle1_rad) * flow_[:, :, 0] + np.sin(angle1_rad) * flow_[:, :, 1]
        )
        flow[:, :, 1] = (
            -np.sin(angle1_rad) * flow_[:, :, 0] + np.cos(angle1_rad) * flow_[:, :, 1]
        )
        return img1, img2, flow

    def translation(self, img1, img2, flow):
        h, w = img1.shape[:2]
        th, tw = self.translate
        tw = np.random.randint(-tw, tw)
        th = np.random.randint(-th, th)
        if tw == 0 and th == 0:
            return img1, img2, flow

        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1, x2, x3, x4 = max(0, tw), min(w + tw, w), max(0, -tw), min(w - tw, w)
        y1, y2, y3, y4 = max(0, th), min(h + th, h), max(0, -th), min(h - th, h)
        img1 = img1[y1:y2, x1:x2]
        img2 = img2[y3:y4, x3:x4]
        flow = flow[y1:y2, x1:x2]
        flow[..., 0] += tw
        flow[..., 1] += tw
        return img1, img2, flow

    def spatial_transform(self, img1, img2, flow):
        if self.do_trans_rot:
            if np.random.rand() < self.trans_prob:
                img1, img2, flow = self.translation(img1, img2, flow)

            nan_exist = (
                True
                if np.any(np.isnan(img1))
                or np.any(np.isnan(img2))
                or np.any(np.isnan(flow))
                else False
            )
            if np.random.rand() < self.rot_prob and not nan_exist:
                img1, img2, flow = self.rotation(img1, img2, flow)

        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), (self.crop_size[1] + 8) / float(wd)
        )

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(
                img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            img2 = cv2.resize(
                img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            flow = cv2.resize(
                flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        # print(img1.shape, img2.shape, flow.shape)
        if (
            img1.shape[0] - self.crop_size[0] <= 0
            or img1.shape[1] - self.crop_size[1] <= 0
            or img2.shape[0] - self.crop_size[0] <= 0
            or img2.shape[1] - self.crop_size[1] <= 0
        ):
            # rescale the images if the crop is larger than the original input image
            # img form is (y,x,3)
            scale_x = self.crop_size[1] / img1.shape[1]
            scale_y = self.crop_size[0] / img1.shape[0]

            img1 = cv2.resize(
                img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            img2 = cv2.resize(
                img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            flow = cv2.resize(
                flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            flow = flow * [scale_x, scale_y]
        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

            img1 = img1[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            img2 = img2[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
            flow = flow[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        return img1, img2, flow

    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14
        )
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(
            self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8
        )
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0 : y0 + dy, x0 : x0 + dx, :] = mean_color

        return img1, img2

    @staticmethod
    def resize_sparse_flow_map(flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

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

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), (self.crop_size[1] + 1) / float(wd)
        )

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(
                img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            img2 = cv2.resize(
                img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR
            )
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < 0.5:  # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                valid = valid[:, ::-1]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        img2 = img2[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        flow = flow[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        valid = valid[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        return img1, img2, flow, valid

    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid

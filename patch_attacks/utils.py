import shutil

import numpy as np
import torch
from PIL import Image


def load_as_float(path):
    return np.array(Image.open(path)).astype(np.float32)


def imresize(arr, sz):
    height, width = sz
    return np.array(
        Image.fromarray(arr.astype("uint8")).resize(
            (width, height), resample=Image.BILINEAR
        )
    )


def tensor2array(tensor, max_value=255, colormap="rainbow"):
    if max_value is None:
        max_value = tensor.max()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2

            if cv2.__version__.startswith("3"):
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


def save_checkpoint(
    save_path,
    dispnet_state,
    exp_pose_state,
    flownet_state,
    optimizer_state,
    is_best,
    filename="checkpoint.pth.tar",
):
    file_prefixes = ["dispnet", "exp_pose", "flownet", "optimizer"]
    states = [dispnet_state, exp_pose_state, flownet_state, optimizer_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path / f"{prefix}_{filename}")

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(
                save_path / f"{prefix}_{filename}",
                save_path / f"{prefix}_model_best.pth.tar",
            )


def submatrix(arr):
    x, y = np.nonzero(arr)
    # Using the smallest and largest x and y indices of nonzero elements,
    # we can find the desired rectangular bounds.
    # And don't forget to add 1 to the top bound to avoid the fencepost problem.
    return arr[x.min() : x.max() + 1, y.min() : y.max() + 1]


# def crop_patch(patch):
#     pass


class ToSpaceBGR:
    def __init__(self, is_bgr):
        self.is_bgr = is_bgr

    def __call__(self, tensor):
        if self.is_bgr:
            new_tensor = tensor.clone()
            new_tensor[0] = tensor[2]
            new_tensor[2] = tensor[0]
            tensor = new_tensor
        return tensor


class ToRange255:
    def __init__(self, is_255):
        self.is_255 = is_255

    def __call__(self, tensor):
        if self.is_255:
            tensor.mul_(255)
        return tensor

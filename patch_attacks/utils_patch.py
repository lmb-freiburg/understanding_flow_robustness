import re

import cv2
import numpy as np
import torch
from numpy.linalg import inv
from scipy.ndimage.interpolation import rotate, zoom
from torch import nn

from patch_attacks.utils import imresize, load_as_float


def get_patch_and_mask(args):
    if args.self_correlated_patch:
        patch = create_correlated_patch(args.self_correlated_patch, args.patch_size)
    elif args.random_patch:
        patch = create_random_patch(args.random_patch, args.patch_size)
    else:
        print("Loading patch from ", args.patch_path)
        patch = torch.load(args.patch_path)

    patch_shape = patch.shape

    if args.mask_path:
        mask_image = load_as_float(args.mask_path)
        mask_image = cv2.imresize(mask_image, (patch_shape[-1], patch_shape[-2])) / 256.0
        mask = np.array([mask_image.transpose(2, 0, 1)])
    else:
        if args.patch_type == "circle":
            mask = createCircularMask(patch_shape[-2], patch_shape[-1]).astype("float32")
            mask = np.array([[mask, mask, mask]])
        elif args.patch_type == "square":
            mask = np.ones(patch_shape)

    return patch, patch_shape, mask


def create_random_patch(patch_type: str, patch_size: int):
    if patch_type == "gaussian":
        mean = 0.5
        var = 0.5
        sigma = var**0.5
        patch = np.random.normal(mean, sigma, (patch_size, patch_size, 3))
    elif patch_type == "uniform":
        patch = np.random.uniform(0, 1, (patch_size, patch_size, 3))
    elif patch_type == "black":
        patch = np.zeros((patch_size, patch_size, 3))
    elif patch_type == "white":
        patch = np.ones((patch_size, patch_size, 3))
    elif patch_type == "red":
        patch = np.zeros((patch_size, patch_size, 3))
        patch[..., 0] = 1
    elif patch_type == "gray":
        patch = 0.5 * np.ones((patch_size, patch_size, 3))

    patch = np.transpose(patch, (2, 0, 1))
    patch = patch[np.newaxis, ...]

    return patch


def get_self_correlated_patches():
    return [
        "hstripes",
        "vstripes",
        "vstripes_greenWhite",
        "vstripes_redBlack",
        "vstripes_redBlue",
        "vstripes_greenViolett",
        "vstripes_violettOrange",
        "checkered",
        "sin",
        "circle",
    ]


def create_correlated_patch(patch_type: str, patch_size: int):
    # assert patch_type in get_self_correlated_patches()

    if patch_type == "hstripes":
        patch = np.zeros((patch_size, patch_size, 3))
        strip_thickness = 2
        for strip in range(0, patch_size, 2 * strip_thickness):
            patch[strip : strip + strip_thickness, :, :] = 1

    elif "vstripes_greenWhite" in patch_type:
        patch = np.ones((patch_size, patch_size, 3))
        strip_thickness = 2
        for strip in range(0, patch_size, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, 0] = 0
            patch[:, strip : strip + strip_thickness, 2] = 0

    elif "vstripes_redBlack" in patch_type:
        patch = np.zeros((patch_size, patch_size, 3))
        strip_thickness = 2
        for strip in range(0, patch_size, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, 0] = 1

    elif "vstripes_redBlue" in patch_type:
        patch = np.zeros((patch_size, patch_size, 3))
        strip_thickness = 2
        for strip in range(0, patch_size, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, 0] = 1
        for strip in range(strip_thickness, patch_size, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, 2] = 1

    elif "vstripes_violettOrange" in patch_type:
        patch = np.zeros((patch_size, patch_size, 3))
        strip_thickness = 2
        for strip in range(0, patch_size, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, 0] = 0.9
            patch[:, strip : strip + strip_thickness, 1] = 0.7
            patch[:, strip : strip + strip_thickness, 2] = 0.3
        for strip in range(strip_thickness, patch_size, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, 0] = 0.8
            patch[:, strip : strip + strip_thickness, 1] = 0.1
            patch[:, strip : strip + strip_thickness, 2] = 0.8

    elif "vstripes_greenViolett" in patch_type:
        patch = np.zeros((patch_size, patch_size, 3))
        strip_thickness = 2
        for strip in range(0, patch_size, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, 0] = 0.7
            patch[:, strip : strip + strip_thickness, 1] = 0.8
            patch[:, strip : strip + strip_thickness, 2] = 0.1
        for strip in range(strip_thickness, patch_size, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, 0] = 0.6
            patch[:, strip : strip + strip_thickness, 1] = 0.0
            patch[:, strip : strip + strip_thickness, 2] = 0.6

    elif "vstripes_strip" in patch_type:
        patch = np.zeros((patch_size, patch_size, 3))
        # match = re.search('strip([1-9])', patch_type)
        # strip_thickness = int(match.group(0)[5:])
        strip_thickness = int(re.findall(r"\d+", patch_type)[0])
        for strip in range(0, patch_size, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, :] = 1

    elif "vstripes_rot" in patch_type:
        patch = np.zeros((patch_size * 2, patch_size * 2, 3))
        strip_thickness = 2
        for strip in range(0, patch_size * 2, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, :] = 1

        angle = int(re.findall(r"\d+", patch_type)[0])
        patch = rotate(patch, angle=angle, reshape=False, order=1)
        patch = patch[
            patch_size * 2 // 4 : patch_size * 2 // 4 + patch_size,
            patch_size * 2 // 4 : patch_size * 2 // 4 + patch_size,
            :,
        ]

    elif "vstripes_Bcol" in patch_type and "_col" in patch_type:
        col = float(re.findall(r"[-+]?\d*\.\d+|\d+", patch_type)[0])
        patch = np.ones((patch_size, patch_size, 3)) * col
        strip_thickness = 2
        col = float(re.findall(r"[-+]?\d*\.\d+|\d+", patch_type)[1])
        for strip in range(0, patch_size, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, :] = col

    elif "vstripes_Bcol" in patch_type:
        col = float(re.findall(r"\d+\.\d+", patch_type)[0])
        patch = np.ones((patch_size, patch_size, 3)) * col
        strip_thickness = 2
        for strip in range(0, patch_size, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, :] = 1

    elif "vstripes_col" in patch_type:
        patch = np.zeros((patch_size, patch_size, 3))
        strip_thickness = 2
        col = float(re.findall(r"\d+\.\d+", patch_type)[0])
        print(col)
        for strip in range(0, patch_size, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, :] = col

    elif "vstripes" in patch_type:
        # if patch_size == 102:
        #    patch = np.zeros((patch_size, patch_size, 3))
        #    strip_thickness = patch_size // 48
        #    for strip in range(0, patch_size, 2*strip_thickness):
        #        patch[:, strip:strip + strip_thickness, :] = 1
        # else:
        patch = np.zeros((patch_size, patch_size, 3))
        strip_thickness = 2
        for strip in range(0, patch_size, 2 * strip_thickness):
            patch[:, strip : strip + strip_thickness, :] = 1
    elif patch_type == "checkered":
        patch = np.ones((patch_size, patch_size, 3))
        strip_thickness = patch_size // 48
        for strip in range(0, patch_size, 2 * strip_thickness):
            patch[strip : strip + strip_thickness, :, :] = 0
            patch[:, strip : strip + strip_thickness, :] = 0
    elif patch_type == "sin":
        patch = np.zeros((patch_size, patch_size, 3))
        strip_thickness = patch_size // 24
        sin_offset = (
            5 * np.sin(2 * np.pi * np.arange(0, patch_size) / (0.25 * patch_size)) + 5
        )
        for strip in range(0, patch_size, 2 * strip_thickness):
            # i_offset = np.random.randint(0, patch_size)
            if np.random.randint(0, 2) >= 0:
                green_strip = np.random.randint(1, 2)
                patch[strip : strip + green_strip, :, :] = 1
            for i, y in enumerate(sin_offset):
                y = int(y)
                # i = (i + i_offset)%patch_size
                if y + strip > patch_size:
                    continue
                if patch_size < y + strip + strip_thickness:
                    patch[y + strip :, :, :] = 1
                    patch[y + strip :, :, :] = 0
                patch[y + strip : y + strip + strip_thickness, i, 2] = 1
                patch[y + strip : y + strip + strip_thickness, i, 0] = 0
    elif patch_type == "circle":
        patch = np.ones((patch_size, patch_size, 3)) * 255
        thickness = patch_size // 48
        counter = 0
        for radius in range(0, patch_size // 2, 2 * thickness):
            cv2.circle(
                patch,
                center=(patch_size // 2, patch_size // 2),
                radius=radius,
                color=(0, 0, 0),
                thickness=thickness,
            )
            counter += 4
        patch /= 255
    else:
        raise Exception("Self correlation type is not implemented")

    patch = np.transpose(patch, (2, 0, 1))
    patch = patch[np.newaxis, ...]
    return patch


def createCircularMask(h, w, center=None, radius=None):

    if center is None:  # use the middle of the image
        center = [int(w / 2), int(h / 2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1]) - 2

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def init_patch_circle(image_size, patch_size):
    patch, patch_shape = init_patch_square(image_size, patch_size)
    mask = createCircularMask(patch_shape[-2], patch_shape[-1]).astype("float32")
    mask = np.array([[mask, mask, mask]])
    return patch, mask, patch.shape


def circle_transform(
    patch,
    mask,
    patch_init,
    data_shape,
    patch_shape,
    margin=0,
    center=False,
    norotate=False,
    fixed_loc=(-1, -1),
    moving=False,
):
    # get dummy image
    if not moving:
        patch = patch + np.random.random() * 0.1 - 0.05
    patch = np.clip(patch, 0.0, 1.0)
    patch = patch * mask
    x = np.zeros(data_shape)
    xm = np.zeros(data_shape)
    xp = np.zeros(data_shape)

    # get shape
    image_w, image_h = data_shape[-1], data_shape[-2]

    if not moving:
        zoom_factor = 1 + 0.05 * (np.random.random() - 0.5)
        patch = zoom(patch, zoom=(1, 1, zoom_factor, zoom_factor), order=1)
        mask = zoom(mask, zoom=(1, 1, zoom_factor, zoom_factor), order=0)
        patch_init = zoom(patch_init, zoom=(1, 1, zoom_factor, zoom_factor), order=1)
    patch_shape = patch.shape
    m_size = patch.shape[-1]
    for i in range(x.shape[0]):
        # random rotation
        if not norotate:
            rot = 10 * (np.random.random() - 0.5)
            for j in range(patch[i].shape[0]):
                patch[i][j] = rotate(patch[i][j], angle=rot, reshape=False, order=1)
                patch_init[i][j] = rotate(
                    patch_init[i][j], angle=rot, reshape=False, order=1
                )

        # random location
        # random_x = 2*m_size + np.random.choice(image_w - 4*m_size -2)
        # random_x = m_size + np.random.choice(image_w - 2*m_size -2)
        if fixed_loc[0] < 0 or fixed_loc[1] < 0:
            if center:
                random_x = (image_w - m_size) // 2
            else:
                random_x = (
                    m_size
                    + margin
                    + np.random.choice(image_w - 2 * m_size - 2 * margin - 2)
                )
            assert random_x + m_size < x.shape[-1]
            # while random_x + m_size > x.shape[-1]:
            #     random_x = np.random.choice(image_w - m_size - 1)
            # random_y = m_size + np.random.choice(image_h - 2*m_size -2)
            if center:
                random_y = (image_h - m_size) // 2
            else:
                random_y = m_size + np.random.choice(image_h - 2 * m_size - 2)
            assert random_y + m_size < x.shape[-2]
        #            while random_y + m_size > x.shape[-2]:
        #                random_y = np.random.choice(image_h)
        else:
            random_x = fixed_loc[0]
            random_y = fixed_loc[1]

        # apply patch to dummy image
        x[i][0][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch[i][0]
        x[i][1][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch[i][1]
        x[i][2][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch[i][2]

        # apply mask to dummy image
        xm[i][0][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = mask[i][0]
        xm[i][1][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = mask[i][1]
        xm[i][2][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = mask[i][2]

        # apply patch_init to dummy image
        xp[i][0][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch_init[i][0]
        xp[i][1][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch_init[i][1]
        xp[i][2][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch_init[i][2]

    return x, xm, xp, random_x, random_y, patch_shape


def circle_transform_two_patches(
    patches,
    mask,
    patch_init,
    data_shape,
    patch_shape,
    margin=0,
    center=False,
    norotate=False,
    fixed_loc=(-1, -1),
):
    # get dummy image
    additive_noise = np.random.random() * 0.1 - 0.05
    for i, _ in enumerate(patches):
        patches[i] = patches[i] + additive_noise
        patches[i] = np.clip(patches[i], 0.0, 1.0)
        patches[i] = patches[i] * mask

    x1 = np.zeros(data_shape)
    x2 = np.zeros(data_shape)
    xm = np.zeros(data_shape)
    xp1 = np.zeros(data_shape)
    xp2 = np.zeros(data_shape)

    # get shape
    image_w, image_h = data_shape[-1], data_shape[-2]

    zoom_factor = 1 + 0.05 * (np.random.random() - 0.5)
    patches[0] = zoom(patches[0], zoom=(1, 1, zoom_factor, zoom_factor), order=1)
    patches[1] = zoom(patches[1], zoom=(1, 1, zoom_factor, zoom_factor), order=1)
    mask = zoom(mask, zoom=(1, 1, zoom_factor, zoom_factor), order=0)
    patch_init[0] = zoom(patch_init[0], zoom=(1, 1, zoom_factor, zoom_factor), order=1)
    patch_init[1] = zoom(patch_init[1], zoom=(1, 1, zoom_factor, zoom_factor), order=1)

    assert np.all(patches[0].shape == patches[1].shape)
    patch_shape = patches[0].shape
    m_size = patches[0].shape[-1]
    for i in range(x1.shape[0]):
        # random rotation
        if not norotate:
            rot = 10 * (np.random.random() - 0.5)
            for j in range(patches[0][i].shape[0]):
                patches[0][i][j] = rotate(
                    patches[0][i][j], angle=rot, reshape=False, order=1
                )
                patches[1][i][j] = rotate(
                    patches[1][i][j], angle=rot, reshape=False, order=1
                )
                patch_init[0][i][j] = rotate(
                    patch_init[0][i][j], angle=rot, reshape=False, order=1
                )
                patch_init[1][i][j] = rotate(
                    patch_init[1][i][j], angle=rot, reshape=False, order=1
                )

        # random location
        # random_x = 2*m_size + np.random.choice(image_w - 4*m_size -2)
        # random_x = m_size + np.random.choice(image_w - 2*m_size -2)
        if fixed_loc[0] < 0 or fixed_loc[1] < 0:
            if center:
                random_x = (image_w - m_size) // 2
            else:
                random_x = (
                    m_size
                    + margin
                    + np.random.choice(image_w - 2 * m_size - 2 * margin - 2)
                )
            assert random_x + m_size < x1.shape[-1]
            # while random_x + m_size > x.shape[-1]:
            #     random_x = np.random.choice(image_w - m_size - 1)
            # random_y = m_size + np.random.choice(image_h - 2*m_size -2)
            if center:
                random_y = (image_h - m_size) // 2
            else:
                random_y = m_size + np.random.choice(image_h - 2 * m_size - 2)
            assert random_y + m_size < x1.shape[-2]
        #            while random_y + m_size > x.shape[-2]:
        #                random_y = np.random.choice(image_h)
        else:
            random_x = fixed_loc[0]
            random_y = fixed_loc[1]

        # apply patch to dummy image
        x1[i][0][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patches[0][i][0]
        x1[i][1][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patches[0][i][1]
        x1[i][2][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patches[0][i][2]

        x2[i][0][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patches[1][i][0]
        x2[i][1][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patches[1][i][1]
        x2[i][2][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patches[1][i][2]

        # apply mask to dummy image
        xm[i][0][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = mask[i][0]
        xm[i][1][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = mask[i][1]
        xm[i][2][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = mask[i][2]

        # apply patch_init to dummy image
        xp1[i][0][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch_init[0][i][0]
        xp1[i][1][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch_init[0][i][1]
        xp1[i][2][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch_init[0][i][2]

        xp2[i][0][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch_init[1][i][0]
        xp2[i][1][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch_init[1][i][1]
        xp2[i][2][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch_init[1][i][2]

    return [x1, x2], xm, [xp1, xp2], random_x, random_y, patch_shape


def circle_transform_different(
    patch,
    mask,
    patch_init,
    data_shape,
    patch_shape,
    margin=0,
    center=False,
    norotate=False,
    fixed_loc=(-1, -1),
    moving=False,
):
    # get dummy image
    if not moving:
        patch_tgt = patch + np.random.random() * 0.1 - 0.05
    patch_tgt = np.clip(patch_tgt, 0.0, 1.0)
    patch_tgt = patch_tgt * mask
    x_tgt = np.zeros(data_shape)
    xm_tgt = np.zeros(data_shape)
    xp_tgt = np.zeros(data_shape)

    # get shape
    image_w, image_h = data_shape[-1], data_shape[-2]

    if not moving:
        zoom_factor_tgt = 1 + 0.05 * (np.random.random() - 0.5)
        patch_tgt = zoom(
            patch_tgt, zoom=(1, 1, zoom_factor_tgt, zoom_factor_tgt), order=1
        )
        mask_tgt = zoom(mask, zoom=(1, 1, zoom_factor_tgt, zoom_factor_tgt), order=0)
        patch_init_tgt = zoom(
            patch_init, zoom=(1, 1, zoom_factor_tgt, zoom_factor_tgt), order=1
        )
    patch_tgt_shape = patch_tgt.shape
    m_size = patch.shape[-1]
    for i in range(x_tgt.shape[0]):
        # random rotation
        if not norotate:
            rot_tgt = 10 * (np.random.random() - 0.5)
            for j in range(patch_tgt[i].shape[0]):
                patch_tgt[i][j] = rotate(
                    patch_tgt[i][j], angle=rot_tgt, reshape=False, order=1
                )
                patch_init_tgt[i][j] = rotate(
                    patch_init_tgt[i][j], angle=rot_tgt, reshape=False, order=1
                )

        # random location
        # random_x = 2*m_size + np.random.choice(image_w - 4*m_size -2)
        # random_x = m_size + np.random.choice(image_w - 2*m_size -2)
        if fixed_loc[0] < 0 or fixed_loc[1] < 0:
            if center:
                random_x = (image_w - m_size) // 2
            else:
                random_x = (
                    m_size
                    + margin
                    + np.random.choice(image_w - 2 * m_size - 2 * margin - 2)
                )
            assert random_x + m_size < x_tgt.shape[-1]
            # while random_x + m_size > x.shape[-1]:
            #     random_x = np.random.choice(image_w - m_size - 1)
            # random_y = m_size + np.random.choice(image_h - 2*m_size -2)
            if center:
                random_y = (image_h - m_size) // 2
            else:
                random_y = m_size + np.random.choice(image_h - 2 * m_size - 2)
            assert random_y + m_size < x_tgt.shape[-2]
        #            while random_y + m_size > x.shape[-2]:
        #                random_y = np.random.choice(image_h)
        else:
            random_x = fixed_loc[0]
            random_y = fixed_loc[1]

        # apply patch to dummy image
        x_tgt[i][0][
            random_y : random_y + patch_tgt_shape[-2],
            random_x : random_x + patch_tgt_shape[-1],
        ] = patch_tgt[i][0]
        x_tgt[i][1][
            random_y : random_y + patch_tgt_shape[-2],
            random_x : random_x + patch_tgt_shape[-1],
        ] = patch_tgt[i][1]
        x_tgt[i][2][
            random_y : random_y + patch_tgt_shape[-2],
            random_x : random_x + patch_tgt_shape[-1],
        ] = patch_tgt[i][2]

        # apply mask to dummy image
        xm_tgt[i][0][
            random_y : random_y + patch_tgt_shape[-2],
            random_x : random_x + patch_tgt_shape[-1],
        ] = mask_tgt[i][0]
        xm_tgt[i][1][
            random_y : random_y + patch_tgt_shape[-2],
            random_x : random_x + patch_tgt_shape[-1],
        ] = mask_tgt[i][1]
        xm_tgt[i][2][
            random_y : random_y + patch_tgt_shape[-2],
            random_x : random_x + patch_tgt_shape[-1],
        ] = mask_tgt[i][2]

        # apply patch_init to dummy image
        xp_tgt[i][0][
            random_y : random_y + patch_tgt_shape[-2],
            random_x : random_x + patch_tgt_shape[-1],
        ] = patch_init_tgt[i][0]
        xp_tgt[i][1][
            random_y : random_y + patch_tgt_shape[-2],
            random_x : random_x + patch_tgt_shape[-1],
        ] = patch_init_tgt[i][1]
        xp_tgt[i][2][
            random_y : random_y + patch_tgt_shape[-2],
            random_x : random_x + patch_tgt_shape[-1],
        ] = patch_init_tgt[i][2]

    flow = np.zeros_like(patch_tgt)
    flow[:, -1, ...] = 1

    # noise for patch
    if not moving:
        patch_ref = patch + np.random.random() * 0.1 - 0.05
    patch_ref = np.clip(patch_ref, 0.0, 1.0)
    patch_ref = patch_ref * mask

    if not moving:
        zoom_factor_ref = 1 + 0.05 * (np.random.random() - 0.5)
        patch_ref = zoom(
            patch_ref, zoom=(1, 1, zoom_factor_ref, zoom_factor_ref), order=1
        )
        mask_ref = zoom(mask, zoom=(1, 1, zoom_factor_ref, zoom_factor_ref), order=0)
        patch_init_ref = zoom(
            patch_init, zoom=(1, 1, zoom_factor_ref, zoom_factor_ref), order=1
        )

    patch_ref_shape = patch_ref.shape

    # rotation of patch
    if not norotate:
        rot_ref = 360 * (np.random.random() - 0.5)
        for j in range(patch_ref[i].shape[0]):
            patch_ref[i][j] = rotate(
                patch_ref[i][j], angle=rot_ref, reshape=False, order=1
            )
            patch_init_ref[i][j] = rotate(
                patch_init_ref[i][j], angle=rot_ref, reshape=False, order=1
            )

        target = flow[i, :2].transpose(1, 2, 0)
        diff_rad = rot_ref * np.pi / 180
        h, w, _ = target.shape
        warped_coords = np.mgrid[:w, :h].T + target
        warped_coords -= np.array([w / 2, h / 2])

        warped_coords_rot = np.zeros_like(target)
        warped_coords_rot[..., 0] = (np.cos(diff_rad) - 1) * warped_coords[
            ..., 0
        ] + np.sin(diff_rad) * warped_coords[..., 1]
        warped_coords_rot[..., 1] = (
            -np.sin(diff_rad) * warped_coords[..., 0]
            + (np.cos(diff_rad) - 1) * warped_coords[..., 1]
        )
        target += warped_coords_rot

        flow[i, :2] = target.transpose(2, 0, 1)

    # translation of patch
    patch_translation_u = round(100 * ((np.random.random() - 0.5) / 0.5))
    while patch_translation_u + random_x < 0:
        patch_translation_u += 1
    while patch_translation_u + random_x + patch_ref_shape[-1] > data_shape[-1]:
        patch_translation_u -= 1
    random_x_ref = random_x + patch_translation_u

    patch_translation_v = round(100 * ((np.random.random() - 0.5) / 0.5))
    while patch_translation_v + random_y < 0:
        patch_translation_v += 1
    while patch_translation_v + random_y + patch_ref_shape[-2] > data_shape[-2]:
        patch_translation_v -= 1
    random_y_ref = random_y + patch_translation_v

    flow[:, 0, ...] += patch_translation_u
    flow[:, 1, ...] += patch_translation_v
    flow[:, :2, ...] *= zoom_factor_ref / zoom_factor_tgt

    x_ref = np.zeros(data_shape)
    xm_ref = np.zeros(data_shape)
    xp_ref = np.zeros(data_shape)
    for i in range(x_ref.shape[0]):
        # apply patch to dummy image
        x_ref[i][0][
            random_y_ref : random_y_ref + patch_ref_shape[-2],
            random_x_ref : random_x_ref + patch_ref_shape[-1],
        ] = patch_ref[i][0]
        x_ref[i][1][
            random_y_ref : random_y_ref + patch_ref_shape[-2],
            random_x_ref : random_x_ref + patch_ref_shape[-1],
        ] = patch_ref[i][1]
        x_ref[i][2][
            random_y_ref : random_y_ref + patch_ref_shape[-2],
            random_x_ref : random_x_ref + patch_ref_shape[-1],
        ] = patch_ref[i][2]

        # apply mask to dummy image
        xm_ref[i][0][
            random_y_ref : random_y_ref + patch_ref_shape[-2],
            random_x_ref : random_x_ref + patch_ref_shape[-1],
        ] = mask_ref[i][0]
        xm_ref[i][1][
            random_y_ref : random_y_ref + patch_ref_shape[-2],
            random_x_ref : random_x_ref + patch_ref_shape[-1],
        ] = mask_ref[i][1]
        xm_ref[i][2][
            random_y_ref : random_y_ref + patch_ref_shape[-2],
            random_x_ref : random_x_ref + patch_ref_shape[-1],
        ] = mask_ref[i][2]

        # apply patch_init to dummy image
        xp_ref[i][0][
            random_y_ref : random_y_ref + patch_ref_shape[-2],
            random_x_ref : random_x_ref + patch_ref_shape[-1],
        ] = patch_init_ref[i][0]
        xp_ref[i][1][
            random_y_ref : random_y_ref + patch_ref_shape[-2],
            random_x_ref : random_x_ref + patch_ref_shape[-1],
        ] = patch_init_ref[i][1]
        xp_ref[i][2][
            random_y_ref : random_y_ref + patch_ref_shape[-2],
            random_x_ref : random_x_ref + patch_ref_shape[-1],
        ] = patch_init_ref[i][2]

    x = [x_tgt, x_ref]
    xm = [xm_tgt, xm_ref]
    xp = [xp_tgt, xp_ref]

    out_flow = np.zeros_like(x_tgt)
    out_flow[0][0][
        random_y : random_y + patch_tgt_shape[-2],
        random_x : random_x + patch_tgt_shape[-1],
    ] = (
        flow[0, 0, ...] * mask_tgt[0, 0, ...]
    )
    out_flow[0][1][
        random_y : random_y + patch_tgt_shape[-2],
        random_x : random_x + patch_tgt_shape[-1],
    ] = (
        flow[0, 1, ...] * mask_tgt[0, 1, ...]
    )
    out_flow[0][2][
        random_y : random_y + patch_tgt_shape[-2],
        random_x : random_x + patch_tgt_shape[-1],
    ] = (
        flow[0, 2, ...] * mask_tgt[0, 2, ...]
    )

    random_x = [random_x, random_x_ref]
    random_y = [random_y, random_y_ref]

    return x, xm, out_flow, xp, random_x, random_y, patch_shape


def init_patch_square(image_size, patch_size):
    # get mask
    # image_size = image_size**2
    noise_size = image_size * patch_size
    noise_dim = int(noise_size)  # **(0.5))
    patch = np.random.rand(1, 3, noise_dim, noise_dim)
    return patch, patch.shape


def init_patch_from_image(image_path, mask_path, image_size, patch_size):
    noise_size = np.floor(image_size * np.sqrt(patch_size))
    patch_image = load_as_float(image_path)
    patch_image = imresize(patch_image, (int(noise_size), int(noise_size))) / 128.0 - 1
    patch = np.array([patch_image.transpose(2, 0, 1)])

    mask_image = load_as_float(mask_path)
    mask_image = imresize(mask_image, (int(noise_size), int(noise_size))) / 256.0
    mask = np.array([mask_image.transpose(2, 0, 1)])
    return patch, mask, patch.shape


def square_transform(patch, mask, patch_init, data_shape, patch_shape, norotate=False):
    # get dummy image
    image_w, image_h = data_shape[-1], data_shape[-2]
    x = np.zeros(data_shape)
    xm = np.zeros(data_shape)
    xp = np.zeros(data_shape)
    # get shape
    m_size = patch_shape[-1]

    for i in range(x.shape[0]):

        # random rotation
        if not norotate:
            rot = np.random.choice(4)
            for j in range(patch[i].shape[0]):
                patch[i][j] = np.rot90(patch[i][j], rot)
                mask[i][j] = np.rot90(mask[i][j], rot)

                patch_init[i][j] = np.rot90(patch_init[i][j], rot)

        # random location
        random_x = np.random.choice(image_w - m_size - 1)
        if random_x + m_size > x.shape[-1]:
            while random_x + m_size > x.shape[-1]:
                random_x = np.random.choice(image_w)
        random_y = np.random.choice(image_h - m_size - 1)
        if random_y + m_size > x.shape[-2]:
            while random_y + m_size > x.shape[-2]:
                random_y = np.random.choice(image_h)

        # apply patch to dummy image
        x[i][0][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch[i][0]
        x[i][1][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch[i][1]
        x[i][2][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch[i][2]
        # apply mask to dummy image
        xm[i][0][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = mask[i][0]
        xm[i][1][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = mask[i][1]
        xm[i][2][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = mask[i][2]

        # apply patch_init to dummy image
        xp[i][0][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch_init[i][0]
        xp[i][1][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch_init[i][1]
        xp[i][2][
            random_y : random_y + patch_shape[-2], random_x : random_x + patch_shape[-1]
        ] = patch_init[i][2]

    # mask = np.copy(x)
    # mask[mask != 0] = 1.0

    return x, xm, xp, random_x, random_y


def project_patch_3d_scene(
    calib,
    poses,
    disp_gt,
    patch_var,
    mask_var,
    random_x,
    random_y,
    patch_shape,
    flow_loader_w,
    flow_loader_h,
    forward_patch_flow,
    args,
):
    # #################################### ONLY WORKS WITH BATCH SIZE 1 ####################################
    # imu2vel = calib["imu2vel"]["RT"][0].numpy()
    imu2cam = calib["P_imu_cam"][0].numpy()
    imu2img = calib["P_imu_img"][0].numpy()

    pose_past = poses[0][0].numpy()
    pose_ref = poses[1][0].numpy()
    # inv_pose_ref = inv(pose_ref)
    pose_fut = poses[2][0].numpy()

    # get point in IMU
    patch_disp = disp_gt[
        0,
        random_y : random_y + patch_shape[-2],
        random_x : random_x + patch_shape[-1],
    ]
    valid = patch_disp > 0
    # set to object or free space disparity
    if (
        False  # pylint: disable=condition-evals-to-constant
        and args.fixed_loc_x > 0
        and args.fixed_loc_y > 0
    ):
        # disparity = patch_disp[valid].mean() - 3  # small correction for gps errors
        disparity = patch_disp[valid].mean()
    else:
        subset = patch_disp[valid]
        min_disp = 0
        if len(subset) > 0:
            min_disp = subset.min()
        max_disp = disp_gt.max()

        disparity = np.random.uniform(min_disp, max_disp)  # disparity

    # print('Disp from ', min_disp, ' to ', max_disp)
    depth = calib["cam"]["focal_length_x"] * calib["cam"]["baseline"] / disparity
    p_cam0 = np.array([[0], [0], [0], [1]])
    p_cam0[0] = depth * (random_x - calib["cam"]["cx"]) / calib["cam"]["focal_length_x"]
    p_cam0[1] = depth * (random_y - calib["cam"]["cy"]) / calib["cam"]["focal_length_y"]
    p_cam0[2] = depth

    # transform
    T_p_cam0 = np.eye(4)
    T_p_cam0[0:4, 3:4] = p_cam0

    # transformation to generate patch points
    patch_size = -0.25
    pts = np.array(
        [
            [0, 0, 0, 1],
            [0, patch_size, 0, 1],
            [patch_size, 0, 0, 1],
            [patch_size, patch_size, 0, 1],
        ]
    ).T
    pts = inv(imu2cam).dot(T_p_cam0.dot(pts))

    # get points in reference image
    pts_src = pose_ref.dot(pts)
    pts_src = imu2img.dot(pts_src)
    pts_src = pts_src[:3, :] / pts_src[2:3, :].repeat(3, 0)

    # get points in past image
    pts_past = pose_past.dot(pts)
    pts_past = imu2img.dot(pts_past)
    pts_past = pts_past[:3, :] / pts_past[2:3, :].repeat(3, 0)

    # get points in future image
    pts_fut = pose_fut.dot(pts)
    pts_fut = imu2img.dot(pts_fut)
    pts_fut = pts_fut[:3, :] / pts_fut[2:3, :].repeat(3, 0)

    # find homography between points
    H_past, _ = cv2.findHomography(pts_src.T, pts_past.T, cv2.RANSAC)
    H_fut, _ = cv2.findHomography(pts_src.T, pts_fut.T, cv2.RANSAC)

    # import pdb; pdb.set_trace()
    refMtrx = torch.from_numpy(H_fut).float().cuda()
    refMtrx = refMtrx.repeat(args.batch_size, 1, 1)
    # get pixel origins
    X, Y = np.meshgrid(np.arange(flow_loader_w), np.arange(flow_loader_h))
    X, Y = X.flatten(), Y.flatten()
    XYhom = np.stack([X, Y, np.ones_like(X)], axis=1).T
    XYhom = np.tile(XYhom, [args.batch_size, 1, 1]).astype(np.float32)
    XYhom = torch.from_numpy(XYhom).cuda()
    XHom, YHom, _ = torch.unbind(XYhom, dim=1)
    XHom = XHom.resize_((args.batch_size, flow_loader_h, flow_loader_w))
    YHom = YHom.resize_((args.batch_size, flow_loader_h, flow_loader_w))
    # warp the canonical coordinates
    XYwarpHom = refMtrx.matmul(XYhom)
    XwarpHom, YwarpHom, ZwarpHom = torch.unbind(XYwarpHom, dim=1)
    Xwarp = (XwarpHom / (ZwarpHom + 1e-8)).resize_(
        (args.batch_size, flow_loader_h, flow_loader_w)
    )
    Ywarp = (YwarpHom / (ZwarpHom + 1e-8)).resize_(
        (args.batch_size, flow_loader_h, flow_loader_w)
    )
    # get forward flow
    u = (XHom - Xwarp).unsqueeze(1)
    v = (YHom - Ywarp).unsqueeze(1)
    flow = torch.cat((u, v), 1)
    _, _, h_gt, w_gt = forward_patch_flow.shape
    flow = nn.functional.upsample(flow, size=(h_gt, w_gt), mode="bilinear")
    flow[:, 0, :, :] = flow[:, 0, :, :] * (w_gt / flow_loader_w)
    flow[:, 1, :, :] = flow[:, 1, :, :] * (h_gt / flow_loader_h)
    forward_patch_flow[:, :2, :, :] = flow
    # get grid for resampling
    Xwarp = 2 * ((Xwarp / (flow_loader_w - 1)) - 0.5)
    Ywarp = 2 * ((Ywarp / (flow_loader_h - 1)) - 0.5)
    grid = torch.stack([Xwarp, Ywarp], dim=-1)
    # sampling with bilinear interpolation
    patch_var_future = torch.nn.functional.grid_sample(patch_var, grid, mode="bilinear")
    mask_var_future = torch.nn.functional.grid_sample(mask_var, grid, mode="bilinear")

    # use past homography
    refMtrxP = torch.from_numpy(H_past).float().cuda()
    refMtrx = refMtrx.repeat(args.batch_size, 1, 1)
    # warp the canonical coordinates
    XYwarpHomP = refMtrxP.matmul(XYhom)
    XwarpHomP, YwarpHomP, ZwarpHomP = torch.unbind(XYwarpHomP, dim=1)
    XwarpP = (XwarpHomP / (ZwarpHomP + 1e-8)).resize_(
        (args.batch_size, flow_loader_h, flow_loader_w)
    )
    YwarpP = (YwarpHomP / (ZwarpHomP + 1e-8)).resize_(
        (args.batch_size, flow_loader_h, flow_loader_w)
    )
    # get grid for resampling
    XwarpP = 2 * ((XwarpP / (flow_loader_w - 1)) - 0.5)
    YwarpP = 2 * ((YwarpP / (flow_loader_h - 1)) - 0.5)
    gridP = torch.stack([XwarpP, YwarpP], dim=-1)
    # sampling with bilinear interpolation
    patch_var_past = torch.nn.functional.grid_sample(patch_var, gridP, mode="bilinear")
    mask_var_past = torch.nn.functional.grid_sample(mask_var, gridP, mode="bilinear")

    return patch_var_future, mask_var_future, patch_var_past, mask_var_past

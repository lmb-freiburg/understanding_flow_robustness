import os
import re

import cv2
import numpy as np
from PIL import Image

try:
    import png

    has_png = True
except ImportError:
    has_png = False
    png = None

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)


def load_as_float(path):
    return np.array(Image.open(path)).astype(np.float32)


def imresize(arr, sz):
    height, width = sz
    return np.array(
        Image.fromarray(arr.astype("uint8")).resize(
            (width, height), resample=Image.BILINEAR
        )
    )


def readFlow(fn):
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print("Magic number incorrect. Invalid .flo file")
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def readPFM(file):
    with open(file, "rb") as file:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == b"PF":
            color = True
        elif header == b"Pf":
            color = False
        else:
            raise Exception("Not a PFM file.")

        dim_match = re.match(rb"^(\d+)\s(\d+)\s$", file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(file.readline().rstrip())
        if scale < 0:  # little-endian
            endian = "<"
            scale = -scale
        else:
            endian = ">"  # big-endian

        data = np.fromfile(file, endian + "f")
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
    return data


def read_gen(file_name, pil=False):  # pylint: disable=unused-argument
    ext = os.path.splitext(file_name)[-1]
    if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
        return Image.open(file_name)
    elif ext == ".bin" or ext == ".raw":
        return np.load(file_name)
    elif ext == ".flo":
        return readFlow(file_name).astype(np.float32)
    elif ext == ".pfm":
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []


def read_paths(filepath):
    """
    Reads line delimited paths from file

    Args:
        filepath : str
            path to file containing line delimited paths
    Returns:
        list : list of paths
    """
    path_list = []
    with open(filepath, encoding="utf-8") as f:
        while True:
            path = f.readline().rstrip("\n")
            # If there was nothing to read
            if path == "":
                break
            path_list.append(path)

    return path_list


def write_paths(filepath, paths):
    """
    Stores line delimited paths into file

    Args:
        filepath : str
            path to file to save paths
        paths : list
            paths to write into file
    """
    with open(filepath, "w", encoding="utf-8") as o:
        for path in paths:
            o.write(path + "\n")


def load_image(path):
    """
    Loads an RGB image

    Args:
        path : str
            path to image
        shape : list
            (H, W) of image
        normalize : bool
            if set, normalize image between 0 and 1

    Returns:
        numpy : 3 x H x W RGB image
    """
    image = load_as_float(path)
    return image


def load_disparity(path, shape=(None, None)):
    """
    Loads a disparity image

    Args:
        path : str
            path to disparity image
        shape : list
            (H, W) of disparity image
    Returns:
        numpy : H x W disparity image
    """

    # Load image and resize
    disparity = Image.open(path).convert("I")
    o_width, o_height = disparity.size

    n_height, n_width = shape

    if n_height is None or n_width is None:
        n_height = o_height
        n_width = o_width

    # Resize to dataset shape
    disparity = disparity.resize((n_width, n_height), Image.NEAREST)

    # Convert unsigned int16 to disparity values
    disparity = np.asarray(disparity, np.uint16)
    disparity = disparity / 256.0

    # Adjust disparity based on resize
    scale = np.asarray(n_width, np.float32) / np.asarray(o_width, np.float32)
    disparity = disparity * scale

    return np.asarray(disparity, np.float32)


def flow_read_png(fpath):
    """
    Read KITTI optical flow, returns u,v,valid mask

    """
    if not has_png:
        print("Error. Please install the PyPNG library")
        return

    R = png.Reader(fpath)
    width, height, data, _ = R.asDirect()
    # This only worked with python2.
    # I = np.array(map(lambda x:x,data)).reshape((height,width,3))
    I = np.array([x for x in data]).reshape((height, width, 3))
    u_ = I[:, :, 0]
    v_ = I[:, :, 1]
    valid = I[:, :, 2]

    u = (u_.astype("float64") - 2**15) / 64.0
    v = (v_.astype("float64") - 2**15) / 64.0

    return u, v, valid


def writeFlow(filename, uv, v=None):
    """Write optical flow to file.

    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert uv.ndim == 3
        assert uv.shape[2] == 2
        u = uv[:, :, 0]
        v = uv[:, :, 1]
    else:
        u = uv

    assert u.shape == v.shape
    height, width = u.shape
    with open(filename, "wb") as f:
        # write the header
        f.write(TAG_CHAR)
        np.array(width).astype(np.int32).tofile(f)
        np.array(height).astype(np.int32).tofile(f)
        # arrange into matrix form
        tmp = np.zeros((height, width * nBands))
        tmp[:, np.arange(width) * 2] = u
        tmp[:, np.arange(width) * 2 + 1] = v
        tmp.astype(np.float32).tofile(f)


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    flow = flow[:, :, ::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    return flow, valid


def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    flow = np.stack([-disp, np.zeros_like(disp)], -1)
    return flow, valid


def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])

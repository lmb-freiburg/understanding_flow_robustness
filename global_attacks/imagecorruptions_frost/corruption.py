import cv2
import numpy as np
from pkg_resources import resource_filename
from scipy.ndimage import zoom as scizoom

# /////////////// Corruption Helpers ///////////////


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=256, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert mapsize & (mapsize - 1) == 0
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
        calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[
            stepsize // 2 : mapsize : stepsize, stepsize // 2 : mapsize : stepsize
        ]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2 : mapsize : stepsize] = wibbledmean(
            ltsum
        )
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2 : mapsize : stepsize, 0:mapsize:stepsize] = wibbledmean(
            ttsum
        )

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    # clipping along the width dimension:
    ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
    top0 = (img.shape[0] - ch0) // 2

    # clipping along the height dimension:
    ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
    top1 = (img.shape[1] - ch1) // 2

    img = scizoom(
        img[top0 : top0 + ch0, top1 : top1 + ch1], (zoom_factor, zoom_factor, 1), order=1
    )

    return img


def getOptimalKernelWidth1D(radius, sigma):  # pylint: disable=unused-argument
    return radius * 2 + 1


def gauss_function(x, mean, sigma):  # pylint: disable=unused-argument
    return (np.exp(-(x**2) / (2 * (sigma**2)))) / (np.sqrt(2 * np.pi) * sigma)


def getMotionBlurKernel(width, sigma):
    k = gauss_function(np.arange(width), 0, sigma)
    Z = np.sum(k)
    return k / Z


def shift(image, dx, dy):
    if dx < 0:
        shifted = np.roll(image, shift=image.shape[1] + dx, axis=1)
        shifted[:, dx:] = shifted[:, dx - 1 : dx]
    elif dx > 0:
        shifted = np.roll(image, shift=dx, axis=1)
        shifted[:, :dx] = shifted[:, dx : dx + 1]
    else:
        shifted = image

    if dy < 0:
        shifted = np.roll(shifted, shift=image.shape[0] + dy, axis=0)
        shifted[dy:, :] = shifted[dy - 1 : dy, :]
    elif dy > 0:
        shifted = np.roll(shifted, shift=dy, axis=0)
        shifted[:dy, :] = shifted[dy : dy + 1, :]
    return shifted


# /////////////// End Corruption Helpers ///////////////


# /////////////// Corruptions ///////////////
def frost(x, severity=1, idx=None):
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]

    idx = idx if idx else np.random.randint(5)
    filename = [
        resource_filename(__name__, "./frost/frost1.png"),
        resource_filename(__name__, "./frost/frost2.png"),
        resource_filename(__name__, "./frost/frost3.png"),
        resource_filename(__name__, "./frost/frost4.jpg"),
        resource_filename(__name__, "./frost/frost5.jpg"),
        resource_filename(__name__, "./frost/frost6.jpg"),
    ][idx]
    frost = cv2.imread(filename)
    frost_shape = frost.shape
    x_shape = np.array(x).shape

    # resize the frost image so it fits to the image dimensions
    scaling_factor = 1
    if frost_shape[0] >= x_shape[0] and frost_shape[1] >= x_shape[1]:
        scaling_factor = 1
    elif frost_shape[0] < x_shape[0] and frost_shape[1] >= x_shape[1]:
        scaling_factor = x_shape[0] / frost_shape[0]
    elif frost_shape[0] >= x_shape[0] and frost_shape[1] < x_shape[1]:
        scaling_factor = x_shape[1] / frost_shape[1]
    elif (
        frost_shape[0] < x_shape[0] and frost_shape[1] < x_shape[1]
    ):  # If both dims are too small, pick the bigger scaling factor
        scaling_factor_0 = x_shape[0] / frost_shape[0]
        scaling_factor_1 = x_shape[1] / frost_shape[1]
        scaling_factor = np.maximum(scaling_factor_0, scaling_factor_1)

    scaling_factor *= 1.1
    new_shape = (
        int(np.ceil(frost_shape[1] * scaling_factor)),
        int(np.ceil(frost_shape[0] * scaling_factor)),
    )
    frost_rescaled = cv2.resize(frost, dsize=new_shape, interpolation=cv2.INTER_CUBIC)

    # randomly crop
    x_start, y_start = (
        np.random.randint(0, frost_rescaled.shape[0] - x_shape[0]),
        np.random.randint(0, frost_rescaled.shape[1] - x_shape[1]),
    )

    if len(x_shape) < 3 or x_shape[2] < 3:
        frost_rescaled = frost_rescaled[
            x_start : x_start + x_shape[0], y_start : y_start + x_shape[1]
        ]
        frost_rescaled = rgb2gray(frost_rescaled)
    else:
        frost_rescaled = frost_rescaled[
            x_start : x_start + x_shape[0], y_start : y_start + x_shape[1]
        ][..., [2, 1, 0]]
    return np.clip(c[0] * np.array(x) + c[1] * frost_rescaled, 0, 255)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# /////////////// End Corruptions ///////////////

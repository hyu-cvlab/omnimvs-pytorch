# utils.image
# 
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#
import sys
import os
import os.path as osp
import traceback
import matplotlib
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
import skimage.io
import skimage.transform
from utils.common import *
from utils.log import *

## visualize =================================

def colorMap(colormap_name: str, arr: np.ndarray,
             min_v=None, max_v=None, alpha=None) -> np.ndarray:
    arr = toNumpy(arr).astype(np.float64).squeeze()
    if colormap_name == 'oliver': return colorMapOliver(arr, min_v, max_v)
    cmap = matplotlib.cm.get_cmap(colormap_name)
    if max_v is None: max_v = np.max(arr)
    if min_v is None: min_v = np.min(arr)
    arr[arr > max_v] = max_v
    arr[arr < min_v] = min_v
    arr = (arr - min_v) / (max_v - min_v)
    if alpha is None:
        out = cmap(arr)
        out = out[:, :, 0:3]
    else:
        out = cmap(arr, alpha=alpha)
    return np.round(255 * out).astype(np.uint8)

#
# code adapted from Oliver Woodford's sc.m
_CMAP_OLIVER = np.array(
    [[0,0,0,114], [0,0,1,185], [1,0,0,114], [1,0,1,174], [0,1,0,114],
     [0,1,1,185], [1,1,0,114], [1,1,1,0]]).astype(np.float64)
#
def colorMapOliver(arr: np.ndarray, min_v=None, max_v=None) -> np.ndarray:
    arr = toNumpy(arr).astype(np.float64).squeeze()
    height, width = arr.shape
    arr = arr.reshape([1, -1])
    if max_v is None: max_v = np.max(arr)
    if min_v is None: min_v = np.min(arr)
    arr[arr < min_v] = min_v
    arr[arr > max_v] = max_v
    arr = (arr - min_v) / (max_v - min_v)
    bins = _CMAP_OLIVER[:-1, 3]
    cbins = np.cumsum(bins)
    bins = bins / cbins[-1]
    cbins = cbins[:-1] / cbins[-1]
    ind = np.sum(
        np.tile(arr, [6, 1]) > \
        np.tile(np.reshape(cbins,[-1, 1]), [1, arr.size]), axis=0)
    ind[ind > 6] = 6
    bins = 1 / bins
    cbins = np.array([0.0] + cbins.tolist())
    arr = (arr - cbins[ind]) * bins[ind]
    arr = _CMAP_OLIVER[ind, :3] * np.tile(np.reshape(1 - arr,[-1, 1]),[1,3]) + \
        _CMAP_OLIVER[ind+1, :3] * np.tile(np.reshape(arr,[-1, 1]),[1,3])
    arr[arr < 0] = 0
    arr[arr > 1] = 1
    out = np.reshape(arr, [height, width, 3])
    out = np.round(255 * out).astype(np.uint8)
    return out

## image transform =================================

def rgb2gray(I: np.ndarray, channel_wise_mean=True) -> np.ndarray:
    I = toNumpy(I)
    dtype = I.dtype
    I = I.astype(np.float64)
    if channel_wise_mean:
        return np.mean(I, axis=2).squeeze().astype(dtype)
    else:
        return np.dot(I[...,:3], [0.299, 0.587, 0.114]).astype(dtype)

def imrescale(image: np.ndarray, scale: float) -> np.ndarray:
    image = toNumpy(image)
    dtype = image.dtype
    multi_channel = True if len(image.shape) == 3 else False
    out = skimage.transform.rescale(image, scale, 
        multichannel=multi_channel, preserve_range=True)
    return out.astype(dtype)

imresize = skimage.transform.resize

def interp2D(I, grid):
    istensor = type(I) == torch.Tensor
    I = torch.tensor(I).float().squeeze().unsqueeze(0) # make 1 x C x H x W
    grid = torch.Tensor(grid).squeeze().unsqueeze(0) # make 1 x npts x 2
    if len(I.shape) < 4 : # if 1D channel image
        I = I.unsqueeze(0)
    out = F.grid_sample(I, grid, mode='bilinear', align_corners=True).squeeze()
    if not istensor: out = out.numpy()
    return out

def pixelToGrid(pts, target_resolution: (int, int), 
                source_resolution: (int, int)):
    h, w = target_resolution
    height, width = source_resolution
    xs = (pts[0,:]) / (width - 1) * 2 - 1
    ys = (pts[1,:]) / (height - 1) * 2 - 1
    xs = xs.reshape((h, w, 1))
    ys = ys.reshape((h, w, 1))
    return concat((xs, ys), 2)

def normalizeImage(image: np.ndarray, mask=None,
                   channel_wise_mean=True) -> np.ndarray:
    image = toNumpy(image)
    def __normalizeImage1D(image, mask):
        image = image.squeeze().astype(np.float32)
        if mask is not None: image[mask] = np.nan
        # normalize intensities
        image = (image - np.nanmean(image.flatten())) / \
            np.nanstd(image.flatten())
        if mask is not None: image[mask] = 0
        return image
    if len(image.shape) == 3 and image.shape[2] == 3:
        if channel_wise_mean:
            return np.concatenate(
                [__normalizeImage1D(image[:,:,i], mask)[..., np.newaxis] 
                    for i in range(3)], axis=2)
        else:
            image = image.squeeze().astype(np.float32)
            mask = np.tile(mask[..., np.newaxis], (1, 1, 3))
            if mask is not None: image[mask] = np.nan
            # normalize intensities
            image = (image - np.nanmean(image.flatten())) / \
                np.nanstd(image.flatten())
            if mask is not None: image[mask] = 0
            return image
    else:
        return __normalizeImage1D(image, mask)

## image file I/O =================================

def writeImageFloat(image: np.ndarray, tiff_path: str, thumbnail = None):
    image = toNumpy(image)
    with tifffile.TiffWriter(tiff_path) as tiff:
        if thumbnail is not None:
            if not thumbnail.dtype == np.uint8:
                thumbnail = thumbnail.astype(np.uint8)
            tiff.save(thumbnail, photometric='RGB', planarconfig='CONTIG',
                bitspersample=8)
        if not image.dtype == np.float32:
            image = image.astype(np.float32)
        tiff.save(image, photometric='MINISBLACK', planarconfig='CONTIG',
                bitspersample=32, compress=9)

def readImageFloat(tiff_path: str, return_thumbnail = False,
                   read_or_die = True):
    try:
        multi_image = skimage.io.MultiImage(tiff_path)
        num_read_images = len(multi_image)
        if num_read_images == 0:
            raise Exception('No images found.')
        elif num_read_images == 1:
            return multi_image[0].squeeze(), None
        elif num_read_images == 2: # returns float, thumnail
            if multi_image[0].dtype == np.uint8:
                if not return_thumbnail: return multi_image[1].squeeze()
                else: return multi_image[1].squeeze(), multi_image[0].squeeze()
            else:
                if not return_thumbnail: return multi_image[0].squeeze()
                else: return multi_image[0].squeeze(), multi_image[1].squeeze()
        else: # returns list of images
            return [im.squeeze() for im in multi_image]       
    except Exception as e:
        LOG_ERROR('Failed to read image float: "%s"' %(e))
        if read_or_die:
            traceback.print_tb(e.__traceback__)
            sys.exit()
        return None, None

def writeImage(image: np.ndarray, path: str):
    image = toNumpy(image)
    skimage.io.imsave(path, image)

def readImage(path: str, read_or_die = True):
    try:
        return skimage.io.imread(path)
    except Exception as e:
        LOG_ERROR('Failed to read image: "%s"' % (e))
        if read_or_die:
            traceback.print_tb(e.__traceback__)
            sys.exit()
        return None





# array_utils.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#

import torch
import numpy as np

def sqrt(x):
    if type(x) == torch.Tensor: return torch.sqrt(x)
    else: return np.sqrt(x)

def atan2(y, x):
    if type(x) == torch.Tensor: return torch.atan2(y, x)
    else: return np.arctan2(y, x)

def asin(x):
    if type(x) == torch.Tensor: return torch.asin(x)
    else: return np.arcsin(x)

def acos(x):
    if type(x) == torch.Tensor: return torch.acos(x)
    else: return np.arccos(x)

def cos(x):
    if type(x) == torch.Tensor: return torch.cos(x)
    else: return np.cos(x)

def sin(x):
    if type(x) == torch.Tensor: return torch.sin(x)
    else: return np.sin(x)

def exp(x):
    if type(x) == torch.Tensor: return torch.exp(x)
    else: return np.exp(x)

def reshape(x, shape):
    if type(x) == torch.Tensor: return x.view(shape)
    else: return x.reshape(shape)

def toNumpy(arr) -> np.ndarray:
    if type(arr) == torch.Tensor: arr = arr.cpu().numpy()
    return arr

def concat(arr_list: list, axis=0):
    if type(arr_list[0]) == torch.Tensor: return torch.cat(arr_list, dim=axis)
    else: return np.concatenate(arr_list, axis=axis)

def polyval(P, x):
    if type(x) == torch.Tensor: P = torch.Tensor(P).to(x.device)
    if type(P) == torch.Tensor:
        npol = P.shape[0]
        val = torch.zeros_like(x)
        for i in range(npol-1):
            val = val * x + P[i] * x
        val += P[-1]
        return val
    else:
        return np.polyval(P, x)
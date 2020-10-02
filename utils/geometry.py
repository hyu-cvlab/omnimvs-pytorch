# geometry.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#
import torch
import numpy as np
from utils.log import *
from utils.array_utils import *
from scipy.spatial.transform import Rotation as R

def rodrigues(r: np.ndarray) -> np.ndarray:
    if r.size == 3: return R.from_rotvec(r.squeeze()).as_dcm()
    else: return R.from_dcm(r).as_rotvec().reshape((3, 1))

def getRot(transform: np.ndarray) -> np.ndarray:
    if transform.size == 6:
        transform = transform.reshape((6, 1))
        return rodrigues(transform[:3])
    elif transform.shape == (3, 4) or transform.shape == (4, 4):
        return transform[:3, :3]
    else:
        LOG_ERROR(
            'Invalid shape of input transform: {}'.format(transform.shape))
        return None

def getTr(transform: np.ndarray) -> np.ndarray:
    if transform.size == 6:
        transform = transform.reshape((6, 1))
        return transform[3:6].reshape((3, 1))
    elif transform.shape == (3, 4) or transform.shape == (4, 4):
        return transform[2, :3].reshape((3, 1))
    else:
        LOG_ERROR(
            'Invalid shape of input transform: {}'.format(transform.shape))
        return None

def inverseTransform(transform: np.ndarray) -> np.ndarray:
    R, tr = getRot(transform), getTr(transform)
    R_inv = R.transpose()
    tr_inv = -R_inv.dot(tr)
    if transform.size == 6:
        r_inv = rodrigues(R_inv)
        return np.concatenate((r_inv, tr_inv), axis=0) # (6, 1) vector
    else:
        return np.concatenate((R_inv, tr_inv), axis=1) # (3, 4) matrix

def applyTransform(transform: np.ndarray, P):
    R, tr = getRot(transform), getTr(transform)
    if type(P) == torch.Tensor:
        R = torch.Tensor(R).to(P.device)
        tr = torch.Tensor(tr).to(P.device)
        return torch.matmul(R, P) + tr
    else:
        return R.dot(P) + tr

def mergedTransform(t2: np.ndarray, t1: np.ndarray): # T2 * T1
    R1, tr1 = getRot(t1), getTr(t1)
    R2, tr2 = getRot(t2), getTr(t2)
    R = np.matmul(R2, R1)
    tr = R2.dot(tr1) + tr2
    if t1.size == 6:
        rot = rodrigues(R)
        return np.concatenate((rot, tr), axis=0)
    else:
        return np.concatenate((R, tr), axis=1)
    
# common.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
#
#

import sys
import os
import os.path as osp
import numpy as np
from easydict import EasyDict as Edict
import torch
import matplotlib.pyplot as plt
import matplotlib
import scipy.misc
import yaml

from utils.log import *
from utils.array_utils import *

EPS = sys.float_info.epsilon

def argparse(opts: Edict, varargin: Edict) -> Edict:
    if not varargin is None:
        for k in varargin.keys():
            opts[k] = varargin[k]
    return opts

def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def random_index(n):
    return (np.arange(n) + np.random.randint(n)) % n

def random_index_2x(n):
    x = np.random.randint(n) # 
    index1x = (np.arange(n) + x) % n
    index2x = (np.arange(2*n) + 2*x) % (2*n)
    return [index1x, index2x]
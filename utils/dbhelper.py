# utils.dbhelper.py
#
# Author: Changhee Won (changhee.1.won@gmail.com)
#
import os
import os.path as osp
import yaml
from utils.log import *
from utils.common import *
from utils.image import *
from utils.ocam import *

def loadDBConfigs(dbname: str, dbpath: str, opts: Edict) \
            -> (Edict, [OcamModel]):
    config_file = osp.join(dbpath, 'config.yaml')
    config = yaml.safe_load(open(config_file))

    for k in config['config'].keys(): opts[k] = config['config'][k]
    opts.min_depth = opts.omnimvs_sweep_min_depth
    for k in config['dataset'].keys(): opts[k] = config['dataset'][k]

    cameras_cfg = config['cameras']
    ocams = []
    for i in range(4):
        ocam = OcamModel()
        ocam.setConfig(cameras_cfg[i])
        mask_file = osp.join(dbpath, ocam.invalid_mask_file)
        if not osp.exists(mask_file):
            ocam.invalid_mask = ocam.makeInvisibleMask()
            writeImage(ocam.invalid_mask, mask_file)
        ocam.invalid_mask = readImage(mask_file).astype(np.bool)
        ocams.append(ocam)
    
    func = '__load_train_%s(opts)' % (dbname)
    try:
        opts = eval(func)
        LOG_INFO('Found "%s" training configs' % (dbname)) 
        opts.dtype = 'gt'
    except:
        LOG_INFO('Training configs not found "%s"' % (dbname)) 
    finally:
        return opts, ocams

def __load_train_sunny(opts):
    opts.train_idx = list(range(1, 701))
    opts.test_idx = list(range(701, 1001)) 
    opts.gt_phi = 45
    return opts

__load_train_sunset = __load_train_cloudy = __load_train_sunny

def __load_train_omnithings(opts):
    opts.train_idx = list(range(1, 4097)) + list(range(5121, 10241))
    opts.test_idx = list(range(4097, 5121))
    opts.gt_phi = 90
    return opts

def __load_train_omnihouse(opts):
    opts.train_idx = list(range(1, 2049))
    opts.test_idx = list(range(2049, 2561))
    opts.gt_phi = 90
    return opts

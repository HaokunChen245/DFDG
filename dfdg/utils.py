''' 
@copyright Copyright (c) Siemens AG, 2022
@author Haokun Chen <haokun.chen@siemens.com>
SPDX-License-Identifier: Apache-2.0
'''

import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision


def get_date():
    today = datetime.date.today()
    return today.strftime("%y%m%d")

def trial_name_string(trial):
    if hasattr(trial, "config"):
        config = trial.config
    else:
        config = trial
    name = "trial"
    for k in config["hps_list"]:
        v = config[k]
        if isinstance(v, bool):
            if v:
                name += f"_{k}"
        elif k == "folder_name":
            name += f"_{v[0]}"
        elif k == "source_domain":
            name += f"d{k[-1]}_{v}"
        elif isinstance(v, str):
            name += f"_{v}"
        else:
            if k == "batch_size":
                name += f"_bs_{v}"
            else:
                name += f"_{k}_{v}"
    return name


def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def setup_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: True.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

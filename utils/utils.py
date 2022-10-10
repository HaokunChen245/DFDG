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



def lr_policy(lr_fn):
    def _alr(optimizer, epoch):
        lr = lr_fn(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    return _alr


def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            if lr < base_lr * 0.01:
                lr = base_lr * 0.01
        return lr

    return lr_policy(_lr_fn)

def evaluate_onebatch(model, batch, y):
    model.eval()
    ans = 0
    tot = 0
    with torch.no_grad():
        o = model(batch).argmax(1)
        tmp_out = y.cpu() == o.cpu()
        ans += int(tmp_out.sum())
        tot += batch.shape[0]
    return ans / tot


def evaluate(model, loader):
    model.eval()
    ans = 0
    tot = 0
    with torch.no_grad():
        for (imgs, labels) in loader:
            o = model(imgs).argmax(1)
            tmp_out = labels == o
            ans += int(tmp_out.sum())
            tot += imgs.shape[0]
    return ans / tot


def get_net(backbone, num_class, pretrained=False, load_path=None):
    if backbone == "resnet18":
        net = torchvision.models.resnet18(pretrained=pretrained)
        net.fc = nn.Linear(512, num_class)
    elif backbone == "resnet34":
        net = torchvision.models.resnet34(pretrained=pretrained)
        net.fc = nn.Linear(512, num_class)
    elif backbone == "resnet50":
        net = torchvision.models.resnet50(pretrained=pretrained)
        net.fc = nn.Linear(2048, num_class)
    elif backbone == "alexnet":
        net = torchvision.models.alexnet(pretrained=pretrained)
        cls = list(net.classifier[:-1]) + [nn.Linear(4096, num_class)]
        net.classifier = nn.Sequential(*cls)

    if load_path:
        net.load_state_dict(torch.load(load_path))
        net.eval()
        for param in net.parameters():
            param.requires_grad = False

    return net

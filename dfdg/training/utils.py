''' 
@copyright Copyright (c) Siemens AG, 2022
@author Haokun Chen <haokun.chen@siemens.com>
SPDX-License-Identifier: Apache-2.0
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


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


def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = (
        torch.norm(diff1)
        + torch.norm(diff2)
        + torch.norm(diff3)
        + torch.norm(diff4)
    )
    loss_var_l1 = (
        (diff1.abs() / 255.0).mean()
        + (diff2.abs() / 255.0).mean()
        + (diff3.abs() / 255.0).mean()
        + (diff4.abs() / 255.0).mean()
    )
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def tv_loss(inputs_jit):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
    loss_var = (
        torch.norm(diff1)
        + torch.norm(diff2)
        + torch.norm(diff3)
        + torch.norm(diff4)
    )
    return loss_var


def kl_loss(y, teacher_scores, temp=3, softmax_applied=False):
    p = F.log_softmax(y / temp, dim=1)

    if softmax_applied:
        q = teacher_scores
    else:
        q = F.softmax(teacher_scores / temp, dim=1)

    l_kl = F.kl_div(p, q, reduction="batchmean")
    l_kl = l_kl * temp**2
    return l_kl


class DeepInversionFeatureHook_wDiversify:
    """
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    code from https://github.com/NVlabs/DeepInversion/blob/5e4a57883b60f024ada940fcb8a0d9be479dca82/cifar10/deepinversion_cifar10.py#L48
    """

    def __init__(
        self,
        module,
        idx,
        layer_index,
        use_slack=30,
        cross_domain_generation=False,
        use_tensor_in_slack=False,
        more_slack_1st_var=False,
    ):

        self.hook = module.register_forward_hook(self.hook_fn)
        self.layer_index = layer_index
        self.perc = use_slack
        self.idx = idx
        self.more_slack_1st_var = more_slack_1st_var
        self.use_tensor_in_slack = use_tensor_in_slack
        self.mean_slack = 0
        self.var_slack = 0
        self.r_feature = None
        self.r_feature_slacked = None

        if cross_domain_generation:
            assert use_slack > 0

        # For initialization of statistics
        self.computing_target_stats = False
        self.target_mean = module.running_mean
        self.target_var = module.running_var

        self.computing_base_stats = False
        self.base_mean = module.running_mean
        self.base_var = module.running_var
        self.diff_mean = None
        self.diff_var = None
        self.compute_slackness()

    def compute_slackness(self):
        diff_mean = torch.abs(
            self.target_mean.data.to("cuda:0") - self.base_mean
        ).cpu()
        diff_var = torch.abs(
            self.target_var.data.to("cuda:0") - self.base_var
        ).cpu()
        if not self.use_tensor_in_slack:
            # use percentile of the calculated diff values.
            mean_slack = np.percentile(diff_mean, self.perc)
            if self.more_slack_1st_var and self.idx == 0:
                # use more slack on the first BN layer of the teacher
                var_slack = np.percentile(diff_var, 90)
            else:
                var_slack = np.percentile(diff_var, self.perc)

            self.mean_slack = mean_slack
            self.var_slack = var_slack
        else:
            # use diff values tensor.
            mean_slack = diff_mean * (self.perc / 100)
            if self.more_slack_1st_var and self.idx == 0:
                # use more slack on the first BN layer of the teacher
                var_slack = diff_var * (90 / 100)
            else:
                var_slack = diff_var * (self.perc / 100)

            self.mean_slack = mean_slack.to("cuda:0")
            self.var_slack = var_slack.to("cuda:0")

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3]).to("cuda:0")
        var = (
            input[0]
            .permute(1, 0, 2, 3)
            .contiguous()
            .view([nch, -1])
            .var(1)
            .to("cuda:0")
        )

        if self.computing_target_stats:
            self.target_mean = mean
            self.target_var = var
            self.compute_slackness()
            return

        # computing base statistics for slack computation
        if self.computing_base_stats:
            # using current stats as the base.
            self.base_mean = mean
            self.base_var = var
            self.compute_slackness()
            return

        r_feature = torch.norm(
            self.target_var.data.type(var.type()).to("cuda:0") - var, 2
        ) + torch.norm(
            self.target_mean.data.type(mean.type()).to("cuda:0") - mean, 2
        )
        self.r_feature = r_feature

        if self.perc == 0:
            self.r_feature_slacked = self.r_feature
        else:
            diff_mean = (
                torch.abs(
                    self.target_mean.data.type(mean.type()).to("cuda:0") - mean
                )
                - self.mean_slack
            )
            diff_var = (
                torch.abs(
                    self.target_var.data.type(var.type()).to("cuda:0") - var
                )
                - self.var_slack
            )

            zeros_mean = torch.zeros(diff_mean.shape)
            zeros_var = torch.zeros(diff_var.shape)

            r_feature_slacked = torch.norm(
                torch.where(diff_mean < 0, zeros_mean.to("cuda:0"), diff_mean),
                2,
            ) + torch.norm(
                torch.where(diff_var < 0, zeros_var.to("cuda:0"), diff_var), 2
            )
            self.r_feature_slacked = r_feature_slacked

        self.diff_mean = torch.norm(
            self.target_mean.data.type(mean.type()).to("cuda:0") - mean, 2
        )
        self.diff_var = torch.norm(
            self.target_var.data.type(var.type()).to("cuda:0") - var, 2
        )

    def close(self):
        self.hook.remove()


def print_moment_loss(T, loss_r_feature_layers, cmd=True):
    loss_moment = sum(
        [m.r_feature for (idx, m) in enumerate(loss_r_feature_layers)]
    )
    ma = get_map(T)
    out = [0, 0, 0, 0, 0]
    for idx, m in enumerate(loss_r_feature_layers):
        out[ma[idx]] += m.r_feature.item()

    if cmd:
        print("total Moment Loss: {:.4f}".format(loss_moment.item()))
        print("first BN: {:.4f}".format(out[0]))
        for i in range(4):
            print("layer{}: {:.4f}".format(i, out[i + 1]))

    return out


def get_map(T):
    m = {}
    m[0] = 0
    count = 0
    for module in T.layer1.modules():
        if isinstance(module, nn.BatchNorm2d):
            count += 1
            m[count] = 1
    for module in T.layer2.modules():
        if isinstance(module, nn.BatchNorm2d):
            count += 1
            m[count] = 2
    for module in T.layer3.modules():
        if isinstance(module, nn.BatchNorm2d):
            count += 1
            m[count] = 3
    for module in T.layer4.modules():
        if isinstance(module, nn.BatchNorm2d):
            count += 1
            m[count] = 4
    return m


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

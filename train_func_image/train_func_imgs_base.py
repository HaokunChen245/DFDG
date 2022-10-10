import os
import random

import torch
from torch import nn
import torchvision
import torchvision.transforms as tfs
from ray import tune
from utils.dataset_utils import get_class_number
from utils.feature_matching import print_moment_loss
from utils.losses import get_image_prior_losses
from utils.utils import evaluate_onebatch, lr_cosine_policy


def train_one_batch(
    config,
    inputs,
    y_label,
    batch_count,
    loss_r_feature_layers,
    rescale,
    t_source_a,
    t_source_b=None,
):
    lambda_ce = 1
    lambda_s = config["Ws"]
    lambda_l2 = 1.5 * 1e-5
    lambda_t = 1e-4

    best_loss_moment = 1e5
    best_loss = 1e5
    tot_it = -1
    t_source_b_acc = -1

    image_mean = [0.485, 0.456, 0.406]
    image_var = [0.229, 0.224, 0.225]
    norm = tfs.Normalize(mean=image_mean, std=image_var)

    save_nrow = get_class_number(config["dataset"])
    if save_nrow > 10:
        save_nrow = 5

    cls_criterion = nn.CrossEntropyLoss()
    random_erasing = tfs.RandomErasing(value=0, p=0.5)
    pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)

    inputs = inputs.requires_grad_(True)

    iters_low_res = int(config["max_iters_train"] * 0.65)
    iters_high_res = int(config["max_iters_train"] * 0.35)

    if not config["multi_resolution"]:
        optimizer = torch.optim.Adam(
            [inputs], lr=config["lr"], betas=[0.9, 0.99], eps=1e-8
        )
        lr_scheduler = lr_cosine_policy(config["lr"], 100, config["max_iters_train"])

    for res in [2, 1]:
        if res == 2:
            curr_max_iter = iters_low_res
        else:
            curr_max_iter = iters_high_res

        if config["multi_resolution"]:
            optimizer = torch.optim.Adam(
                [inputs], lr=config["lr"], betas=[0.5, 0.9], eps=1e-8
            )
            lr_scheduler = lr_cosine_policy(config["lr"], 100, curr_max_iter)

        for it in range(curr_max_iter + 1):
            if config["multi_resolution"]:
                cur_lr = lr_scheduler(optimizer, it)
            else:
                if res == 2:
                    cur_lr = lr_scheduler(optimizer, it)
                else:
                    cur_lr = lr_scheduler(
                        optimizer, it + iters_low_res
                    )

            tot_it += 1
            if config["dataset"] == "Digits":
                lim_0, lim_1 = 2, 2
            elif config["dataset"] == "miniDomainNet":
                lim_0, lim_1 = 9, 9
            else:
                # for 224x224 images
                lim_0, lim_1 = 30, 30
            imgs = inputs
            if res == 2 and config["multi_resolution"]:
                lim_0, lim_1 = int(lim_0) / 2, int(lim_1) / 2
                imgs = pooling_function(inputs)

            # apply random jitter offsets
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            imgs = torch.roll(imgs, shifts=(off1, off2), dims=(2, 3))

            # Flipping
            flip = random.random() > 0.5
            if flip and not config["dataset"] == "Digits":
                imgs = torch.flip(imgs, dims=(3,))

            # Random Erasing
            imgs = random_erasing(imgs)

            optimizer.zero_grad()
            t_source_a.zero_grad()
            if t_source_b:
                t_source_b.zero_grad()

            # total variation loss
            loss_tv_l1, loss_tv_l2 = get_image_prior_losses(imgs)
            loss_tv = loss_tv_l1 + loss_tv_l2

            # Image L2 loss
            loss_l2 = torch.norm(imgs.view(imgs.shape[0], -1), dim=1).mean()

            # classification loss
            logits_a = t_source_a(norm(imgs))
            if t_source_b:
                logits_b = t_source_b(norm(imgs))

            # use KL instead of two CE loss, Teacher_A as the target logits
            # loss_CE = kl_loss(logits_b, logits_a, temp=3, softmax_applied=False)
            loss_ce = cls_criterion(logits_a, y_label.squeeze().to(config["device"]))
            if t_source_b:
                cls_criterion(logits_b, y_label.squeeze().to(config["device"]))

            loss = lambda_t * loss_tv + lambda_l2 * loss_l2 + lambda_ce * loss_ce

            # moment matching loss
            loss_moment = sum(
                [
                    m.r_feature * rescale[idx]
                    for (idx, m) in enumerate(loss_r_feature_layers)
                ]
            )
            loss_moment_slacked = sum(
                [
                    m.r_feature_slacked * rescale[idx]
                    for (idx, m) in enumerate(loss_r_feature_layers)
                ]
            )

            loss = loss + lambda_s * loss_moment_slacked
            best_loss_moment = min(loss_moment_slacked, best_loss_moment)

            loss.backward()
            optimizer.step()

            if tot_it % 200 == 0:
                moment_loss_per_layer = print_moment_loss(
                    t_source_a, loss_r_feature_layers, cmd=False
                )

                with torch.no_grad():
                    temp_imgs = imgs.clone().detach()
                    t_source_a_acc = evaluate_onebatch(
                        t_source_a, norm(temp_imgs), y_label.squeeze().cpu()
                    )
                    if t_source_b:
                        t_source_b_acc = evaluate_onebatch(
                            t_source_b, norm(temp_imgs), y_label.squeeze().cpu()
                        )

                tune.report(
                    iteration=tot_it,
                    mLoss_slacked=float(loss_moment_slacked),
                    mLoss=float(loss_moment),
                    tvLoss=float(loss_tv),
                    l2Loss=float(loss_l2),
                    CELoss=float(loss_ce),
                    loss=float(loss),
                    first_BN_mean_diff=float(loss_r_feature_layers[0].diff_mean),
                    first_BN_var_diff=float(loss_r_feature_layers[0].diff_var),
                    first_BN_mLoss=float(moment_loss_per_layer[0]),
                    layer0_mLoss=float(moment_loss_per_layer[1]),
                    layer1_mLoss=float(moment_loss_per_layer[2]),
                    layer2_mLoss=float(moment_loss_per_layer[3]),
                    layer3_mLoss=float(moment_loss_per_layer[4]),
                    Teacher_A_acc=float(t_source_a_acc),
                    Teacher_B_acc=float(t_source_b_acc),
                    lr=float(cur_lr),
                )

            inputs.data = torch.clamp(inputs.data, min=0, max=1)

            if tot_it % 50 == 0:
                if t_source_b:
                    torchvision.utils.save_image(
                        inputs,
                        os.path.join(
                            config["image_snapshot_dir"],
                            f'imgs_{config["source_domain_B"]}_{batch_count}.jpg',
                        ),
                        nrow=save_nrow,
                    )
                else:
                    torchvision.utils.save_image(
                        inputs,
                        os.path.join(
                            config["image_snapshot_dir"], f"imgs_{batch_count}.jpg"
                        ),
                        nrow=save_nrow,
                    )

            if best_loss > loss:
                best_loss = loss
                if t_source_b:
                    torch.save(
                        inputs,
                        os.path.join(
                            config["image_snapshot_dir"],
                            f'imgs_{config["source_domain_B"]}_{batch_count}.pt',
                        ),
                    )
                else:
                    torch.save(
                        inputs,
                        os.path.join(
                            config["image_snapshot_dir"], f"imgs_{batch_count}.pt"
                        ),
                    )

    del imgs
    del inputs
    del loss
    torch.cuda.empty_cache()

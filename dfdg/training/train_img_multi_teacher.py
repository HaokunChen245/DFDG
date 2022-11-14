''' 
@copyright Copyright (c) Siemens AG, 2022
@author Haokun Chen <haokun.chen@siemens.com>
SPDX-License-Identifier: Apache-2.0
'''

import os

import torch
import torch.nn as nn
import yaml

from dfdg.datasets.syn_datasets import FAKE
from dfdg.datasets.utils import get_dataset_stats
from dfdg.training.train_img_base import train_img_base
from dfdg.training.utils import DeepInversionFeatureHook_wDiversify
from dfdg.training.utils import get_map
from dfdg.training.utils import get_net
from dfdg.utils import trial_name_string


def train_img_multi_teacher(config):
    config["device"] = torch.device("cuda:0")

    image_mean, image_var = get_dataset_stats(config['dataset'])
    source_domains = config["source_domains"]
    source_domains.remove(config["source_domain_A"])

    config["image_snapshot_dir"] = os.path.join(
        config["log_root_dir"], config["log_run_name"], config["trial_name"]
    )
    with open(
        os.path.join(
            config['image_snapshot_dir'], "images_generation_setting.yml"
        ),
        "w",
    ) as f:
        yaml.dump(config, f)

    ini_cls = list(range(config["num_class"]))
    for b in range(config["batch_num"]):
        tmp = (b * config["batch_size"]) % config["num_class"]
        cls = ini_cls[tmp:] + ini_cls[:tmp]
        y_label = torch.LongTensor(
            cls * (config["batch_size"] // config["num_class"])
            + cls[: config["batch_size"] % config["num_class"]]
        ).view((config["batch_size"], 1))

        # for every batch, we adapt it for all other domains.
        for config["source_domain_B"] in source_domains:
            torch.save(
                y_label,
                os.path.join(
                    config["image_snapshot_dir"],
                    f'labels_{config["source_domain_A"]}_{config["source_domain_B"]}_{b}.pt',
                ),
            )

            teacher_snapshot_dir = os.path.join(
                config["teacher_dir"],
                f'{config["source_domain_A"]}_{config["teacher_backbone"]}',
                "model_best.pth",
            )
            t_source_a = (
                get_net(
                    config["teacher_backbone"],
                    config["num_class"],
                    pretrained=False,
                    load_path=teacher_snapshot_dir,
                )
                .to(config["device"])
                .eval()
            )
            teacher_snapshot_dir = os.path.join(
                config["teacher_dir"],
                f'{config["source_domain_B"]}_{config["teacher_backbone"]}',
                "model_best.pth",
            )
            t_source_b = (
                get_net(
                    config["teacher_backbone"],
                    config["num_class"],
                    pretrained=False,
                    load_path=teacher_snapshot_dir,
                )
                .to(config["device"])
                .eval()
            )

            loss_r_feature_layers = []
            layer_mapping = get_map(t_source_a)
            count = -1
            rescale = []
            for module in t_source_a.modules():
                if isinstance(module, nn.BatchNorm2d):
                    count += 1
                    loss_r_feature_layers.append(
                        DeepInversionFeatureHook_wDiversify(
                            module,
                            count,
                            layer_index=layer_mapping[count],
                            cross_domain_generation=True,
                            use_slack=config["slack"],
                            more_slack_1st_var=True,
                            use_tensor_in_slack=True,
                        )
                    )
                    if count == 0:
                        rescale.append(1.0)
                    elif layer_mapping[count] == 4:  # not using AdaBN in layer4
                        rescale.append(0.0)
                    else:
                        rescale.append(1.0)

            # first forward
            for l in loss_r_feature_layers:
                l.computing_target_stats = True
            current_folder_name = config["img_dir_stage1"]
            fakeset_source_b = FAKE(
                dataset=config["dataset"],
                image_root_dir=config["log_root_dir"],
                source_domain_a=config["source_domain_B"],
                folder_name=current_folder_name,
                portion=-1,
                mode="stage1",
            )
            fakeset_source_b_loader = torch.utils.data.DataLoader(
                dataset=fakeset_source_b,
                shuffle=True,
                batch_size=len(fakeset_source_b),
                collate_fn=fakeset_source_b.collate_fn,
            )
            for imgs, _, _ in fakeset_source_b_loader:
                with torch.no_grad():
                    _ = t_source_a(imgs.to(config["device"]))
                break
            del fakeset_source_b, fakeset_source_b_loader
            torch.cuda.empty_cache()
            for l in loss_r_feature_layers:
                l.computing_target_stats = False

            imgs_init_mean = image_mean
            imgs_init_var = image_var

            inputs = []
            with torch.no_grad():
                for i in range(3):
                    inputs_v = (
                        torch.ones(
                            (
                                config["batch_size"],
                                config["img_size"],
                                config["img_size"],
                            )
                        )
                        * imgs_init_var[i]
                    )
                    inputs.append(
                        torch.normal(mean=imgs_init_mean[i], std=inputs_v)
                    )
                inputs = torch.stack(inputs, 1).to(config["device"])
                inputs = torch.clamp(inputs, min=0, max=1)

            ################  training
            train_img_base(
                config,
                inputs,
                y_label,
                b,
                loss_r_feature_layers,
                rescale,
                t_source_a,
                t_source_b,
            )

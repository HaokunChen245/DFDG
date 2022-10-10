import os

import torch
import torch.nn as nn
import yaml
from datasets import FAKE

from train_func_image.train_func_imgs_base import train_one_batch
from utils.feature_matching import DeepInversionFeatureHook_wDiversify, get_map
from utils.utils import get_net, setup_seed, trial_name_string


def train_one_setting_multi_teacher(config):
    setup_seed(config["trial_seed"])
    config["device"] = torch.device("cuda:0")

    image_mean = [0.485, 0.456, 0.406]
    image_var = [0.229, 0.224, 0.225]

    source_domains = config["source_domains"]
    source_domains.remove(config["source_domain_A"])

    config["trial_name"] = trial_name_string(config)
    config["image_snapshot_dir"] = os.path.join(
        config["log_root_dir"], config["log_run_name"], config["trial_name"]
    )
    with open(os.path.join(
        config['image_snapshot_dir'], "images_generation_setting.yml"
        )
        , "w"
    ) as f:
        yaml.dump(config, f)

    ini_cls = list(range(config["num_class"]))
    for b in range(config["batch_numbers"]):
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
                    f'labels_{config["source_domain_B"]}_{b}.pt',
                ),
            )

            teacher_snapshot_dir = os.path.join(
                config["teacher_snapdir"],
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
                config["teacher_snapdir"],
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
                            more_slack_1st_var=config["more_slack_1st_var"],
                            use_tensor_in_slack=True,
                        )
                    )
                    if count == 0:
                        rescale.append(config["scale_1BN"])
                    elif layer_mapping[count] == 4:  # not using AdaBN in layer4
                        rescale.append(0.0)
                    else:
                        rescale.append(1.0)

            # first forward
            for l in loss_r_feature_layers:
                l.computing_target_stats = True
            current_folder_name = config["stage1_snapdir"].replace(
                "domain", config["source_domain_B"]
            )
            fakeset_source_b = FAKE(
                dataset=config["dataset"],
                image_root_dir=config["log_root_dir"],
                source_domain_a=config["source_domain_B"],
                folder_name=current_folder_name,
                portion=-1,
                mode="DI",
            )
            fakeset_source_b_loader = torch.utils.data.DataLoader(
                dataset=fakeset_source_b,
                shuffle=True,
                batch_size=len(fakeset_source_b),
                collate_fn=fakeset_source_b.collate_fn,
            )
            for imgs, _, _ in fakeset_source_b_loader:
                print(imgs.shape)
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
                    inputs.append(torch.normal(mean=imgs_init_mean[i], std=inputs_v))
                inputs = torch.stack(inputs, 1).to(config["device"])
                inputs = torch.clamp(inputs, min=0, max=1)

            ################  training
            train_one_batch(
                config,
                inputs,
                y_label,
                b,
                loss_r_feature_layers,
                rescale,
                t_source_a,
                t_source_b,
            )

import os

import torch
import torch.nn as nn
import yaml

from train_func_image.train_func_imgs_base import train_one_batch
from utils.feature_matching import DeepInversionFeatureHook_wDiversify, get_map
from utils.utils import get_net, setup_seed, trial_name_string


def train_one_setting_single_teacher(config):
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
    if not os.path.isdir(config["image_snapshot_dir"]):
        os.system(f"mkdir -p {config['image_snapshot_dir']}")
    with open(
        os.path.join(config['image_snapshot_dir'],
        "images_generation_setting.yml")
        , "w"
    ) as f:
        yaml.dump(config, f)

    ini_cls = list(range(config["num_class"]))

    for b in range(config["batch_numbers"]):
        # label saving
        tmp = (b * config["batch_size"]) % config["num_class"]
        cls = ini_cls[tmp:] + ini_cls[:tmp]
        y_label = torch.LongTensor(
            cls * (config["batch_size"] // config["num_class"])
            + cls[: config["batch_size"] % config["num_class"]]
        ).view((config["batch_size"], 1))
        torch.save(
            y_label, os.path.join(config["image_snapshot_dir"], f"labels_{b}.pt")
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
        # not computing the grads regarding these params.
        # But they will still be in the computational graph.
        for param in t_source_a.parameters():
            param.requires_grad = False

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
                        cross_domain_generation=False,
                        use_slack=config["slack"],
                        more_slack_1st_var=config["more_slack_1st_var"],
                        use_tensor_in_slack=False,
                    )
                )
                if count == 0:
                    rescale.append(config["scale_1BN"])
                else:
                    rescale.append(1.0)

        for l in loss_r_feature_layers:
            l.computing_base_stats = True
        random_inputs = torch.randn(
            (config["batch_size"], 3, config["img_size"], config["img_size"]),
            dtype=torch.float,
        )
        with torch.no_grad():
            _ = t_source_a(random_inputs.to(config["device"]))
        del random_inputs
        for l in loss_r_feature_layers:
            l.computing_base_stats = False

        imgs_init_mean = image_mean
        imgs_init_var = image_var

        inputs = []
        with torch.no_grad():
            for i in range(3):
                inputs_v = (
                    torch.ones(
                        (config["batch_size"], config["img_size"], config["img_size"])
                    )
                    * imgs_init_var[i]
                )
                inputs.append(torch.normal(mean=imgs_init_mean[i], std=inputs_v))
            inputs = torch.stack(inputs, 1).to(config["device"])
            inputs = torch.clamp(inputs, min=0, max=1)

        train_one_batch(
            config,
            inputs,
            y_label,
            b,
            loss_r_feature_layers,
            rescale,
            t_source_a,
        )

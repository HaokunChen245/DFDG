import os

import torch
import torch.nn.functional as F
import yaml
from ray import tune
from torch.utils.data import ConcatDataset
from datasets import FAKE, PACS, VLCS, Digits, OfficeHome, miniDomainNet
from utils.dataset_utils import get_class_number, get_source_domains
from utils.losses import kl_loss
from utils.utils import (evaluate, get_net, lr_cosine_policy, setup_seed,
                         trial_name_string)


def train_one_setting_student_image(config):
    config["folder_dicts"] = [
        {
            "folder_name": config['stage1_snapdir'],
            "batch_nr": -1,
            "mode": "DI",
            "augment": True,
            "select": True,
        },
        {
            "folder_name": config['stage2_snapdir'],
            "batch_nr": -1,
            "mode": "adabn",
            "augment": True,
            "select": True,
        },
    ]

    config["trial_name"] = trial_name_string(config)
    setup_seed(config["trial_seed"])
    dataset_mapping = {
        "PACS": PACS,
        "VLCS": VLCS,
        "OfficeHome": OfficeHome,
        "Digits": Digits,
        "miniDomainNet": miniDomainNet,
    }
    assert config["dataset"] in dataset_mapping.keys()
    dataset = dataset_mapping[config["dataset"]]

    # for fake image: we only use the validation set of target domain as model selection
    valset = dataset(
        config["dataset_dir"],
        mode="val",
        img_size=config["img_size"],
        domain=config["target_domain"],
    )
    valset_loader = torch.utils.data.DataLoader(
        dataset=valset,
        batch_size=config["eval_batch_size"],
        collate_fn=valset.collate_fn,
    )

    trainsets = []
    for source_domain_a in config["source_domains"]:
        if source_domain_a == config["target_domain"]:
            continue

        for f in config["folder_dicts"]:
            if not f["select"]:
                continue
            current_folder_name = f["folder_name"].replace("domain", source_domain_a)
            trainset = FAKE(
                dataset=config["dataset"],
                source_domain_a=source_domain_a,
                image_root_dir=config["log_root_dir"],
                folder_name=current_folder_name,
                portion=-1,
                mode=f["mode"],
                augment_in_fake=f["augment"],
                target_domain=config["target_domain"],
            )
            trainsets.append(trainset)

    trainset_loader = torch.utils.data.DataLoader(
        dataset=ConcatDataset(trainsets),
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=trainset.collate_fn,
    )

    #########Here start from ImageNet pretrained network.##################
    net = get_net(config["teacher_backbone"], config["num_class"], pretrained=True)
    net = net.cuda().train()

    #########See whether to fix bn statistics###########
    # net.apply(set_bn_eval)

    student_folder_dir = os.path.join(
        config["log_root_dir"], config["log_run_name"], config["trial_name"]
    )
    with open(os.path.join(student_folder_dir, "fake_images_folders_setting.yml"), "w") as f:
        yaml.dump_all(config["folder_dicts"], f)

    max_ep = config["max_iters_train"]
    optimizer = torch.optim.SGD(
        net.parameters(),
        weight_decay=0.0005,
        momentum=0.9,
        nesterov=True,
        lr=config["lr"],
    )
    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(max_ep*0.6), int(max_ep*0.8)], gamma=0.9)
    lr_scheduler = lr_cosine_policy(
        config["lr"], 100, config["max_iters_train"] * len(trainset_loader)
    )
    it = 0
    temperature = 3
    num_cls = get_class_number(config["dataset"])

    teachers = []
    for d in get_source_domains(config["dataset"]):
        if d == config["target_domain"]:
            teachers.append(None)
        else:
            teacher = get_net(config["teacher_backbone"], num_cls)
            teacher.load_state_dict(
                torch.load(
                    os.path.join(
                        config["teacher_snapdir"],
                        f"{d}_{config['teacher_backbone']}",
                        "model_best.pth",
                    )
                )
            )
            teacher.eval().cuda()  # for evaluation
            teachers.append(teacher)

    cur_trial_best = 0
    best_it = 0
    for ep in range(max_ep):
        for (imgs, _, index_ts) in trainset_loader:
            it += 1
            cur_lr = lr_scheduler(optimizer, it)
            optimizer.zero_grad()
            outputs = net(imgs)

            index_ts = index_ts.unsqueeze(1)
            # prepare teacher logits
            with torch.no_grad():
                logits = []
                for teacher in teachers:
                    if teacher:
                        logit = teacher(imgs)
                        logits.append(logit)
                    else:
                        logits.append(torch.zeros((imgs.shape[0], num_cls)).cuda())

                logits = torch.stack(logits, 1)
                logits = F.softmax(logits / temperature, dim=2)
                targets = index_ts @ logits
                targets = targets.squeeze()

            l_kl = kl_loss(outputs, targets, temp=temperature, softmax_applied=True)

            l_kl.backward()
            optimizer.step()
            tune.report(
                loss=float(l_kl),
                lr=float(cur_lr),
            )
            #         scheduler.step()

            if it % 20 == 0:

                val_acc = evaluate(net, valset_loader)
                tune.report(acc=float(val_acc))
                if not os.path.exists(os.path.join(student_folder_dir, "best_acc.pt")):
                    torch.save(
                        net.state_dict(),
                        os.path.join(student_folder_dir, "model_best.pth"),
                    )
                    torch.save(
                        torch.Tensor([val_acc]),
                        os.path.join(student_folder_dir, "best_acc.pt"),
                    )
                    best = float(val_acc)
                else:
                    best = float(
                        torch.load(os.path.join(student_folder_dir, "best_acc.pt"))
                    )
                    if best < float(val_acc):
                        torch.save(
                            torch.Tensor([val_acc]),
                            os.path.join(student_folder_dir, "best_acc.pt"),
                        )
                        torch.save(
                            net.state_dict(),
                            os.path.join(student_folder_dir, "model_best.pth"),
                        )
                        best = val_acc

                if cur_trial_best < val_acc:
                    cur_trial_best = max(val_acc, cur_trial_best)
                    best_it = it

                if val_acc < cur_trial_best and it - best_it >= 300 and ep >= 8:
                    tune.report(done=True)
                    return

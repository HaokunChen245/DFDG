''' 
@copyright Copyright (c) Siemens AG, 2022
@author Haokun Chen <haokun.chen@siemens.com>
SPDX-License-Identifier: Apache-2.0
'''

import os
import torch
import torch.utils.data as data
import random
import argparse
import sys
from dfdg.datasets.utils import get_backbone, get_dataset
from dfdg.datasets.utils import get_class_number
from dfdg.datasets.utils import get_source_domains
from dfdg.datasets.utils import get_img_size
from dfdg.evaluation.utils import evaluate_oneset
from dfdg.training.utils import get_net


def evaluate(
    dataset,
    dataset_dir,
    model_dir,
    student_dir,
    batch_size,
):
    dataset_class = get_dataset(dataset)
    backbone = get_backbone(dataset)
    num_class = get_class_number(dataset)
    img_size = get_img_size(dataset)
    testset_loaders = {}
    for d in get_source_domains(dataset):
        testset = dataset_class(
            os.path.realpath(os.path.join(dataset_dir, dataset)),
            mode='test',
            img_size=img_size,
            domain=d,
        )
        testset_loaders[d] = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=batch_size,
            collate_fn=testset.collate_fn,
        )

    accs = {}
    for p in os.listdir(os.path.join(model_dir, student_dir)):
        if not 'model' in p:
            continue
        model_snapshot_dir = os.path.join(model_dir, student_dir, p)
        T = get_net(backbone, num_class, pretrained=True)
        T.load_state_dict(torch.load(model_snapshot_dir))
        T.eval().cuda()  # for evaluation
        target_domain = p.split('_')[1]
        if target_domain == 'art':
            target_domain = 'art_painting'
        accs[target_domain] = evaluate_oneset(T, testset_loaders[target_domain])

    return accs

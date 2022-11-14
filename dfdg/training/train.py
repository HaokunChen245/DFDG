''' 
@copyright Copyright (c) Siemens AG, 2022
@author Haokun Chen <haokun.chen@siemens.com>
SPDX-License-Identifier: Apache-2.0
'''

import argparse
import os
import time

import ray
from ray import tune

from dfdg.training.train_img_multi_teacher import train_img_multi_teacher
from dfdg.training.train_img_single_teacher import train_img_single_teacher
from dfdg.training.train_student_syn_img import train_student_syn_img
from dfdg.datasets.utils import get_backbone, get_img_size
from dfdg.datasets.utils import get_class_number
from dfdg.datasets.utils import get_source_domains
from dfdg.utils import trial_name_string


def start_ray_tuning(args, func):
    resources_per_trial = {"gpu": 0.5, "cpu": 1}
    args['trial_name'] = trial_name_string(args)
    ray.init()
    tune.run(
        func,
        name=args['log_run_name'],
        config=args,
        resources_per_trial=resources_per_trial,
        local_dir=args['log_root_dir'],
        trial_name_creator=trial_name_string,
        trial_dirname_creator=trial_name_string,
        # to resume one failed trial, specify the name and set the resume argument
        # use "PROMPT" to restart from local dir
        #                 resume = "PROMPT",
        #                 name = "aws_folder"
    )
    ray.shutdown()


def train_image_stage1(
    dataset,
    batch_size,
    batch_num,
    lr,
    lambda_moment,
    iterations,
    slack,
    model_dir,
    seed,
):

    train_args = {}
    train_args['dataset'] = dataset
    train_args['batch_size'] = batch_size
    train_args['batch_num'] = batch_num
    train_args['lr'] = lr
    train_args['lambda_moment'] = lambda_moment
    train_args['img_size'] = get_img_size(dataset)
    train_args['teacher_backbone'] = get_backbone(dataset)
    train_args['num_class'] = get_class_number(train_args['dataset'])
    train_args['source_domains'] = get_source_domains(train_args['dataset'])
    train_args['teacher_dir'] = os.path.join(
        os.path.realpath(model_dir), 'teacher_models', dataset
    )
    train_args['seed'] = seed
    train_args['source_domain_A'] = tune.grid_search(
        train_args['source_domains']
    )
    train_args['iterations'] = iterations
    train_args['slack'] = slack
    train_args['hps_list'] = ['lr', 'lambda_moment', 'slack']
    train_args['log_root_dir'] = os.path.realpath(model_dir)
    train_args['log_run_name'] = 'image_stage1_' + time.strftime(
        '%Y%m%d_%H%M%S', time.localtime(time.time())
    )

    start_ray_tuning(train_args, train_img_single_teacher)

    return os.path.join(train_args['log_run_name'], train_args['trial_name'])


def train_image_stage2(
    dataset,
    batch_size,
    batch_num,
    lr,
    lambda_moment,
    iterations,
    slack,
    model_dir,
    img_dir_stage1,
    seed,
):

    train_args = {}
    train_args['dataset'] = dataset
    train_args['batch_size'] = batch_size
    train_args['batch_num'] = batch_num
    train_args['lr'] = lr
    train_args['lambda_moment'] = lambda_moment
    train_args['img_dir_stage1'] = img_dir_stage1
    train_args['img_size'] = get_img_size(dataset)
    train_args['teacher_backbone'] = get_backbone(dataset)
    train_args['num_class'] = get_class_number(train_args['dataset'])
    train_args['source_domains'] = get_source_domains(train_args['dataset'])
    train_args['teacher_dir'] = os.path.join(
        os.path.realpath(model_dir), 'teacher_models', dataset
    )
    train_args['seed'] = seed
    train_args['source_domain_A'] = tune.grid_search(
        train_args['source_domains']
    )
    train_args['iterations'] = iterations
    train_args['slack'] = slack
    train_args['hps_list'] = ['lr', 'lambda_moment', 'slack']
    train_args['log_root_dir'] = os.path.realpath(model_dir)
    train_args['log_run_name'] = 'image_stage2_' + time.strftime(
        '%Y%m%d_%H%M%S', time.localtime(time.time())
    )

    start_ray_tuning(train_args, train_img_multi_teacher)

    return os.path.join(train_args['log_run_name'], train_args['trial_name'])


def train_student(
    dataset,
    dataset_dir,
    batch_size,
    lr,
    iterations,
    model_dir,
    img_dir_stage1,
    img_dir_stage2,
    seed,
):

    train_args = {}
    train_args['dataset'] = dataset
    train_args['dataset_dir'] = os.path.join(
        os.path.realpath(dataset_dir), dataset
    )
    train_args['batch_size'] = batch_size
    train_args['lr'] = lr
    train_args['img_dir_stage1'] = img_dir_stage1
    train_args['img_dir_stage2'] = img_dir_stage2
    train_args['img_size'] = get_img_size(dataset)
    train_args['teacher_backbone'] = get_backbone(dataset)
    train_args['num_class'] = get_class_number(train_args['dataset'])
    train_args['source_domains'] = get_source_domains(train_args['dataset'])
    train_args['teacher_dir'] = os.path.join(
        os.path.realpath(model_dir), 'teacher_models', dataset
    )
    train_args['seed'] = seed
    train_args['target_domain'] = tune.grid_search(train_args['source_domains'])
    train_args['iterations'] = iterations
    train_args['hps_list'] = ['lr', 'lambda_moment', 'slack']
    train_args['log_root_dir'] = os.path.realpath(model_dir)
    train_args['log_run_name'] = 'student_' + time.strftime(
        '%Y%m%d_%H%M%S', time.localtime(time.time())
    )

    train_args['hps_list'] = [
        'lr',
        'batch_size',
    ]

    start_ray_tuning(train_args, train_student_syn_img)

    return os.path.join(train_args['log_run_name'], train_args['trial_name'])


def train(
    dataset,
    dataset_dir,
    lr_img_stage1,
    batch_size_stage1,
    batch_num_stage1,
    lambda_moment_stage1,
    iterations_img_stage1,
    slack_stage1,
    lr_img_stage2,
    batch_size_stage2,
    batch_num_stage2,
    lambda_moment_stage2,
    iterations_img_stage2,
    slack_stage2,
    lr_student,
    batch_size_student,
    iterations_student,
    model_dir,
    seed,
):
    img_dir_stage1 = train_image_stage1(
        dataset,
        batch_size_stage1,
        batch_num_stage1,
        lr_img_stage1,
        lambda_moment_stage1,
        iterations_img_stage1,
        slack_stage1,
        model_dir,
        seed,
    )

    img_dir_stage2 = train_image_stage2(
        dataset,
        batch_size_stage2,
        batch_num_stage2,
        lr_img_stage2,
        lambda_moment_stage2,
        iterations_img_stage2,
        slack_stage2,
        model_dir,
        img_dir_stage1,
        seed,
    )

    student_dir = train_student(
        dataset,
        dataset_dir,
        batch_size_student,
        lr_student,
        iterations_student,
        model_dir,
        img_dir_stage1,
        img_dir_stage2,
        seed,
    )

    return student_dir

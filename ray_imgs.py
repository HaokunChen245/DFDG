import argparse
import time
import os

import ray
from ray import tune

from train_func_image import (train_one_setting_multi_teacher,
                              train_one_setting_single_teacher)
from utils import *


def get_config(
    dataset,
    stage,
    teacher_snapdir,
    stage1_snapdir,
    trial_seed,
    teacher_backbone,
    batch_size,
    img_size,
):
    train_args = {}
    train_args['stage'] = stage
    train_args['dataset'] = dataset
    train_args['batch_size'] = batch_size
    train_args['num_class'] = get_class_number(train_args['dataset'])
    train_args['source_domains'] = get_source_domains(train_args['dataset'])
    train_args['teacher_snapdir'] = os.path.join(teacher_snapdir, dataset)
    train_args['trial_seed'] = trial_seed
    train_args['scale_1BN'] = 1
    train_args['source_domain_A'] = tune.grid_search(train_args['source_domains'])
    train_args['multi_resolution'] = False
    train_args['more_slack_1st_var'] = True
    train_args['use_fake_stats_init'] = False
    train_args['stage1_snapdir'] = stage1_snapdir
    if dataset=='Digits':
        train_args['img_size'] = 32
        train_args['teacher_backbone'] = 'resnet18'
        train_args['batch_numbers'] = 1
        train_args['batch_size'] = 16

        if train_args['stage']==2:
            train_args['max_iters_train'] = 100
            train_args['lr'] = tune.grid_search([0.01])
            train_args['Ws'] = tune.grid_search([10])
            train_args['slack'] = tune.grid_search([70])

        else:
            train_args['max_iters_train'] = 200
            train_args['lr'] = tune.grid_search([0.1])
            train_args['Ws'] = tune.grid_search([1])
            train_args['slack'] = tune.grid_search([0])

    train_args['hps_list'] = ['lr', 'Ws', 'slack', 'source_domain_A']

    return train_args

def main():

    parser = argparse.ArgumentParser(description='PyTorch image generation code')
    parser.add_argument('--dataset', type=str, required=True, help="name of the dataset.")
    parser.add_argument('--log_snapdir', type=str, help="directory of the logging")
    parser.add_argument('--stage1_snapdir', type=str, help="directory of the teacher models.")
    parser.add_argument('--teacher_snapdir', type=str, required=True, help="directory of the teacher models.")
    parser.add_argument('--teacher_backbone', type=str, help="backbone of teacher network")
    parser.add_argument('--batch_size', type=int, help='training batch size')
    parser.add_argument('--img_size', type=int, help='image size of generation')
    parser.add_argument('--trial_seed', type=int, default=0, help='random of seed of the program')
    parser.add_argument('--stage', type=int, default=1, help='image generation stage')
    args = parser.parse_args()

    on_autoscale_cluster = False
    ray.init()
    resources_per_trial = {"gpu": 1, "cpu": 3}
    logs_local_dir = args.log_snapdir

    train_config = get_config(
            dataset = args.dataset,
            stage = args.stage,
            teacher_snapdir = args.teacher_snapdir,
            stage1_snapdir = args.stage1_snapdir,
            trial_seed = args.trial_seed,
            teacher_backbone = args.teacher_backbone,
            batch_size = args.batch_size,
            img_size = args.img_size,
        )

    train_config['log_root_dir'] = logs_local_dir
    train_config['log_run_name'] = 'image_' + time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))

    if train_config['stage']==1:
        training = train_one_setting_single_teacher
    else:
        training = train_one_setting_multi_teacher

    try:
        tune.run(
                training,
                name = train_config['log_run_name'],
                config= train_config,
                resources_per_trial= resources_per_trial,
                local_dir = train_config['log_root_dir'],
                trial_name_creator=trial_name_string,
                trial_dirname_creator=trial_name_string,

                #to resume one failed trial, specify the name and set the resume argument
                #use "PROMPT" to restart from local dir
#                 resume = "PROMPT",
#                 name = "aws_folder"
        )
    finally:
        if on_autoscale_cluster:
            print("Downscaling cluster in 2 minutes...")
            time.sleep(120)  # Wait for any syncing to complete.

if __name__=='__main__':
    main()

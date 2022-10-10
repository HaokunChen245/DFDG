import argparse
import random
import time
import os

import ray
from ray import tune

from train_func_student import train_one_setting_student_image
from utils import *


def prepare_string():
    seed = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    sa = []
    for _ in range(32):
        sa.append(random.choice(seed))
    salt = ''.join(sa)
    return salt

def get_config(
    dataset,
    dataset_dir,
    stage1_snapdir,
    stage2_snapdir,
    teacher_snapdir,
    student_snapdir,
    trial_seed,
    img_size,
    max_iters_train,
    model_selection,
    teacher_backbone,
    student_backbone,
):
    train_args = {}
    train_args['dataset'] = dataset
    train_args['use_real_datas'] = False
    train_args['img_size'] = img_size
    train_args['dataset_dir'] = os.path.join(dataset_dir, dataset)
    train_args['stage1_snapdir'] = stage1_snapdir
    train_args['stage2_snapdir'] = stage2_snapdir
    train_args['teacher_snapdir'] = os.path.join(teacher_snapdir, dataset)
    train_args['student_snapdir'] = student_snapdir
    train_args['trial_seed'] = trial_seed
    train_args['max_iters_train'] = max_iters_train
    train_args['teacher_backbone'] = teacher_backbone
    train_args['student_backbone'] = student_backbone

    train_args['num_class'] = get_class_number(train_args['dataset'])
    train_args['source_domains'] = get_source_domains(train_args['dataset'])

    assert model_selection in ['target_domain_valset', 'source_domains_valset']
    train_args['model_selection'] = model_selection

    train_args['target_domain'] = tune.grid_search(train_args['source_domains'])
    train_args['input_dims'] = 1024

    if dataset=='Digits':
        train_args['max_iters_train'] = 20
        train_args['img_size'] = 32
        train_args['lr'] = tune.grid_search([0.01])
        train_args['batch_size'] = tune.grid_search([256])
        train_args['teacher_backbone'] = 'resnet18'
        train_args['student_backbone'] = 'resnet18'
        train_args['eval_batch_size'] = 2048
        train_args['multi_domain_prob'] = 2 #no multi domain, when >1 then no multi_domain
        train_args['use_generator'] = False
        train_args['class_spec'] = True

    train_args['hps_list'] = ['lr', 'batch_size', 'target_domain', 'multi_domain_prob']

    #switch to tune grid search
    return train_args

def main():
    parser = argparse.ArgumentParser(description='PyTorch Teacher Network training code')
    parser.add_argument('--dataset', type=str, help="name of the dataset.")
    parser.add_argument('--dataset_dir', type=str, help="directory of the dataset.")
    parser.add_argument('--log_snapdir', type=str, help="directory of the logging")
    parser.add_argument('--stage1_snapdir', type=str, help="directory of the dataset.")
    parser.add_argument('--stage2_snapdir', type=str, help="directory of the dataset.")
    parser.add_argument('--teacher_snapdir', type=str, help="directory of the teacher models.")
    parser.add_argument('--student_snapdir', type=str, help="directory of the student models.")
    parser.add_argument('--max_iters_train', type=int, default=10, help='total iterations of training')
    parser.add_argument('--teacher_backbone', type=str, default='resnet50', help="backbone of teacher network")
    parser.add_argument('--student_backbone', type=str, default='resnet50', help="backbone of student network")
    parser.add_argument('--img_size', type=int, default=224, help="input image size.")
    parser.add_argument('--trial_seed', type=int, default=0, help='random of seed of the program')
    parser.add_argument('--model_selection', type=str, default='target_domain_valset', help='method of selecting the final model')
    args = parser.parse_args()
    on_autoscale_cluster = False
    ray.init()
    resources_per_trial = {"gpu": 1, "cpu": 3}
    logs_local_dir = args.log_snapdir

    train_config = get_config(
            dataset = args.dataset,
            dataset_dir = args.dataset_dir,
            stage1_snapdir = args.stage1_snapdir,
            stage2_snapdir = args.stage2_snapdir,
            teacher_snapdir = args.teacher_snapdir,
            student_snapdir = args.student_snapdir,
            trial_seed = args.trial_seed,
            teacher_backbone = args.teacher_backbone,
            student_backbone = args.student_backbone,
            img_size = args.img_size,
            max_iters_train = args.max_iters_train,
            model_selection = args.model_selection,
    )

    train_one_setting = train_one_setting_student_image

    train_config['log_root_dir'] = logs_local_dir
    train_config['log_run_name'] = 'student_' + time.strftime('%Y%m%d_%H%M%S',time.localtime(time.time()))

    try:
        tune.run(
                train_one_setting,
                name = train_config['log_run_name'],
                config= train_config,
                resources_per_trial= resources_per_trial,
                local_dir = train_config['log_root_dir'],
                raise_on_failed_trial=False if on_autoscale_cluster else True,
                trial_name_creator=trial_name_string,
                trial_dirname_creator=trial_name_string,

                #resume = 'ERRORED_ONLY',
                #name = 'train_one_setting_2021-09-09_20-10-52',
        )
    finally:
        if on_autoscale_cluster:
            print("Downscaling cluster in 2 minutes...")
            time.sleep(120)  # Wait for any syncing to complete.

if __name__=='__main__':
    main()

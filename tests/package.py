"""
@copyright Copyright (c) Siemens AG, 2022
@author Haokun Chen <haokun.chen@siemens.com>
SPDX-License-Identifier: Apache-2.0
"""
import sys

sys.path.append("..")

from dfdg.download.dataset_download import download_dataset
from dfdg.download.teacher_download import download_teacher
from dfdg.training.train import train
from dfdg.evaluation.evaluate import evaluate

# from dfdg.evaluation.evaluate import evaluate

DATASET_NAME = "Digits"
# Download dataset.
download_dataset(DATASET_NAME, "./data")

# Download pretrained teacher models.
download_teacher("./models")

# # Train a student model for each domain using the pretrained teacher models.

student_dir = train(
    dataset=DATASET_NAME,
    dataset_dir="./data",
    lr_img_stage1=0.1,
    batch_size_stage1=128,
    batch_num_stage1=2,
    lambda_moment_stage1=1,
    iterations_img_stage1=200,
    slack_stage1=10,
    lr_img_stage2=0.1,
    batch_size_stage2=128,
    batch_num_stage2=2,
    lambda_moment_stage2=1,
    iterations_img_stage2=200,
    slack_stage2=10,
    lr_student=0.1,
    batch_size_student=256,
    iterations_student=5,
    model_dir="./models",
    seed=1,
)


# Evaluate the student models.
result = evaluate(
    dataset=DATASET_NAME,
    dataset_dir="./data",
    model_dir="./models",
    student_dir=student_dir,
    batch_size=1024,
)

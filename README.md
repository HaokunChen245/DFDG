# DFDG

![Teaser image](docs/assets/method.png)

This repo contains code for our paper:

> [**Towards Data-Free Domain Generalization**](https://arxiv.org/pdf/2110.04545.pdf)<br>
> [Ahmed Frikha](https://scholar.google.de/citations?user=NiarLswAAAAJ&hl=en)\*, [Haokun Chen](https://scholar.google.com/citations?user=ilbqzDwAAAAJ&hl=en)\*, [Denis Krompass](https://www.dbs.ifi.lmu.de/~krompass/), [Volker Tresp](https://www.dbs.ifi.lmu.de/~tresp/)<br>
> LMU, Siemens<br>
> ACML 2022

## Installation

1.  OPTIONAL (but recommended): Create a virtual environment using Python's builtin [venv](https://docs.python.org/3/library/venv.html) ...

    ```bash
    $ python -m venv .venv
    $ source .venv/bin/activate
    ```

    ... or [Conda](https://conda.io):

    ```bash
    $ conda create -n ENV_NAME python=X.Y
    $ conda activate ENV_NAME
    ```

1.  Install Poetry and install dependencies:
    ```bash
    $ curl -sSL https://install.python-poetry.org | python3 -
    $ poetry install
    ```

## Quickstart

### Library

```python
from dfdg.download.dataset_download import download_dataset
from dfdg.download.teacher_download import download_teacher
from dfdg.training.train import train
from dfdg.evaluation.evaluate import evaluate

# Download dataset.
download_dataset(DATASET_NAME, './data')

# Download pretrained teacher models.
download_teacher('./model')

# Train a student model for each domain using the pretrained teacher models.
student_dir = train(
    dataset=DATASET_NAME,
    dataset_dir='./data',
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
    model_dir='./models',
    seed=1,
)

# Evaluate the student models.
result = evaluate(
    dataset=DATASET_NAME,
    dataset_dir='./data',
    model_dir='./models',
    student_dir=student_dir,
    batch_size=1024,
)
print(result)
```

### CLI

1.  Download the dataset and the teacher models

    ```bash
    $ python -m dfdg dataset_download
    $ python -m dfdg teacher_download
    ```

1.  Train a model:

    ```bash
    $ python -m dfdg train
    ```

1.  Evaluate a model:

    ```bash
    $ python -m dfdg evaluate
    ```

## Citation

If you use our code in your research or wish to refer to the results published in our work, please cite our work with the following BibTeX entry.

```
@article{frikha2021towards,
  title={Towards Data-Free Domain Generalization},
  author={Frikha, Ahmed and Chen, Haokun and Krompa{\ss}, Denis and Runkler, Thomas and Tresp, Volker},
  journal={arXiv preprint arXiv:2110.04545},
  year={2021}
}
```


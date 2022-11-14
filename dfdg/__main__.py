'''
@copyright Copyright (c) Siemens AG, 2022
@author Haokun Chen <haokun.chen@siemens.com>
SPDX-License-Identifier: Apache-2.0
'''

"""Command-line interface."""

import argparse
from dfdg.clis.training_clis import subparser_train
from dfdg.clis.download_clis import subparser_dataset_download
from dfdg.clis.download_clis import subparser_teacher_download
from dfdg.clis.evaluation_clis import subparser_evaluation

def main(args=None):
    """CLI entrypoint.

    Parameters
    ----------
    args : list of str, optional
        List of command-line arguments. Defaults to `None`.

    """
    parser = argparse.ArgumentParser('dfdg')
    parser.set_defaults(func=lambda _: parser.print_help())

    subparsers = parser.add_subparsers()
    subparser_dataset_download(subparsers)
    subparser_teacher_download(subparsers)
    subparser_train(subparsers)
    subparser_evaluation(subparsers)

    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)

if __name__ == '__main__':
    main()

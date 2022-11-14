''' 
@copyright Copyright (c) Siemens AG, 2022
@author Haokun Chen <haokun.chen@siemens.com>
SPDX-License-Identifier: Apache-2.0
'''

"""Command-line interface tests."""
import pytest

from dfdg.__main__ import main


def test_help():
    """Test calling the CLI with the --help exits with exit code 0."""
    with pytest.raises(SystemExit) as e:
        main(['--help'])
        assert e.code == 0

# Contributing

We welcome contributions in several forms, e.g.:

- Documenting
- Testing
- Coding
- etc.

Please read [14 Ways to Contribute to Open Source without Being a Programming Genius or a Rock Star](http://blog.smartbear.com/programming/14-ways-to-contribute-to-open-source-without-being-a-programming-genius-or-a-rock-star/).

## Code

This project uses the following tools to ensure code quality:

- [Black](https://github.com/python/black) for automatic code formatting
- [isort](https://github.com/timothycrosley/isort) for reordering imports
- [Pylint](https://www.pylint.org) for code linting (static code analysis)
- [pydocstyle](https://github.com/PyCQA/pydocstyle) for docstring validation
- [mypy](http://mypy-lang.org) for static type checking
- [pre-commit](https://github.com/pre-commit/pre-commit) for ensuring code quality before committing changes to the Git repository

### Prerequisites

> NOTE: You may skip these steps if you have already performed them before.

1.  Install [Conda](https://conda.io) as a virtual environment manager with cross-platform support for installing arbitrary Python interpreter versions. Follow the [instructions for installing Miniconda](https://docs.conda.io/en/latest/miniconda.html).

1.  Install [Poetry](https://github.com/python-poetry/poetry):

    ```bash
    $ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python
    ```

1.  Configure Poetry to _not_ create a new virtual environment, i.e. to reuse the Conda environment instead:

    ```bash
    $ poetry config virtualenvs.create 0
    ```

### Development

1.  Create a new Conda virtual environment:

    ```bash
    $ conda create -n ENV_NAME python=X.Y
    $ conda activate ENV_NAME
    ```

    > NOTE: It is _imperative_ (and best practices) to create a fresh virtual environment for each project to avoid package conflicts! _DO NOT_ reuse an existing virtual environment from another project!
    >
    > NOTE: It is recommended to use the _lowest_ Python version that this project supports in order to avoid using language features that are only available in higher versions. For instance, if the project supports Python 3.6+, use Python 3.6 for development.

1.  Install runtime and development dependencies:

    > NOTE: If you're using GPU acceleration, it is recommended to install the necessary versions (check TensorFlow's documentation for this information) of [CUDA](https://developer.nvidia.com/cuda-zone) and [cuDNN](https://developer.nvidia.com/cudnn) using Conda:
    >
    > ```bash
    > $ conda install cudatoolkit~=X.Y cudnn~=A.B
    > ```

    ```bash
    $ poetry install -E cpu

    # or with GPU support

    $ poetry install -E gpu
    ```

1.  Install Git pre-commit hooks to enable code quality checks when making a Git commit:

    ```bash
    $ pre-commit install
    ```

1.  Make changes and don't forget to implement tests.

    If you need to add or remove runtime dependencies, edit `pyproject.toml` by adding or removing a package in the section `[tool.poetry.dependencies]`. Similarly, development dependencies are located in the section `[tool.poetry.dev-dependencies]`. Finally, run `poetry lock` in order to update the `poetry.lock` file.

    > NOTE: This package is declared compatible with Python 3.8+ (<4). Please ensure forward compatibility with new minor Python releases. To automate compatibility checks, extend the test suite in [`tox.ini`](./tox.ini) and the CI pipeline in [`.gitlab-ci.yml`](./.gitlab-ci.yml) to new Python versions as they become available.

1.  Run tests quickly during active development:

    ```bash
    $ pytest
    ```

1.  OPTIONAL: Run the full test suite in isolated environments including code quality tests:

    ```bash
    $ tox
    ```

    > NOTE: If package dependencies have changed since the last run of `tox`, run
    >
    > ```bash
    > $ tox -r
    > ```
    >
    > to force recreation of virtual environments.

1.  OPTIONAL: Build and view documentation:

    ```bash
    $ mkdocs build
    ```

    Serve the documentation locally:

    ```bash
    $ mkdocs serve
    ```

    Then view the documentation in the browser at [http://localhost:8000](http://localhost:8000).

1.  Commit changes:

    ```bash
    $ git add .
    $ git commit
    isort....................................................................Passed
    black....................................................................Passed
    pydocstyle...............................................................Passed
    pylint...................................................................Passed
    mypy.....................................................................Passed
    poetry...................................................................Passed
    Trim Trailing Whitespace.................................................Passed
    Fix End of Files.........................................................Passed
    ```

    > NOTE: Please run the above Git commands from inside the Conda virtual environment to benefit from Git pre-commit hooks.
    >
    > NOTE: (Some) GUI-based Git tools don't seem to support pre-commit hook execution.

    See [Git guidelines](#git-guidelines) for details about the Git workflow.

1.  OPTIONAL: Build distribution files (`.tar.gz` and `.whl`):

    ```bash
    $ poetry build
    ```

1.  OPTIONAL: Build and view documentation:

    ```bash
    $ mkdocs build
    ```

    Serve the documentation locally:

    ```bash
    $ mkdocs serve
    ```

    Then view the documentation in the browser at [http://localhost:8000](http://localhost:8000).

## Git guidelines

### Workflow

We currently recommend the [Feature Branch Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow).

The mentioned links from Atlassian are the recommended documentation to read and understand the Git workflows.

### Git commit

The cardinal rule for creating good commits is to ensure there is only one
"logical change" per commit. There are many reasons why this is an important
rule:

- The smaller the amount of code being changed, the quicker & easier it is to review & identify potential flaws.
- If a change is found to be flawed later, it may be necessary to revert the
  broken commit. This is much easier to do if there are not other unrelated code changes entangled with the original commit.
- When troubleshooting problems using Git's bisect capability, small well defined changes will aid in isolating exactly where the code problem was introduced.
- When browsing history using Git annotate/blame, small well-defined changes also aid in isolating exactly where a piece of code came from and why it exists.

Things to avoid when creating commits:

- Mixing whitespace changes with functional code changes.
- Mixing two unrelated functional changes.
- Sending large new features in a single giant commit.

Please read [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/) for more details.

## Versioning

This project conforms with [Semantic Versioning 2.0.0](https://semver.org).
The version set in the file `pyproject.toml` is a placeholder and should not be changed directly as any change is overridden by the CI scripts.
To publish a new version, add a tag with the desired version number to the `master` branch, e.g. a tag called `v1.2.3`.
By default, the CI pipeline will then publish this version to the local PyPi-registry on `code.siemens.com`.

## FAQ

### Why does VS Code not sort my imports when saving `.py` files?

Set the following snippet in your VS Code settings:

```json
"[python]": {
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

# Contributing to MESS

We are interested in hearing any and all feedback so feel free to raise any questions,
bugs encountered, or enhancement requests as
[Issues](https://github.com/valence-labs/mess/issues).

## Setting up a development environment

The following assumes that you have already set up an install of conda and that the
conda command is available on your system path. Refer to your preferred conda installer:

- [miniforge installation](https://github.com/conda-forge/miniforge#install)
- [conda installation documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

1. Create a new conda environment with the minimum python version required:

   ```bash
   conda create -n mess python=3.11
   ```

1. Install all required packages for developing MESS:

   ```bash
   pip install -e .[dev]
   ```

1. Install the pre-commit hooks

   ```bash
   pre-commit install
   ```

1. Create a feature branch, make changes, and when you commit them the pre-commit hooks
   will run.

   ```bash
   git checkout -b feature
   ...
   git push --set-upstream origin feature
   ```

   The last command will print a link that you can follow to open a PR.

## Testing

Run all the tests using `pytest`

```bash
pytest
```

## Building Documentation

From the project root, you can build the documentation with:

```bash
jupyter-book build docs
```

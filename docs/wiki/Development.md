# Development

## Code layout

```
workspace/          the program (run from here)
tests/              pytest tests (no PyTorch needed)
notebooks/          the Colab notebook
.github/            CI workflow and Dependabot config
requirements.txt    runtime dependencies
requirements-dev.txt  test and lint tools
setup.cfg           flake8 and pytest settings
```

## Running the checks

```bash
pip install -r requirements-dev.txt
flake8 workspace tests
pytest
```

There are 74 tests. PyTorch, torchvision and the classifier are replaced by
small fakes, so the suite runs in seconds and never downloads weights. The
tests cover every pipeline step, the printed output, the CSV export, the
charts (including a check that scatter labels don't overlap), an end-to-end
run of `check_images.py`, and a regression test for each bug fixed.

## CI

GitHub Actions runs full flake8, compiles every module and runs the tests on
Python 3.10, 3.12 and 3.13 on every push and pull request. Dependabot opens a
weekly pull request for newer pip packages and GitHub Actions; ranges in
`requirements.txt` only change when a release falls outside them.

## Branches

- `master`: the current code
- `original-code`: the untouched original submission

## Conventions

- Keep Udacity's results-list format (`[pet_label, classifier_label, match,
  pet_is_dog, classifier_is_dog]`): Udacity's check functions and the project
  rubric rely on it.
- Output wording that `print_model_tables.py` parses (the summary header and
  statistic labels) is defined once in `print_results.py`; change it there.
- Open changes as pull requests; CI must pass before merging.

Ideas for further work are in the
[TODO list](https://github.com/thomasmd321/AI_Programming_with_Python_Project_1/blob/master/TODO.md).

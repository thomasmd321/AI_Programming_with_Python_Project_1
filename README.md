# AI Programming with Python – Project 1: Classifying Pet Images

[![CI](https://github.com/thomasmd321/AI_Programming_with_Python_Project_1/actions/workflows/ci.yml/badge.svg)](https://github.com/thomasmd321/AI_Programming_with_Python_Project_1/actions/workflows/ci.yml)

Project 1 of Udacity's *AI Programming with Python* Nanodegree. The program
uses convolutional neural networks (CNNs) pretrained on ImageNet to classify
pet images. It checks whether each model correctly identifies dogs versus
non-dogs and names the right dog breed, then compares three architectures:
**ResNet-18**, **AlexNet** and **VGG-16**.

The true identity of each pet comes from the image filename
(`Boston_terrier_02259.jpg` → `boston terrier`). The program compares that
label with the CNN's prediction.

## Results

Results on the 40 images in `pet_images/` (30 dogs, 10 non-dogs), from the
saved `*_pet-images.txt` files:

| CNN model | % Not-a-dog correct | % Dogs correct | % Breeds correct | % Match labels |
|-----------|--------------------:|---------------:|-----------------:|---------------:|
| ResNet    |  90.00 | 100.00 | 90.00 | 82.50 |
| AlexNet   | 100.00 | 100.00 | 80.00 | 75.00 |
| VGG       | 100.00 | 100.00 | 93.33 | 87.50 |

**VGG** does best overall: it has the highest breed and label-match accuracy,
and, like AlexNet, it labels every non-dog correctly.

Results on the 7 images in `uploaded_images/` (2 dogs, 5 non-dogs), from the
`*_uploaded-images.txt` files:

| CNN model | % Not-a-dog correct | % Dogs correct | % Breeds correct | % Match labels |
|-----------|--------------------:|---------------:|-----------------:|---------------:|
| ResNet    | 60.00 | 100.00 | 100.00 | 42.86 |
| AlexNet   | 60.00 | 100.00 |   0.00 | 14.29 |
| VGG       | 60.00 | 100.00 | 100.00 | 42.86 |

The two Labradoodle photos count as "not a dog" errors. A labradoodle isn't
in `dognames.txt` (it's not an ImageNet class), so its pet label is treated
as not-a-dog, while every model correctly calls it a dog.

> **About the saved output files:** they were regenerated after the bug
> fixes in [TODO.md](TODO.md) by replaying the CNN labels recorded in the
> original runs through the fixed code. The classifications and runtimes
> come from those original runs. The old uploaded-images runs also included
> two images (`dog_01.jpg`, `dog_02.jpg`) that were never committed, so the
> regenerated files cover the 7 committed images.

## Project layout

```
.
├── README.md
├── TODO.md                      # Code improvements: done and still open
├── requirements.txt             # Runtime dependencies (PyTorch, Pillow)
├── requirements-dev.txt         # Test/lint tools
├── setup.cfg                    # flake8 and pytest settings
├── .github/workflows/ci.yml     # CI: lint + unit tests
├── tests/                       # pytest tests (no PyTorch needed)
└── workspace/
    ├── check_images.py          # Main program – runs the whole pipeline
    ├── get_input_args.py        # 1. Parses --dir, --arch, --dogfile
    ├── get_pet_labels.py        # 2. Builds pet labels from filenames
    ├── classify_images.py       # 3. Runs the CNN and compares labels
    ├── adjust_results4_isadog.py# 4. Flags whether each label is a dog
    ├── calculates_results_stats.py # 5. Computes counts and percentages
    ├── print_results.py         # 6. Prints the summary and misclassifications
    ├── print_model_tables.py    # Comparison table across the 3 models
    ├── classifier.py            # CNN wrapper (Udacity, revised)
    ├── print_functions_for_lab_checks.py # Step-by-step check printers (Udacity)
    ├── test_classifier.py       # Demo of classifier() (Udacity)
    ├── run_models_batch.sh      # Runs all 3 models on pet_images/, then prints the table
    ├── run_models_batch_uploaded.sh # Same for uploaded_images/
    ├── dognames.txt             # Every dog name the classifier can return
    ├── imagenet1000_clsid_to_human.txt # ImageNet class index → name
    ├── pet_images/              # 40 course-supplied test images
    ├── uploaded_images/         # Extra user-supplied test images
    └── *_pet-images.txt, *_uploaded-images.txt # Saved output of each run
```

### How the pipeline works

`check_images.py` passes a single **results dictionary** through each step.
It's keyed by image filename, and each step appends to the value list:

| Index | Added by | Meaning |
|------:|----------|---------|
| 0 | `get_pet_labels` | pet label from the filename |
| 1 | `classify_images` | classifier label (lower case) |
| 2 | `classify_images` | 1 if the pet label appears as whole words in the classifier label, else 0 |
| 3 | `adjust_results4_isadog` | 1 if the pet label is a dog |
| 4 | `adjust_results4_isadog` | 1 if the classifier label is a dog |

`calculates_results_stats` then summarizes the dictionary as counts (`n_*`)
and percentages (`pct_*`), and `print_results` prints them.

## Requirements

- Python 3.7+
- [PyTorch](https://pytorch.org/), torchvision and Pillow

```bash
pip install -r requirements.txt
```

Pretrained weights are downloaded automatically the first time each model is
used. Only the model named by `--arch` is loaded.

## Usage

Run the commands from inside `workspace/`: the default `--dir` and
`--dogfile` values, and the batch scripts, use paths relative to it.

```bash
cd workspace

# Classify pet_images/ with one model
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt

# Run all three models and save the output to *_pet-images.txt
sh run_models_batch.sh

# Same for your own images in uploaded_images/
sh run_models_batch_uploaded.sh

# Reprint the comparison tables for both folders from the saved outputs
python print_model_tables.py
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dir` | `pet_images/` | Image folder (hidden and non-image files are skipped) |
| `--arch` | `vgg` | `resnet`, `alexnet` or `vgg` |
| `--dogfile` | `dognames.txt` | File of valid dog names |

## Tests and CI

The tests cover every pipeline step plus the printed output. The classifier
and PyTorch are replaced by small fakes, so no weights are downloaded and the
suite runs in under a second:

```bash
pip install -r requirements-dev.txt
flake8 workspace tests
pytest
```

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs
full flake8, compiles every module and runs the tests on each push and pull
request.

## Branches

- `master`: cleaned-up, commented code with tests and CI.
- `original-code`: a snapshot of the project exactly as it was submitted,
  before cleanup.

## Author

Thomas Stewart. Starter code and helper modules by Udacity.

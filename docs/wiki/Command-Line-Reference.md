# Command Line Reference

## `check_images.py`

```bash
python check_images.py [--dir DIR] [--arch ARCH] [--dogfile FILE] [--topk K] [--csv PATH]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dir` | `pet_images/` | Image folder; a trailing `/` is optional |
| `--arch` | `vgg` | `resnet`, `alexnet`, `vgg`, `resnet50` or `efficientnet` |
| `--dogfile` | `dognames.txt` | File of dog names |
| `--topk` | `0` (off) | For images whose labels don't match, print the model's top K guesses with their confidence (0–1000) |
| `--csv` | none | Save one row per image to this CSV file |

Examples:

```bash
python check_images.py --arch resnet50 --topk 3
python check_images.py --dir uploaded_images/ --arch vgg --csv vgg.csv
```

## Batch scripts

`run_models_batch.sh` runs every model on `pet_images/` and
`run_models_batch_uploaded.sh` on `uploaded_images/`. For each model they
write:

- `<model>_pet-images.txt` (or `_uploaded-images.txt`): the full printed output
- `<model>_pet-images.csv` (or `_uploaded-images.csv`): per-image results, with `--topk 3`

They then print the comparison tables. They stop at the first model that
fails.

## `print_model_tables.py`

```bash
python print_model_tables.py
```

Prints one table per image folder from the saved `.txt` files, with columns:
% not-a-dog correct, % dogs correct, % breeds correct, % labels matched,
parameters (millions), seconds per image and total runtime. Missing files are
left out; a file without a complete summary (a run cut short) is skipped with
a warning. A value that isn't in a file shows as `n/a`.

## CSV columns

| Column | Meaning |
|--------|---------|
| `model` | Architecture used |
| `filename` | Image file |
| `pet_label` | Label from the filename |
| `classifier_label` | The model's top guess |
| `confidence` | Probability of the top guess (0–1) |
| `labels_match` | 1 if the labels match |
| `pet_is_dog` | 1 if the pet label is a dog |
| `classifier_is_dog` | 1 if the classifier label is a dog |
| `top_guesses` | e.g. `walker hound (55.0%); beagle (40.0%)` |

## Other scripts

- `test_classifier.py`: Udacity's demo; classifies one collie with VGG.

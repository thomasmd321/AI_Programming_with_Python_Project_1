# How It Works

`check_images.py` runs six steps, one module per step. It passes a single
**results dictionary** through them, keyed by image filename; each step
appends to the list stored for each image.

| Step | Module | Adds | Meaning |
|-----:|--------|------|---------|
| 1 | `get_input_args.py` | | Reads the command-line options |
| 2 | `get_pet_labels.py` | index 0 | Pet label from the filename |
| 3 | `classify_images.py` | index 1 | Classifier label: the model's top guess, lower case |
| 3 | `classify_images.py` | index 2 | 1 if the labels match, else 0 |
| 4 | `adjust_results4_isadog.py` | index 3 | 1 if the pet label is a dog |
| 4 | `adjust_results4_isadog.py` | index 4 | 1 if the classifier label is a dog |
| 5 | `calculates_results_stats.py` | | Counts and percentages |
| 6 | `print_results.py` | | Prints the summary and mistakes |

After each step, Udacity's `print_functions_for_lab_checks.py` prints a check
of what the step produced.

## Step 2: pet labels from filenames

The label is the filename without its extension, lower-cased and split on
`_`, keeping only the purely alphabetic words:
`German_shepherd_dog_04890.jpg` → `german shepherd dog`. Files are processed
in sorted order.

## Step 3: classifying and matching

`classifier.predict_batch()` runs the images through the CNN 16 at a time and
returns each image's top guesses with their probabilities. Images are first
turned upright using their EXIF orientation tag and converted to RGB. The top
guess is the classifier label, which can list several names for one class,
e.g. `dalmatian, coach dog, carriage dog`.

The labels **match** when the pet label appears in the classifier label as
**whole words**: `cat` matches `tabby, tabby cat`, but not
`polecat, fitch, foulmart, foumart`.

## Step 4: is it a dog?

`dognames.txt` lists every ImageNet dog class, one per line, plus
`labradoodle`. A label is a dog if it matches a whole line or any
comma-separated name on a line.

## Step 5: statistics

| Statistic | Meaning |
|-----------|---------|
| `pct_correct_dogs` | Dog images the classifier also called dogs |
| `pct_correct_notdogs` | Non-dog images the classifier also called non-dogs |
| `pct_correct_breed` | Dog images that were called dogs *and* had matching labels |
| `pct_match` | All images whose labels matched |

A percentage whose group is empty (e.g. no dog images) is reported as 0%.
These rules agree with Udacity's own check in `print_functions_for_lab_checks.py`.

## Step 6 and beyond

`print_results.py` prints the counts, the percentages, and the images the
model got wrong (dog/not-dog mistakes and breed mistakes). `check_images.py`
then prints the model's size and its classification time per image; the
model is loaded before the timer starts, so the time covers classification
only. With `--topk` and `--csv` it also prints the top guesses for mismatches
and saves a CSV (see [[Command Line Reference]]).

`print_model_tables.py` reads the saved output of every model and prints a
side-by-side table; `plot_results.py` draws the charts in the notebook.

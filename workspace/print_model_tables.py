#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/print_model_tables.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: May 1, 2020
# REVISED DATE: September 23, 2026
# PURPOSE: Builds a side-by-side comparison table of the CNN architectures
#          by reading the output files written by run_models_batch.sh
#          (pet_images) or run_models_batch_uploaded.sh (uploaded_images).
#          Models whose output file doesn't exist yet are left out.
#
# Usage: python print_model_tables.py
##
import logging
from os import path

from get_input_args import ARCHITECTURES
from print_results import COUNT_LABELS, PCT_LABELS, SPEED_LABELS, SUMMARY_HEADER


def output_files(suffix):
    """Output file names for every architecture, e.g. 'vgg_pet-images.txt'."""
    return ['{}_{}.txt'.format(model, suffix) for model in ARCHITECTURES]


PET_IMAGE_FILES = output_files('pet-images')
UPLOADED_IMAGE_FILES = output_files('uploaded-images')

# The last line of each output file, e.g. "** Total Elapsed Runtime: 0:00:32".
RUNTIME_LABEL = "** Total Elapsed Runtime"


def _percent(value):
    return "{:.2f}%".format(value)


def _runtime(seconds):
    return "{:d}:{:02d}:{:02d}".format(seconds // 3600, (seconds % 3600) // 60, seconds % 60)


# Table columns, left to right: (statistic key, column heading, formatter).
# A model whose output file lacks a statistic shows "n/a".
COLUMNS = (('pct_correct_notdogs', '% Not-a-dog Correct', _percent),
           ('pct_correct_dogs', '% Dogs Correct', _percent),
           ('pct_correct_breed', '% Breeds Correct', _percent),
           ('pct_match', '% Match Labels', _percent),
           ('params_m', 'Params (M)', "{:.1f}".format),
           ('sec_per_image', 'Sec/Image', "{:.3f}".format),
           ('runtime', 'Total Runtime', _runtime))
ROW_FORMAT = "{:<24} |" + " |".join("{:<%d}" % max(len(title), 8) for _, title, _ in COLUMNS)


def parse_results_file(filename):
    """
    Reads the model name and the statistics that print_results() wrote into a
    saved check_images.py output file.
    Parameters:
      filename - path to the output file (string)
    Returns:
      (model, stats) - model name in lower case (string) and a dictionary of
                       statistic key -> value (int for counts and the runtime
                       in seconds, float for the rest)
    """
    labels = {label: key for key, label in
              {**COUNT_LABELS, **PCT_LABELS, **SPEED_LABELS}.items()}
    model = None
    stats = {}
    with open(filename, 'r') as infile:
        for line in infile:
            line = line.strip()
            # "*** Results Summary for CNN Model Architecture VGG ***"
            if line.startswith(SUMMARY_HEADER):
                model = line[len(SUMMARY_HEADER):].strip(' *').lower()
                continue
            # "% Correct Dogs      : 100.00". Only read after the header:
            # the earlier lab-check output has similar-looking lines.
            label, sep, value = line.partition(':')
            if model is None or not sep:
                continue
            if label.strip() in labels:
                key = labels[label.strip()]
                stats[key] = int(value) if key.startswith('n_') else float(value)
            elif label.strip() == RUNTIME_LABEL:
                hours, minutes, seconds = (int(part) for part in value.split(':'))
                stats['runtime'] = hours * 3600 + minutes * 60 + seconds
    return model, stats


# Every complete output file has these statistics.
REQUIRED_STATS = set(COUNT_LABELS) | set(PCT_LABELS)


def read_results(files):
    """
    Parses the output files that exist and hold a complete results summary.
    Others are skipped: a missing file quietly (that model wasn't run), and a
    file without a summary (e.g. a batch run cut short) with a warning.
    Parameters:
      files - output files from check_images.py, one per model
    Returns:
      [(model, stats)] in the same order as files
    """
    results = []
    for filename in files:
        if not path.exists(filename):
            continue
        model, stats = parse_results_file(filename)
        if model is None or not REQUIRED_STATS <= set(stats):
            logging.warning("Skipping %s: it has no complete results summary "
                            "(was the run cut short?)", filename)
            continue
        results.append((model, stats))
    return results


def print_models_table(files=PET_IMAGE_FILES):
    """
    Prints the image counts and a table of accuracy, size and speed for each
    model, parsed from saved check_images.py output files. Files that don't
    exist or have no complete results are skipped (see read_results).
    Parameters:
      files - output files from check_images.py, one per model.
              Defaults to the pet_images runs.
    Returns:
      True if the table was printed, False if none of the files could be used.
    """
    results = read_results(files)
    if not results:
        logging.warning("Can't print the model table; none of %s has results. "
                        "Run run_models_batch.sh / run_models_batch_uploaded.sh first.",
                        ", ".join(files))
        return False

    # Image counts are the same for every model, so show the first one's.
    first_stats = results[0][1]
    for key, label in COUNT_LABELS.items():
        print("{:20}| {:3d}".format(label, first_stats[key]))
    print("")

    print(ROW_FORMAT.format('CNN model architecture', *(title for _, title, _ in COLUMNS)))
    for model, stats in results:
        print(ROW_FORMAT.format(model, *(fmt(stats[key]) if key in stats else "n/a"
                                         for key, _, fmt in COLUMNS)))
    return True


def main():
    print("Results for pet_images/:")
    print_models_table(PET_IMAGE_FILES)
    print("\nResults for uploaded_images/:")
    print_models_table(UPLOADED_IMAGE_FILES)


if __name__ == "__main__":
    main()

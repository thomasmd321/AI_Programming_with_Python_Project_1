#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/print_model_tables.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: May 1, 2020
# REVISED DATE: September 23, 2026
# PURPOSE: Builds a side-by-side comparison table of the 3 CNN architectures
#          by reading the output files written by run_models_batch.sh
#          (pet_images) or run_models_batch_uploaded.sh (uploaded_images).
#
# Usage: python print_model_tables.py
##
import logging
from os import path

from print_results import COUNT_LABELS, PCT_LABELS, SUMMARY_HEADER

PET_IMAGE_FILES = ('alexnet_pet-images.txt',
                   'resnet_pet-images.txt',
                   'vgg_pet-images.txt')
UPLOADED_IMAGE_FILES = ('alexnet_uploaded-images.txt',
                        'resnet_uploaded-images.txt',
                        'vgg_uploaded-images.txt')

# Table columns, left to right, as (statistic key, column heading).
COLUMNS = (('pct_correct_notdogs', '% Not-a-dog Correct'),
           ('pct_correct_dogs', '% Dogs Correct'),
           ('pct_correct_breed', '% Breeds Correct'),
           ('pct_match', '% Match Labels'))
ROW_FORMAT = "{:<24} |{:<20} |{:<15} |{:<18} |{:<17}"


def parse_results_file(filename):
    """
    Reads the model name and the statistics that print_results() wrote into a
    saved check_images.py output file.
    Parameters:
      filename - path to the output file (string)
    Returns:
      (model, stats) - model name in lower case (string) and a dictionary of
                       statistic key -> value (int for counts, float for pcts)
    """
    labels = {label: key for key, label in {**COUNT_LABELS, **PCT_LABELS}.items()}
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
            if model is not None and sep and label.strip() in labels:
                key = labels[label.strip()]
                stats[key] = int(value) if key.startswith('n_') else float(value)
    return model, stats


def print_models_table(files=PET_IMAGE_FILES):
    """
    Prints the image counts and a table of percentage statistics for each
    model, parsed from saved check_images.py output files.
    Parameters:
      files - output files from check_images.py, one per model.
              Defaults to the pet_images runs.
    Returns:
      True if the table was printed, False if any file was missing.
    """
    missing = [f for f in files if not path.exists(f)]
    if missing:
        logging.warning("Can't print the model table; missing %s. "
                        "Run run_models_batch.sh / run_models_batch_uploaded.sh first.",
                        ", ".join(missing))
        return False

    results = [parse_results_file(f) for f in files]

    # Image counts are the same for every model, so show the first one's.
    first_stats = results[0][1]
    for key, label in COUNT_LABELS.items():
        print("{:20}| {:3d}".format(label, first_stats[key]))
    print("")

    print(ROW_FORMAT.format('CNN model architecture', *(title for _, title in COLUMNS)))
    for model, stats in results:
        print(ROW_FORMAT.format(model, *("{:.2f}%".format(stats[key]) for key, _ in COLUMNS)))
    return True


def main():
    print("Results for pet_images/:")
    print_models_table(PET_IMAGE_FILES)
    print("\nResults for uploaded_images/:")
    print_models_table(UPLOADED_IMAGE_FILES)


if __name__ == "__main__":
    main()

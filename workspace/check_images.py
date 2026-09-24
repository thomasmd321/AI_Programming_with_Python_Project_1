#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/check_images.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: April 21, 2020
# REVISED DATE: September 23, 2026
# PURPOSE: Classifies pet images using a pretrained CNN model, compares these
#          classifications to the true identity of the pets in the images, and
#          summarizes how well the CNN performed on the image classification task.
#          The true identity of the pet (or object) in each image is taken from
#          the image's filename, so the program first extracts a label from
#          every filename and then classifies the images with the chosen CNN.
#          Running it for each supported architecture lets us compare which one
#          gives the 'best' classification (see print_model_tables.py).
#
# Usage:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
#   Optional extras:
#      --topk 3              show the top 3 guesses for mismatched images
#      --csv results.csv     save one row per image to a CSV file
##
import logging
from time import time

# Helper functions (supplied by Udacity) that print intermediate results so
# each step of the pipeline can be checked by eye.
from print_functions_for_lab_checks import (
    check_command_line_arguments,
    check_creating_pet_image_labels,
    check_classifying_images,
    check_classifying_labels_as_dogs,
    check_calculating_results,
)

# Pipeline steps written for this project, one module per step.
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels
from classify_images import classify_images
from classifier import parameter_count
from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from print_results import print_results, print_model_speed, print_top_predictions
from export_results import write_results_csv


def format_runtime(seconds):
    """Formats a duration in seconds as h:mm:ss, e.g. 32.4 -> '0:00:32'."""
    seconds = int(seconds)
    return "{:d}:{:02d}:{:02d}".format(seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def main():
    """Runs the full classify -> compare -> summarize pipeline."""
    # Warnings go to stderr, so they still show when stdout is redirected to
    # a file by the batch scripts.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    start_time = time()

    # 1. Read --dir, --arch and --dogfile from the command line.
    in_arg = get_input_args()
    check_command_line_arguments(in_arg)

    # 2. Build the results dictionary: {filename: [pet_label]}.
    results = get_pet_labels(in_arg.dir)
    check_creating_pet_image_labels(results)

    # 3. Run the CNN on every image and append the classifier label and a
    #    1/0 "labels match" flag to each entry: [pet_label, clf_label, match].
    #    predictions keeps each image's top guesses and their confidence.
    #    This step is timed on its own, to compare the models' speed. The model
    #    is loaded (and downloaded on first use) before the clock starts, so
    #    the time covers classifying only.
    n_params = parameter_count(in_arg.arch)
    classify_start = time()
    predictions = classify_images(in_arg.dir, results, in_arg.arch, top_k=max(in_arg.topk, 1))
    classify_seconds = time() - classify_start
    check_classifying_images(results)

    # 4. Append "pet label is a dog" and "classifier label is a dog" flags,
    #    so dog-vs-not-dog accuracy can be measured separately from breed.
    adjust_results4_isadog(results, in_arg.dogfile)
    check_classifying_labels_as_dogs(results)

    # 5. Turn the per-image results into counts and percentages.
    results_stats = calculates_results_stats(results)
    check_calculating_results(results, results_stats)

    # 6. Print the summary plus the misclassified dogs and breeds.
    print_results(results, results_stats, in_arg.arch, True, True)
    print_model_speed(n_params, classify_seconds, len(results))

    # Optional extras: the model's top guesses where it got the label wrong,
    # and a CSV of every image's results.
    if in_arg.topk:
        print_top_predictions(results, predictions, in_arg.topk)
    if in_arg.csv:
        write_results_csv(in_arg.csv, results, predictions, in_arg.arch)
        print("\nSaved per-image results to", in_arg.csv)

    print("\n** Total Elapsed Runtime:", format_runtime(time() - start_time))


if __name__ == "__main__":
    main()

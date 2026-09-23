#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/export_results.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: September 23, 2026
# PURPOSE: Saves the per-image results of one run to a CSV file, so they can
#          be opened in a spreadsheet or pandas instead of parsing text output.
##
import csv

CSV_COLUMNS = ('model', 'filename', 'pet_label', 'classifier_label', 'confidence',
               'labels_match', 'pet_is_dog', 'classifier_is_dog', 'top_guesses')


def write_results_csv(path, results_dic, predictions, model):
    """
    Writes one row per image to a CSV file (overwriting it).
    Parameters:
      path - CSV file to write (string)
      results_dic - the results dictionary, with all 5 items per image:
                    [pet_label, classifier_label, match, pet_is_dog,
                     classifier_is_dog]
      predictions - Dictionary with image filename as 'key' and a list of
                    (class name, probability) tuples as 'value', most likely
                    first (as returned by classify_images)
      model - the CNN architecture used for the run (string)
    Returns:
      None
    """
    with open(path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(CSV_COLUMNS)
        for key in sorted(results_dic):
            pet_label, classifier_label, match, pet_is_dog, classifier_is_dog = results_dic[key][:5]
            guesses = predictions[key]
            writer.writerow((
                model, key, pet_label, classifier_label,
                # Probability of the top guess, which is the classifier label.
                "{:.4f}".format(guesses[0][1]),
                match, pet_is_dog, classifier_is_dog,
                # e.g. "golden retriever (92.1%); labrador retriever (5.0%)"
                "; ".join("{} ({:.1f}%)".format(label, prob * 100) for label, prob in guesses),
            ))

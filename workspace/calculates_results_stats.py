#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/calculates_results_stats.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: April 23, 2020
# REVISED DATE:
# PURPOSE: Summarizes the results dictionary as counts and percentages so the
#          CNN architectures can be compared. Keys starting with 'n_' are
#          counts and keys starting with 'pct_' are percentages:
#            n_images - number of images
#            n_dogs_img - number of dog images
#            n_notdogs_img - number of NON-dog images
#            n_match - number of matches between pet & classifier labels
#            n_correct_dogs - number of correctly classified dog images
#            n_correct_notdogs - number of correctly classified NON-dog images
#            n_correct_breed - number of correctly classified dog breeds
#            pct_match - percentage of correct matches
#            pct_correct_dogs - percentage of correctly classified dogs
#            pct_correct_breed - percentage of correctly classified dog breeds
#            pct_correct_notdogs - percentage of correctly classified NON-dogs
##


def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the program run using the
    classifier's model architecture to classify pet images, and returns them
    in a dictionary for printing.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifier labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
    Returns:
     results_stats_dic - Dictionary that contains the results statistics (either
                    a percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value.
    """
    results_stats_dic = dict()

    # Counters that are incremented while walking through results_dic.
    results_stats_dic['n_dogs_img'] = 0
    results_stats_dic['n_match'] = 0
    results_stats_dic['n_correct_dogs'] = 0
    results_stats_dic['n_correct_notdogs'] = 0
    results_stats_dic['n_correct_breed'] = 0

    for key in results_dic:
        # Pet label and classifier label match.
        if results_dic[key][2] == 1:
            results_stats_dic['n_match'] += 1

        # A dog image whose breed was named by the classifier.
        if results_dic[key][3] == 1 and results_dic[key][0] in results_dic[key][1]:
            results_stats_dic['n_correct_breed'] += 1

        if results_dic[key][3] == 1:
            # Pet image is a dog.
            results_stats_dic['n_dogs_img'] += 1
            # ...and the classifier also said "dog".
            if results_dic[key][4] == 1:
                results_stats_dic['n_correct_dogs'] += 1
        else:
            # Pet image is not a dog.
            # NOTE: this counts every non-dog image as correct without
            # checking the classifier flag (index 4). See TODO.md.
            results_stats_dic['n_correct_notdogs'] += 1

    # Totals derived from the counters above.
    results_stats_dic['n_images'] = len(results_dic)
    results_stats_dic['n_notdogs_img'] = (results_stats_dic['n_images'] -
                                          results_stats_dic['n_dogs_img'])

    # Percentages. Each one falls back to 0.0 when its denominator is 0
    # (for example a folder with no dog images).
    results_stats_dic['pct_match'] = (results_stats_dic['n_match'] /
                                      results_stats_dic['n_images']) * 100.0

    try:
        results_stats_dic['pct_correct_dogs'] = (results_stats_dic['n_correct_dogs'] /
                                                 results_stats_dic['n_dogs_img']) * 100.0
    except ZeroDivisionError:
        print("#Debug ZeroDivisionError prevented")
        results_stats_dic['pct_correct_dogs'] = 0.0

    try:
        results_stats_dic['pct_correct_breed'] = (results_stats_dic['n_correct_breed'] /
                                                  results_stats_dic['n_dogs_img']) * 100.0
    except ZeroDivisionError:
        results_stats_dic['pct_correct_breed'] = 0.0
        print("#Debug ZeroDivisionError prevented")

    if results_stats_dic['n_notdogs_img'] > 0:
        results_stats_dic['pct_correct_notdogs'] = (results_stats_dic['n_correct_notdogs'] /
                                                    results_stats_dic['n_notdogs_img']) * 100.0
    else:
        results_stats_dic['pct_correct_notdogs'] = 0.0
        print("#Debug ZeroDivisionError prevented")

    return results_stats_dic

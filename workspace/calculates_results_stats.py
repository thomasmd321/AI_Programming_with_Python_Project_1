#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/calculates_results_stats.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: April 23, 2020
# REVISED DATE: September 23, 2026
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


def pct(part, whole):
    """Returns part as a percentage of whole, or 0.0 when whole is 0."""
    return (part / whole) * 100.0 if whole else 0.0


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
    stats = {
        'n_dogs_img': 0,
        'n_match': 0,
        'n_correct_dogs': 0,
        'n_correct_notdogs': 0,
        'n_correct_breed': 0,
    }

    for key in results_dic:
        _, _, is_match, pet_is_dog, classifier_is_dog = results_dic[key][:5]

        # Pet label and classifier label match.
        if is_match == 1:
            stats['n_match'] += 1

        if pet_is_dog == 1:
            stats['n_dogs_img'] += 1
            # The classifier also said "dog"...
            if classifier_is_dog == 1:
                stats['n_correct_dogs'] += 1
                # ...and named the right breed.
                if is_match == 1:
                    stats['n_correct_breed'] += 1
        elif classifier_is_dog == 0:
            # Not a dog, and the classifier agreed.
            stats['n_correct_notdogs'] += 1

    stats['n_images'] = len(results_dic)
    stats['n_notdogs_img'] = stats['n_images'] - stats['n_dogs_img']

    stats['pct_match'] = pct(stats['n_match'], stats['n_images'])
    stats['pct_correct_dogs'] = pct(stats['n_correct_dogs'], stats['n_dogs_img'])
    stats['pct_correct_breed'] = pct(stats['n_correct_breed'], stats['n_dogs_img'])
    stats['pct_correct_notdogs'] = pct(stats['n_correct_notdogs'], stats['n_notdogs_img'])

    return stats

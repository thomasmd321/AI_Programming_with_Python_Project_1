#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/print_results.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: April 27, 2020
# REVISED DATE: September 23, 2026
# PURPOSE: Prints the summary statistics for one model run and, when asked,
#          the images whose dog / not-dog classification was wrong and the
#          dog images whose breed was wrong.
##

# Display names for the statistics, in print order. print_model_tables.py
# uses these same names to read the values back out of saved output files.
COUNT_LABELS = {
    'n_images': 'N Images',
    'n_dogs_img': 'N Dog Images',
    'n_notdogs_img': 'N Not-Dog Images',
}
PCT_LABELS = {
    'pct_match': '% Match',
    'pct_correct_dogs': '% Correct Dogs',
    'pct_correct_breed': '% Correct Breed',
    'pct_correct_notdogs': '% Correct Not-a-Dog',
}
SUMMARY_HEADER = "*** Results Summary for CNN Model Architecture"


def print_results(results_dic, results_stats_dic, model,
                  print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
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
      results_stats_dic - Dictionary that contains the results statistics (either
                   a  percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
      model - Indicates which CNN model architecture will be used by the
              classifier function to classify the pet images,
              values must be either: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and
                             False doesn't print anything(default) (bool)
      print_incorrect_breed - True prints incorrectly classified dog breeds and
                              False doesn't print anything(default) (bool)
    Returns:
           None - simply printing results.
    """
    print("\n\n{} {} ***".format(SUMMARY_HEADER, model.upper()))
    for key, label in COUNT_LABELS.items():
        print("{:20}: {:3d}".format(label, results_stats_dic[key]))

    print("*** Results Statistics for CNN Model Architecture {} ***".format(model.upper()))
    for key, label in PCT_LABELS.items():
        print("{:20}: {:6.2f}".format(label, results_stats_dic[key]))

    # Dog / not-dog mistakes: only printed when requested and when at least
    # one image was put on the wrong side.
    if (print_incorrect_dogs and
            (results_stats_dic['n_correct_dogs'] + results_stats_dic['n_correct_notdogs']
             != results_stats_dic['n_images'])):
        print("\nINCORRECT Dog/NOT Dog Assignments:")
        for key in results_dic:
            # The two dog flags disagree: a dog called not-a-dog, or the
            # other way round.
            if results_dic[key][3] != results_dic[key][4]:
                print("Real: {:>26}   Classifier: {:>30}".format(
                    results_dic[key][0], results_dic[key][1]))

    # Breed mistakes: only printed when requested and when some dogs were
    # recognized as dogs but given the wrong breed.
    if (print_incorrect_breed and
            results_stats_dic['n_correct_dogs'] != results_stats_dic['n_correct_breed']):
        print("\nINCORRECT Dog Breed Assignment:")
        for key in results_dic:
            # Both labels say "dog" (flags sum to 2) but the labels don't match.
            if sum(results_dic[key][3:]) == 2 and results_dic[key][2] == 0:
                print("Real: {:>26}   Classifier: {:>30}".format(
                    results_dic[key][0], results_dic[key][1]))

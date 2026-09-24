#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/classify_images.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: April 21, 2020
# REVISED DATE: September 23, 2026
# PURPOSE: Runs the CNN classifier on every image and records how its label
#          compares with the pet label. For each entry in the results
#          dictionary this appends the classifier label at index 1 and a 1/0
#          "labels match" flag at index 2. It also returns the model's top
#          guesses and their confidence for every image.
##
import os
import re

from classifier import predict_batch


def labels_match(pet_label, classifier_label):
    """
    Returns True when pet_label appears in classifier_label as whole words.
    A classifier label can list several names for one breed separated by
    commas, e.g. 'dalmatian, coach dog, carriage dog'. Matching on word
    boundaries means 'cat' matches 'tabby, tabby cat' but not
    'polecat, fitch, foulmart, foumart'.
    """
    if not pet_label:
        return False
    pattern = r'(?<![a-z])' + re.escape(pet_label) + r'(?![a-z])'
    return re.search(pattern, classifier_label) is not None


def classify_images(images_dir, results_dic, model, top_k=1, batch_size=16):
    """
    Creates classifier labels with the classifier, compares pet labels
    to the classifier labels, and adds the classifier label and the comparison
    result to the results dictionary. Classifier labels are lower-cased and
    stripped so they are formatted like the pet labels.
    Parameters:
      images_dir - The (full) path to the folder of images that are to be
                   classified by the classifier function (string)
      results_dic - Results Dictionary with 'key' as image filename and 'value'
                    as a List. Where the list will contain the following items:
                  index 0 = pet image label (string)
                --- where index 1 & index 2 are added by this function ---
                  NEW - index 1 = classifier label (string)
                  NEW - index 2 = 1/0 (int)  where 1 = match between pet image
                    and classifier labels and 0 = no match between labels
      model - Indicates which CNN model architecture will be used by the
              classifier function to classify the pet images,
              e.g. resnet alexnet vgg (string)
      top_k - how many of the model's most likely classes to keep (int)
      batch_size - how many images to run through the model at once (int)
    Returns:
      predictions - Dictionary with image filename as 'key' and a list of
                    (lower-case class name, probability) tuples as 'value',
                    most likely first. results_dic is updated in place.
    """
    # Classify every image, a batch at a time.
    keys = list(results_dic)
    all_guesses = predict_batch([os.path.join(images_dir, key) for key in keys],
                                model, max(top_k, 1), batch_size)

    predictions = {}
    for key, guesses in zip(keys, all_guesses):
        predictions[key] = [(label.lower().strip(), prob) for label, prob in guesses]

        # The top guess, normalized to match pet labels, is the classifier label.
        model_label = predictions[key][0][0]

        # The pet label from the filename is the ground truth.
        truth = results_dic[key][0]
        results_dic[key].extend((model_label, int(labels_match(truth, model_label))))

    return predictions

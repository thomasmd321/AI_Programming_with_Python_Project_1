#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/classify_images.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: April 21, 2020
# REVISED DATE:
# PURPOSE: Runs the CNN classifier on every image and records how its label
#          compares with the pet label. For each entry in the results
#          dictionary this appends the classifier label at index 1 and a 1/0
#          "labels match" flag at index 2.
##
from classifier import classifier


def classify_images(images_dir, results_dic, model):
    """
    Creates classifier labels with the classifier function, compares pet labels
    to the classifier labels, and adds the classifier label and the comparison
    result to the results dictionary. Classifier labels are lower-cased and
    stripped so they are formatted like the pet labels.
    A classifier label can list several names for one breed separated by
    commas, e.g. 'dalmatian, coach dog, carriage dog'. A pet label counts as a
    match when it appears anywhere in that string.
    Parameters:
      images_dir - The (full) path to the folder of images that are to be
                   classified by the classifier function (string). Must end
                   with a '/' because paths are built as images_dir + filename.
      results_dic - Results Dictionary with 'key' as image filename and 'value'
                    as a List. Where the list will contain the following items:
                  index 0 = pet image label (string)
                --- where index 1 & index 2 are added by this function ---
                  NEW - index 1 = classifier label (string)
                  NEW - index 2 = 1/0 (int)  where 1 = match between pet image
                    and classifier labels and 0 = no match between labels
      model - Indicates which CNN model architecture will be used by the
              classifier function to classify the pet images,
              values must be either: resnet alexnet vgg (string)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    for key in results_dic:
        # Classify the image, then normalize the label to match pet labels.
        model_label = classifier(images_dir + key, model)
        model_label = model_label.lower().strip()

        # The pet label from the filename is the ground truth.
        truth = results_dic[key][0]
        if truth in model_label:
            results_dic[key].append(model_label)
            results_dic[key].append(1)
        else:
            results_dic[key].append(model_label)
            results_dic[key].append(0)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: April 21, 2020
# REVISED DATE: September 23, 2026
# PURPOSE: Builds the pet label for every image from its filename and returns
#          them in the results dictionary. The dictionary's key is the image
#          filename and its value is a list whose item at index 0 is the pet
#          image label (string). Later pipeline steps append more items.
##
import logging
import os

# File types the classifier can open. Anything else in the folder is skipped.
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


def is_image_file(filename):
    """Returns True for visible files with an image extension."""
    if filename.startswith('.'):
        return False
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS


def label_from_filename(filename):
    """
    Turns an image filename into a pet label: drop the extension, lower-case,
    split on '_' and keep only the purely alphabetic words.
    (ex. 'Boston_terrier_02259.jpg' -> 'boston terrier')
    """
    stem = os.path.splitext(filename)[0].lower()
    return " ".join(word for word in stem.split("_") if word.isalpha())


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames
    of the image files. The filenames hold the true identity of the pet, so
    these labels are used to check the labels returned by the classifier.
    Hidden files and files without an image extension are skipped.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a
      List. The list contains the following item:
         index 0 = pet image label (string)
    """
    results_dic = dict()
    # Sorted so the output order is the same on every machine.
    for filename in sorted(os.listdir(image_dir)):
        if not is_image_file(filename):
            logging.info("Skipping non-image file: %s", filename)
            continue
        results_dic[filename] = [label_from_filename(filename)]

    return results_dic

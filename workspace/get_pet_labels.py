#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_pet_labels.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: April 21, 2020
# REVISED DATE:
# PURPOSE: Builds the pet label for every image from its filename and returns
#          them in the results dictionary. The dictionary's key is the image
#          filename and its value is a list whose item at index 0 is the pet
#          image label (string). Later pipeline steps append more items.
##
from os import listdir


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels (results_dic) based upon the filenames
    of the image files. The filenames hold the true identity of the pet, so
    these labels are used to check the labels returned by the classifier.
    Labels are lower case, keep only the alphabetic words of the filename and
    have surrounding whitespace stripped.
    (ex. filename = 'Boston_terrier_02259.jpg' Pet label = 'boston terrier')
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by the classifier function (string)
    Returns:
      results_dic - Dictionary with 'key' as image filename and 'value' as a
      List. The list contains the following item:
         index 0 = pet image label (string)
    """
    results_dic = dict()
    pet_labels = []
    filename_list = listdir(image_dir)

    # Turn each filename into a label: split on '_' and keep only the purely
    # alphabetic pieces, which drops the numeric id and the '.jpg' part.
    pet_name = ""
    for idx in range(0, len(filename_list), 1):
        low_pet_image = filename_list[idx].lower()
        word_list_pet_image = low_pet_image.split("_")
        for word in word_list_pet_image:
            if word.isalpha():
                pet_name += word + " "

        pet_name = pet_name.strip()
        pet_labels.append(pet_name)
        pet_name = ''

    # Pair each filename with its label. pet_labels is in the same order as
    # filename_list, so the same index lines them up.
    for idx in range(0, len(filename_list), 1):
        if filename_list[idx] not in results_dic:
            results_dic[filename_list[idx]] = [pet_labels[idx]]
    # NOTE: this 'else' belongs to the for loop, not the if, so it runs every
    # time the loop finishes and always prints a warning for the last file.
    # See TODO.md.
    else:
        print("** Warning: Key=", filename_list[idx],
              "already exists in results_dic with value =",
              results_dic[filename_list[idx]])

    return results_dic

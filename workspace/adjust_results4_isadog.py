#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/adjust_results4_isadog.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: April 22, 2020
# REVISED DATE: May 1, 2020
# PURPOSE: Marks, for every image, whether the pet label is a dog and whether
#          the classifier label is a dog. A label is a dog when it appears in
#          the dog names file (dognames.txt). The two 1/0 flags are appended
#          to the results dictionary at index 3 (pet label) and index 4
#          (classifier label).
##


def adjust_results4_isadog(results_dic, dogfile):
    """
    Adjusts the results dictionary to determine if classifier correctly
    classified images 'as a dog' or 'not a dog' especially when not a match.
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with 'key' as image filename and 'value' as a
                    List. Where the list will contain the following items:
                  index 0 = pet image label (string)
                  index 1 = classifier label (string)
                  index 2 = 1/0 (int)  where 1 = match between pet image
                    and classifier labels and 0 = no match between labels
                ------ where index 3 & index 4 are added by this function -----
                 NEW - index 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                 NEW - index 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
     dogfile - A text file that contains names of all dogs from the classifier
               function and dog names from the pet image files. This file has
               one dog name per line dog names are all in lowercase with
               spaces separating the distinct words of the dog name. Dog names
               from the classifier function can be a string of dog names separated
               by commas when a particular breed of dog has multiple dog names
               associated with that breed (ex. maltese dog, maltese terrier,
               maltese) (string - indicates text file's filename)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    # Load the dog names into a dictionary used as a fast lookup set. Each
    # whole line is stored, and a line holding several comma-separated names
    # also has each name stored on its own.
    dognames_dic = dict()
    with open(dogfile, "r") as infile:
        line = infile.readline()
        while line != "":
            line = line.rstrip('\n')
            if len(dognames_dic) > 0:
                if line not in dognames_dic:
                    dognames_dic[line] = 1
                if ',' in line:
                    # Several names for the same breed.
                    temp_keys = line.split(',')
                    for sub_key in temp_keys:
                        sub_key = sub_key.lstrip()
                        if sub_key not in dognames_dic:
                            dognames_dic[sub_key] = 1
                        else:
                            print("Key is already in dognames_dic {}".format(sub_key))
                else:
                    if line not in dognames_dic:
                        dognames_dic[line] = 1
                    else:
                        print("Key is already in dognames_dic {}".format(line))
            else:
                # First line: the dictionary is still empty, so just add it.
                # NOTE: a first line holding comma-separated names is not
                # split. See TODO.md.
                dognames_dic[line] = 1
            line = infile.readline()

    # Append (pet label is a dog, classifier label is a dog) as 1/0 flags.
    for key in results_dic:
        if results_dic[key][0] in dognames_dic:
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((1, 1))
            else:
                results_dic[key].extend((1, 0))
        else:
            if results_dic[key][1] in dognames_dic:
                results_dic[key].extend((0, 1))
            else:
                results_dic[key].extend((0, 0))

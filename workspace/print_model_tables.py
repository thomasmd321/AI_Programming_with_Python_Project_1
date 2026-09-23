#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/print_model_tables.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: May 1, 2020
# REVISED DATE: May 2, 2020
# PURPOSE: Builds a side-by-side comparison table of the 3 CNN architectures
#          by reading the output files written by run_models_batch.sh
#          (pet_images) or run_models_batch_uploaded.sh (uploaded_images).
#
# Usage: python print_model_tables.py
##
from os import path


def print_models_table(file1='alexnet_pet-images.txt',
                       file2='resnet_pet-images.txt',
                       file3='vgg_pet-images.txt'):
    """
    Prints the image counts and a table of percentage statistics for each
    model, parsed from saved check_images.py output files.
    Parameters:
      file1, file2, file3 - output files from check_images.py, one per model.
                            Defaults are the pet_images runs.
    Returns:
      None - prints to the console. Returns early if any file is missing.
    """
    table_dic = {}       # model name -> list of 'pct_...' lines
    mini_table_dic = {}  # model name -> list of 'N ... Images' lines

    # All 3 output files must exist before a table can be built.
    for check_file in (file1, file2, file3):
        if path.exists(check_file):
            print("file: {} Exists".format(check_file))
        else:
            print("Ensure both run_models_batch.sh and run_models_batch_uploaded.sh have been run")
            print("file: {} Missing; exiting".format(check_file))
            return

    for tempfile in [file1, file2, file3]:
        with open(tempfile, 'r') as myFile:
            for line in myFile:
                line = line.rstrip('\n').lstrip()

                # The model name comes from the "arch = <model>" line printed
                # near the top of each file; later lines are filed under it.
                if 'arch =' in line:
                    key = (line.split('=')[-1].strip())

                # Image count lines, e.g. "N Dog Images        :  30".
                if 'N Images' in line and 'N Dog Images' not in line:
                    mini_table_dic[key] = [line]
                if 'N Dog Images' in line and 'N Images' not in line:
                    mini_table_dic[key].append(line)
                if 'N Not-Dog Images' in line and 'N Dog Images' not in line:
                    mini_table_dic[key].append(line)

                # Percentage lines, e.g. "pct_match: 87.50%". pct_match is
                # printed first, so it starts the list and the others follow
                # in print order: [match, dogs, breed, notdogs].
                if 'pct_correct_notdogs:' in line:
                    table_dic[key].append(line)
                if 'pct_correct_dogs:' in line:
                    table_dic[key].append(line)
                if 'pct_correct_breed:' in line:
                    table_dic[key].append(line)
                if 'pct_match:' in line:
                    table_dic[key] = [line]

    # Image counts are the same for every model, so show alexnet's.
    print("{}".format(mini_table_dic['alexnet'][0].replace(':', '|')))
    print("{}".format(mini_table_dic['alexnet'][1].replace(':', '|')))
    print("{}".format(mini_table_dic['alexnet'][2].replace(':', '|')))
    print("")

    # One row per model; the value is the text after the last ':'.
    print("{:<24} |{:<20} |{:<15} |{:<18} |{:<17}".format(
        'CNN model architecture', '% Not-a-dog Correct', '% Dogs Correct',
        '% Breeds Correct', '% Match Labels'))
    for key in table_dic:
        print("{:<24} |{:<20} |{:<15} |{:<18} |{:<17}".format(
            key,
            table_dic[key][3].split(':')[-1].lstrip(),
            table_dic[key][1].split(':')[-1].lstrip(),
            table_dic[key][2].split(':')[-1].lstrip(),
            table_dic[key][0].split(':')[-1].lstrip()))


# Only print the tables when run directly. check_images.py imports this module
# and prints the tables itself; without this guard they printed twice.
if __name__ == "__main__":
    print_models_table()
    print("")
    print_models_table('alexnet_uploaded-images.txt',
                       'resnet_uploaded-images.txt',
                       'vgg_uploaded-images.txt')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: April 21, 2020
# REVISED DATE: September 23, 2026
# PURPOSE: Reads the program's command line arguments with argparse. Any
#          argument the user leaves out falls back to its default:
#     1. Image Folder as --dir with default value 'pet_images/'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
#     4. Number of top guesses to show for mismatches as --topk (default 0 = off)
#     5. CSV file to save per-image results to as --csv (default: don't save)
##
import argparse

# The architectures classifier.py knows how to load. The first three are the
# ones the Udacity project compares. Kept in sync with classifier.ARCHITECTURES
# (checked by the tests); listed here so parsing arguments doesn't need PyTorch.
ARCHITECTURES = ('resnet', 'alexnet', 'vgg', 'resnet50', 'efficientnet')


def get_input_args():
    """
    Retrieves and parses the command line arguments provided by the user when
    they run the program from a terminal window. Missing arguments use the
    defaults listed above.
    Parameters:
     None
    Returns:
     argparse.Namespace with attributes dir, arch, dogfile, topk and csv
    """
    parser = argparse.ArgumentParser(
        description='Classify pet images with a pretrained CNN.')

    parser.add_argument('--dir', type=str, default='pet_images/',
                        help='path to the folder of pet images')
    # argparse rejects anything else with a clear error message.
    parser.add_argument('--arch', type=str, default='vgg', choices=ARCHITECTURES,
                        help='CNN model architecture to use')
    # Text file with one dog name per line, used to decide "is it a dog?".
    parser.add_argument('--dogfile', type=str, default='dognames.txt',
                        help='The file that contains the list of valid dognames')
    parser.add_argument('--topk', type=int, default=0,
                        help="for images whose labels don't match, print the model's "
                             "top K guesses with their confidence (0 = off)")
    parser.add_argument('--csv', type=str, default=None,
                        help='also save one row per image to this CSV file')

    args = parser.parse_args()
    if args.topk < 0:
        parser.error('--topk must be 0 or more')
    return args

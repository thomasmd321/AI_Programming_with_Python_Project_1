#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/get_input_args.py
#
# PROGRAMMER: Thomas Stewart
# DATE CREATED: April 21, 2020
# REVISED DATE:
# PURPOSE: Reads the program's 3 command line arguments with argparse. Any
#          argument the user leaves out falls back to its default:
#     1. Image Folder as --dir with default value 'pet_images/'
#     2. CNN Model Architecture as --arch with default value 'vgg'
#     3. Text File with Dog Names as --dogfile with default value 'dognames.txt'
##
import argparse


def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. Missing arguments use the
    defaults listed above.
    Parameters:
     None
    Returns:
     argparse.Namespace with attributes dir, arch and dogfile
    """
    parser = argparse.ArgumentParser()

    # Folder that holds the images. The trailing slash matters: other modules
    # build image paths as image_dir + filename.
    parser.add_argument('--dir', type=str, default='pet_images/',
                        help='path to the folder of pet images')
    # One of 'resnet', 'alexnet' or 'vgg' (the models in classifier.py).
    parser.add_argument('--arch', type=str, default='vgg',
                        help='CNN model architecture to use')
    # Text file with one dog name per line, used to decide "is it a dog?".
    parser.add_argument('--dogfile', type=str, default='dognames.txt',
                        help='The file that contains the list of valid dognames')

    return parser.parse_args()

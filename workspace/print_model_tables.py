#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/adjust_results4_isadog.py
#                                                                             
# PROGRAMMER: Thomas Stewart
# DATE CREATED: May 1, 2020                                
# REVISED DATE: May 2, 2020
# PURPOSE: Create a function 

import os.path
from os import path

def print_models_table(file1='alexnet_pet-images.txt',file2='resnet_pet-images.txt',file3='vgg_pet-images.txt'):
    table_dic = {}
    mini_table_dic = {}
    # Check output files exist for all three models
    if path.exists(file1):
        print("file: {} Exists".format(file1))
    else:
        print("Enusre both run_models_batch.sh and run_models_batch_uploaded.sh have been run")
        print("file: {} Missing; exiting".format(file1))
        return
    if path.exists(file2):
        print("file: {} Exists".format(file2))
    else:
        print("Enusre both run_models_batch.sh and run_models_batch_uploaded.sh have been run")
        print("file: {} Missing; exiting".format(file2))
        return    
    if path.exists(file3):
        print("file: {} Exists".format(file3))
    else:
        print("Enusre both run_models_batch.sh and run_models_batch_uploaded.sh have been run")
        print("file: {} Missing; exiting".format(file3))
        return    
    file_list = [file1, file2, file3] 
    for tempfile in file_list:
        with open(tempfile,'r') as myFile:
            for line in myFile:
                line = line.rstrip('\n')
                line = line.lstrip()
                    
                               
                if 'arch =' in line:
                    #print("DEBUG {}".format(line))
                    key = (line.split('=')[-1].strip())
                
                if 'N Images' in line and 'N Dog Images' not in line:
                    mini_table_dic[key] =[line]
                    #print("#Debug {}".format(line))
                if 'N Dog Images' in line and 'N Images' not in line:
                    mini_table_dic[key].append(line)
                    #print("#Debug {}".format(line))
                if 'N Not-Dog Images' in line and 'N Dog Images' not in line:
                    mini_table_dic[key].append(line)
                    #print("#Debug {}".format(line))
                #                N Images            :  40
                #N Dog Images        :  30
                #N Not-Dog Images    :  10
                
                if 'pct_correct_notdogs:' in line:
                    #print("DEBUG {}".format(line))
                    #print("DEBUG {}".format(line.split(':')[-1].lstrip()))
                    #table_dic[key].append(line.split(':')[-1].lstrip())
                    table_dic[key].append(line)
                if 'pct_correct_dogs:' in line:
                    #print("DEBUG {}".format(line))
                    #print("DEBUG {}".format(line.split(':')[-1].lstrip()))
                    #table_dic[key].append(line.split(':')[-1].lstrip())
                    table_dic[key].append(line)
                if 'pct_correct_breed:' in line:
                    #print("DEBUG {}".format(line))
                    #print("DEBUG {}".format(line.split(':')[-1].lstrip()))
                    #table_dic[key].append(li)ne.split(':')[-1].lstrip())
                    table_dic[key].append(line)
                if 'pct_match:' in line:
                    #print("DEBUG {}".format(line))
                    #print("DEBUG {}".format(line.split(':')[-1].lstrip()))
                    #table_dic[key]=[line.split(':')[-1].lstrip()]
                    table_dic[key]=[line]
                    
    #print("table_dic - {}".format(table_dic))
    #print("mini_table_dic - {}".format(mini_table_dic))
    #print(table_dic[key][0])
    print ("{}".format(mini_table_dic['alexnet'][0].replace(':','|')))
    print ("{}".format(mini_table_dic['alexnet'][1].replace(':','|')))
    print ("{}".format(mini_table_dic['alexnet'][2].replace(':','|')))
    print("")
    print ("{:<24} |{:<20} |{:<15} |{:<18} |{:<17}".format('CNN model architecture', '% Not-a-dog Correct', '% Dogs Correct', '% Breeds Correct', '% Match Labels'))
    for key in table_dic:
        print ("{:<24} |{:<20} |{:<15} |{:<18} |{:<17}".format(key, table_dic[key][3].split(':')[-1].lstrip(), table_dic[key][1].split(':')[-1].lstrip(), table_dic[key][2].split(':')[-1].lstrip(), table_dic[key][0].split(':')[-1].lstrip()))
# Call to main function to run the program
#if __name__ == "__main__":
print_models_table()
print("")
print_models_table('alexnet_uploaded-images.txt','resnet_uploaded-images.txt','vgg_uploaded-images.txt')
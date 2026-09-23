#!/bin/sh
# */AIPND-revision/intropyproject-classify-pet-images/run_models_batch.sh
#
# PROGRAMMER: Jennifer S., revised by Thomas Stewart
# DATE CREATED: 02/08/2018
# REVISED DATE: September 23, 2026
# PURPOSE: Runs every model on pet_images/ to test which provides the 'best' solution.
#          Each model's output goes to <model>_pet-images.txt, its per-image
#          results to <model>_pet-images.csv, and the comparison table is printed
#          at the end. The Udacity project compares resnet, alexnet and vgg;
#          resnet50 and efficientnet are extra.
#
# Usage: sh run_models_batch.sh    -- run from inside workspace/
#
set -e
for arch in resnet alexnet vgg resnet50 efficientnet; do
    echo "Running $arch on pet_images/ ..."
    python check_images.py --dir pet_images/ --arch "$arch" --dogfile dognames.txt \
        --topk 3 --csv "${arch}_pet-images.csv" > "${arch}_pet-images.txt"
done

# Print the side-by-side comparison of the models
python print_model_tables.py

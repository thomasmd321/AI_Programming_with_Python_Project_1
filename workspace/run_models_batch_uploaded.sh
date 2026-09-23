#!/bin/sh
# */AIPND-revision/intropyproject-classify-pet-images/run_models_batch_uploaded.sh
#
# PROGRAMMER: Jennifer S., revised by Thomas Stewart
# DATE CREATED: 02/08/2018
# REVISED DATE: September 23, 2026
# PURPOSE: Runs every model on uploaded_images/ to test which provides the 'best' solution.
#          Each model's output goes to <model>_uploaded-images.txt, its per-image
#          results to <model>_uploaded-images.csv, and the comparison table is printed
#          at the end. The Udacity project compares resnet, alexnet and vgg;
#          resnet50 and efficientnet are extra.
#
# Usage: sh run_models_batch_uploaded.sh    -- run from inside workspace/
#
set -e
for arch in resnet alexnet vgg resnet50 efficientnet; do
    echo "Running $arch on uploaded_images/ ..."
    python check_images.py --dir uploaded_images/ --arch "$arch" --dogfile dognames.txt \
        --topk 3 --csv "${arch}_uploaded-images.csv" > "${arch}_uploaded-images.txt"
done

# Print the side-by-side comparison of the models
python print_model_tables.py

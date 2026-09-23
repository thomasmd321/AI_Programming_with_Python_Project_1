#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/classifier.py
#
# PROGRAMMER: Udacity (supplied with the project), revised by Thomas Stewart
# REVISED DATE: September 23, 2026
# PURPOSE: Provides classifier(img_path, model_name), which runs one image
#          through a CNN pretrained on ImageNet and returns the predicted
#          ImageNet class name, e.g. 'Maltese dog, Maltese terrier, Maltese'.
#          Supported model names: 'resnet', 'alexnet', 'vgg'.
#
#          Each model is downloaded/loaded the first time it is used and then
#          kept in memory, so a run only loads the one architecture it needs.
##
import ast
import os

import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

# --arch value -> (torchvision builder function, weights enum name).
ARCHITECTURES = {
    'resnet': ('resnet18', 'ResNet18_Weights'),
    'alexnet': ('alexnet', 'AlexNet_Weights'),
    'vgg': ('vgg16', 'VGG16_Weights'),
}

# The standard ImageNet preprocessing: resize, crop to 224x224, convert to a
# tensor and normalize with the ImageNet mean/std. Built once and reused.
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ImageNet labels: the file is a Python dict literal {class index: name}.
# Resolved next to this file so scripts work from any directory.
_LABELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'imagenet1000_clsid_to_human.txt')
with open(_LABELS_FILE) as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

# Models loaded so far, keyed by --arch value.
_loaded_models = {}


def _load_model(model_name):
    """Loads a pretrained model in evaluation mode, caching it for reuse."""
    if model_name not in _loaded_models:
        builder_name, weights_name = ARCHITECTURES[model_name]
        builder = getattr(tv_models, builder_name)
        weights = getattr(tv_models, weights_name, None)
        if weights is not None:
            # torchvision 0.13+
            model = builder(weights=weights.DEFAULT)
        else:
            # Older torchvision without the weights enums.
            model = builder(pretrained=True)
        _loaded_models[model_name] = model.eval()
    return _loaded_models[model_name]


def classifier(img_path, model_name):
    """
    Classifies one image with a pretrained CNN.
    Parameters:
      img_path - path to the image file (string)
      model_name - 'resnet', 'alexnet' or 'vgg' (string)
    Returns:
      ImageNet class name(s) for the top prediction (string, mixed case,
      several names separated by commas)
    """
    if model_name not in ARCHITECTURES:
        raise ValueError("Unknown model {!r}; expected one of {}".format(
            model_name, ", ".join(ARCHITECTURES)))
    model = _load_model(model_name)

    # Some images are grayscale or have an alpha channel; the models expect RGB.
    img_pil = Image.open(img_path).convert('RGB')

    # Preprocess and add a batch dimension: (3, 224, 224) -> (1, 3, 224, 224).
    img_tensor = PREPROCESS(img_pil).unsqueeze(0)

    # Inference only, so skip gradient tracking.
    with torch.no_grad():
        output = model(img_tensor)

    # Return the name of the highest-scoring class.
    pred_idx = int(output.argmax())
    return imagenet_classes_dict[pred_idx]

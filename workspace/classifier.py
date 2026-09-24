#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/classifier.py
#
# PROGRAMMER: Udacity (supplied with the project), revised by Thomas Stewart
# REVISED DATE: September 23, 2026
# PURPOSE: Runs images through a CNN pretrained on ImageNet.
#            predict_batch(img_paths, model_name, k) -> for each image, the k
#                most likely ImageNet classes with their probabilities, e.g.
#                [('golden retriever', 0.92), ('Labrador retriever', 0.05)]
#            predict(img_path, model_name, k) -> the same for one image
#            classifier(img_path, model_name) -> just the top class name
#            parameter_count(model_name) -> the model's size
#          Supported model names are the keys of ARCHITECTURES.
#
#          Each model is downloaded/loaded the first time it is used and then
#          kept in memory, so a run only loads the architectures it needs.
#          A GPU is used automatically when one is available.
##
import ast
import os

import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image, ImageOps

# --arch value -> (torchvision builder function, weights enum name).
# To compare another torchvision model, add it here and to
# get_input_args.ARCHITECTURES.
ARCHITECTURES = {
    'resnet': ('resnet18', 'ResNet18_Weights'),
    'alexnet': ('alexnet', 'AlexNet_Weights'),
    'vgg': ('vgg16', 'VGG16_Weights'),
    'resnet50': ('resnet50', 'ResNet50_Weights'),
    'efficientnet': ('efficientnet_b0', 'EfficientNet_B0_Weights'),
}

# Run on the GPU when there is one.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    if model_name not in ARCHITECTURES:
        raise ValueError("Unknown model {!r}; expected one of {}".format(
            model_name, ", ".join(ARCHITECTURES)))
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
        _loaded_models[model_name] = model.to(DEVICE).eval()
    return _loaded_models[model_name]


def _load_image(img_path):
    """Opens an image upright (applying any EXIF rotation) and in RGB."""
    img = ImageOps.exif_transpose(Image.open(img_path))
    # Some images are grayscale or have an alpha channel; the models expect RGB.
    return img.convert('RGB')


def predict_batch(img_paths, model_name, k=1, batch_size=16):
    """
    Classifies several images with a pretrained CNN, batch_size at a time.
    Running a batch through the model at once is usually faster than one
    image at a time, especially on a GPU (on a CPU the gain is small).
    Parameters:
      img_paths - paths to the image files (list of strings)
      model_name - one of the ARCHITECTURES keys (string)
      k - how many of the most likely classes to return per image (int)
      batch_size - how many images to run through the model at once (int)
    Returns:
      one list per image, in the same order as img_paths, of
      (class name, probability) tuples, most likely first. Class names are
      mixed case, with several names separated by commas.
    """
    model = _load_model(model_name)
    results = []
    for start in range(0, len(img_paths), batch_size):
        chunk = img_paths[start:start + batch_size]

        # Preprocess each image and stack them: (batch, 3, 224, 224).
        batch = torch.stack([PREPROCESS(_load_image(path)) for path in chunk]).to(DEVICE)

        # Inference only, so skip gradient tracking.
        with torch.no_grad():
            output = model(batch)

        # Turn each image's scores for the 1000 classes into probabilities,
        # and keep the top k.
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_idxs = torch.topk(probabilities, k, dim=1)
        for probs, idxs in zip(top_probs.tolist(), top_idxs.tolist()):
            results.append([(imagenet_classes_dict[int(idx)], float(prob))
                            for prob, idx in zip(probs, idxs)])
    return results


def predict(img_path, model_name, k=1):
    """
    Classifies one image with a pretrained CNN.
    Parameters:
      img_path - path to the image file (string)
      model_name - one of the ARCHITECTURES keys (string)
      k - how many of the most likely classes to return (int)
    Returns:
      list of (class name, probability) tuples, most likely first.
    """
    return predict_batch([img_path], model_name, k)[0]


def parameter_count(model_name):
    """Returns the number of learned parameters (weights) in the model."""
    return sum(param.numel() for param in _load_model(model_name).parameters())


def classifier(img_path, model_name):
    """
    Classifies one image with a pretrained CNN.
    Parameters:
      img_path - path to the image file (string)
      model_name - one of the ARCHITECTURES keys (string)
    Returns:
      ImageNet class name(s) for the top prediction (string, mixed case,
      several names separated by commas)
    """
    return predict(img_path, model_name, k=1)[0][0]

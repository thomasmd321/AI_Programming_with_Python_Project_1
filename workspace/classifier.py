#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND-revision/intropyproject-classify-pet-images/classifier.py
#
# PROGRAMMER: Udacity (supplied with the project)
# PURPOSE: Provides classifier(img_path, model_name), which runs one image
#          through a CNN pretrained on ImageNet and returns the predicted
#          ImageNet class name, e.g. 'Maltese dog, Maltese terrier, Maltese'.
#          Supported model names: 'resnet', 'alexnet', 'vgg'.
#
# NOTE: all 3 models are downloaded/loaded when this module is imported,
#       and the ImageNet label file is opened relative to the current working
#       directory, so run the programs from inside workspace/.
##
import ast
from PIL import Image
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
from torch import __version__

# Pretrained ImageNet models, loaded once at import time.
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

# Maps the --arch value to its model. (This rebinds the name 'models', so the
# torchvision module is no longer reachable under that name below.)
models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# Obtain ImageNet labels: the file is a Python dict literal {class index: name}.
with open('imagenet1000_clsid_to_human.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

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
    # load the image
    img_pil = Image.open(img_path)

    # Define transforms: the standard ImageNet preprocessing (resize, crop to
    # 224x224, convert to a tensor, normalize with ImageNet mean/std).
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # preprocess the image
    img_tensor = preprocess(img_pil)
    
    # resize the tensor (add dimension for batch)
    img_tensor.unsqueeze_(0)
    
    # wrap input in variable, wrap input in variable - no longer needed for
    # v 0.4 & higher code changed 04/26/2018 by Jennifer S. to handle PyTorch upgrade
    pytorch_ver = __version__.split('.')
    
    # pytorch versions 0.4 & hihger - Variable depreciated so that it returns
    # a tensor. So to address tensor as output (not wrapper) and to mimic the 
    # affect of setting volatile = True (because we are using pretrained models
    # for inference) we can set requires_gradient to False. Here we just set 
    # requires_grad_ to False on our tensor 
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        img_tensor.requires_grad_(False)
    
    # pytorch versions less than 0.4 - uses Variable because not-depreciated
    else:
        # apply model to input
        # wrap input in variable
        data = Variable(img_tensor, volatile = True) 

    # apply model to input
    model = models[model_name]

    # puts model in evaluation mode
    # instead of (default)training mode
    model = model.eval()
    
    # apply data to model - adjusted based upon version to account for 
    # operating on a Tensor for version 0.4 & higher.
    if int(pytorch_ver[0]) > 0 or int(pytorch_ver[1]) >= 4:
        output = model(img_tensor)

    # pytorch versions less than 0.4
    else:
        # apply data to model
        output = model(data)

    # Return the name of the highest-scoring class.
    pred_idx = output.data.numpy().argmax()

    return imagenet_classes_dict[pred_idx]

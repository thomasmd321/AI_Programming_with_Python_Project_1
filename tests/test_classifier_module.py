"""Tests for workspace/classifier.py with torch, torchvision and Pillow
replaced by fakes, so the model loading and label lookup logic can be checked
without downloading any weights."""
import sys
import types
from unittest import mock

import pytest


@pytest.fixture
def classifier_module(monkeypatch, import_fresh):
    """Imports classifier.py against fake torch/torchvision/PIL modules."""
    torch = types.ModuleType("torch")
    torch.no_grad = mock.MagicMock()

    tv_models = types.ModuleType("torchvision.models")
    built = []

    def make_builder(name):
        def builder(**kwargs):
            built.append((name, kwargs))
            model = mock.MagicMock(name=name)
            model.eval.return_value = model
            # "Predicts" ImageNet class 207 (golden retriever).
            model.return_value.argmax.return_value = 207
            return model
        return builder

    for name, weights in (("resnet18", "ResNet18_Weights"), ("alexnet", "AlexNet_Weights"),
                          ("vgg16", "VGG16_Weights")):
        setattr(tv_models, name, make_builder(name))
        setattr(tv_models, weights, types.SimpleNamespace(DEFAULT=weights + ".DEFAULT"))

    torchvision = types.ModuleType("torchvision")
    torchvision.models = tv_models
    torchvision.transforms = mock.MagicMock()
    pil = types.ModuleType("PIL")
    pil.Image = mock.MagicMock()

    for name, module in {"torch": torch, "torchvision": torchvision,
                         "torchvision.models": tv_models,
                         "torchvision.transforms": torchvision.transforms,
                         "PIL": pil, "PIL.Image": pil.Image}.items():
        monkeypatch.setitem(sys.modules, name, module)

    module = import_fresh("classifier")
    return module, built, tv_models


def test_classifier_returns_imagenet_label(classifier_module):
    classifier, _, _ = classifier_module
    assert classifier.classifier("dog.jpg", "vgg") == "golden retriever"


def test_classifier_loads_only_the_requested_model_once(classifier_module):
    classifier, built, _ = classifier_module

    classifier.classifier("a.jpg", "resnet")
    classifier.classifier("b.jpg", "resnet")

    assert built == [("resnet18", {"weights": "ResNet18_Weights.DEFAULT"})]


def test_classifier_falls_back_to_pretrained_flag(classifier_module, monkeypatch):
    # torchvision older than 0.13 has no weights enums.
    classifier, built, tv_models = classifier_module
    monkeypatch.delattr(tv_models, "AlexNet_Weights")

    classifier.classifier("a.jpg", "alexnet")

    assert built == [("alexnet", {"pretrained": True})]


def test_classifier_rejects_unknown_model(classifier_module):
    classifier, _, _ = classifier_module
    with pytest.raises(ValueError, match="Unknown model"):
        classifier.classifier("a.jpg", "vgg19")

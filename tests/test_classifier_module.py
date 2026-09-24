"""Tests for workspace/classifier.py with torch, torchvision and Pillow
replaced by fakes, so the model loading, top-k and label lookup logic can be
checked without PyTorch or downloading any weights."""
import sys
import types
from unittest import mock

import pytest

import get_input_args

# The fake model "predicts" these ImageNet classes with these probabilities,
# most likely first: 207 golden retriever, 208 Labrador retriever, 222 kuvasz.
FAKE_TOP = [(0.90, 207), (0.06, 208), (0.02, 222)]


class FakeTensor:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class FakeBatch:
    """Stands in for a stacked batch of images, and for the model's output."""
    def __init__(self, size):
        self.size = size

    def to(self, device):
        return self


def fake_topk(probabilities, k, dim):
    # Every image in the batch gets the same fake top k.
    top = FAKE_TOP[:k]
    return (FakeTensor([[p for p, _ in top]] * probabilities.size),
            FakeTensor([[i for _, i in top]] * probabilities.size))


@pytest.fixture
def classifier_module(monkeypatch, import_fresh):
    """Imports classifier.py against fake torch/torchvision/PIL modules."""
    torch = types.ModuleType("torch")
    torch.no_grad = mock.MagicMock()
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda name: "device:" + name
    torch.nn = types.SimpleNamespace(
        functional=types.SimpleNamespace(softmax=lambda x, dim: x))
    torch.topk = fake_topk
    stacked = []
    torch.stack = lambda images: stacked.append(len(images)) or FakeBatch(len(images))

    tv_models = types.ModuleType("torchvision.models")
    built = []

    def make_builder(name):
        def builder(**kwargs):
            built.append((name, kwargs))
            model = mock.MagicMock(name=name)
            model.to.return_value = model
            model.eval.return_value = model
            # The "output" has one row per image in the batch.
            model.side_effect = lambda batch: batch
            model.parameters.return_value = [FakeTensor([0]) for _ in range(3)]
            for param, size in zip(model.parameters.return_value, (100, 20, 3)):
                param.numel = lambda size=size: size
            return model
        return builder

    for name, weights in (("resnet18", "ResNet18_Weights"), ("alexnet", "AlexNet_Weights"),
                          ("vgg16", "VGG16_Weights"), ("resnet50", "ResNet50_Weights"),
                          ("efficientnet_b0", "EfficientNet_B0_Weights")):
        setattr(tv_models, name, make_builder(name))
        setattr(tv_models, weights, types.SimpleNamespace(DEFAULT=weights + ".DEFAULT"))

    torchvision = types.ModuleType("torchvision")
    torchvision.models = tv_models
    torchvision.transforms = mock.MagicMock()
    pil = types.ModuleType("PIL")
    pil.Image = mock.MagicMock()
    pil.ImageOps = mock.MagicMock()

    for name, module in {"torch": torch, "torchvision": torchvision,
                         "torchvision.models": tv_models,
                         "torchvision.transforms": torchvision.transforms,
                         "PIL": pil, "PIL.Image": pil.Image,
                         "PIL.ImageOps": pil.ImageOps}.items():
        monkeypatch.setitem(sys.modules, name, module)

    module = import_fresh("classifier")
    module.stacked = stacked
    return module, built, tv_models, pil


def test_classifier_returns_imagenet_label(classifier_module):
    classifier, _, _, _ = classifier_module
    assert classifier.classifier("dog.jpg", "vgg") == "golden retriever"


def test_predict_returns_top_k_with_probabilities(classifier_module):
    classifier, _, _, _ = classifier_module
    assert classifier.predict("dog.jpg", "vgg", k=2) == [
        ("golden retriever", 0.90), ("Labrador retriever", 0.06)]


def test_predict_applies_exif_orientation(classifier_module):
    classifier, _, _, pil = classifier_module

    classifier.predict("dog.jpg", "vgg")

    pil.ImageOps.exif_transpose.assert_called_once_with(pil.Image.open.return_value)
    pil.ImageOps.exif_transpose.return_value.convert.assert_called_once_with("RGB")


def test_classifier_loads_only_the_requested_model_once(classifier_module):
    classifier, built, _, _ = classifier_module

    classifier.classifier("a.jpg", "resnet")
    classifier.classifier("b.jpg", "resnet")

    assert built == [("resnet18", {"weights": "ResNet18_Weights.DEFAULT"})]


def test_models_are_moved_to_the_device(classifier_module):
    classifier, _, _, _ = classifier_module

    classifier.classifier("a.jpg", "efficientnet")

    assert classifier.DEVICE == "device:cpu"
    classifier._loaded_models["efficientnet"].to.assert_called_once_with("device:cpu")


def test_classifier_falls_back_to_pretrained_flag(classifier_module, monkeypatch):
    # torchvision older than 0.13 has no weights enums.
    classifier, built, tv_models, _ = classifier_module
    monkeypatch.delattr(tv_models, "AlexNet_Weights")

    classifier.classifier("a.jpg", "alexnet")

    assert built == [("alexnet", {"pretrained": True})]


def test_classifier_rejects_unknown_model(classifier_module):
    classifier, _, _, _ = classifier_module
    with pytest.raises(ValueError, match="Unknown model"):
        classifier.classifier("a.jpg", "vgg19")


def test_architecture_lists_match(classifier_module):
    # get_input_args keeps its own copy so parsing arguments doesn't need PyTorch.
    classifier, _, _, _ = classifier_module
    assert tuple(classifier.ARCHITECTURES) == get_input_args.ARCHITECTURES


def test_predict_batch_splits_into_batches_and_keeps_order(classifier_module):
    classifier, _, _, pil = classifier_module
    paths = ["img{}.jpg".format(i) for i in range(5)]

    results = classifier.predict_batch(paths, "vgg", k=2, batch_size=2)

    assert classifier.stacked == [2, 2, 1]
    assert len(results) == 5
    assert all(r == [("golden retriever", 0.90), ("Labrador retriever", 0.06)] for r in results)
    opened = [call.args[0] for call in pil.Image.open.call_args_list]
    assert opened == paths


def test_predict_batch_empty(classifier_module):
    classifier, _, _, _ = classifier_module
    assert classifier.predict_batch([], "vgg") == []


def test_parameter_count(classifier_module):
    classifier, _, _, _ = classifier_module
    assert classifier.parameter_count("alexnet") == 123

# Makes the modules in workspace/ importable from the tests, since the
# project code imports its siblings by bare module name, and provides a
# stand-in for classifier.py so tests never need PyTorch.
import importlib
import os
import sys
import types

import pytest

WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "workspace"))
sys.path.insert(0, WORKSPACE)


@pytest.fixture
def fake_classifier(monkeypatch):
    """
    Replaces the classifier module with a fake whose predict functions
    answer from a dict keyed by image filename, and whose parameter_count()
    is always 1234567. Yields (labels, calls): fill in labels with either a
    class name (confidence 1.0) or a list of (class name, probability)
    guesses; calls records every (img_path, model_name) received. Modules
    that import classifier are re-imported so they pick up the fake.
    """
    labels = {}
    calls = []

    def predict(img_path, model_name, k=1):
        calls.append((img_path, model_name))
        guesses = labels[os.path.basename(img_path)]
        if isinstance(guesses, str):
            guesses = [(guesses, 1.0)]
        return guesses[:k]

    def predict_batch(img_paths, model_name, k=1, batch_size=16):
        return [predict(path, model_name, k) for path in img_paths]

    fake = types.ModuleType("classifier")
    fake.predict = predict
    fake.predict_batch = predict_batch
    fake.classifier = lambda img_path, model_name: predict(img_path, model_name)[0][0]
    fake.parameter_count = lambda model_name: 1234567
    monkeypatch.setitem(sys.modules, "classifier", fake)
    for name in ("classify_images", "check_images"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield labels, calls


@pytest.fixture
def import_fresh(monkeypatch):
    """Imports a module from scratch, restoring the old one afterwards."""
    def _import(name):
        monkeypatch.delitem(sys.modules, name, raising=False)
        return importlib.import_module(name)
    return _import

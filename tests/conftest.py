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
    Replaces the classifier module with a fake whose classifier() returns
    labels from a dict keyed by image filename. Yields (labels, calls):
    fill in labels; calls records every (img_path, model_name) received.
    Modules that import classifier are re-imported so they pick up the fake.
    """
    labels = {}
    calls = []

    def classifier(img_path, model_name):
        calls.append((img_path, model_name))
        return labels[os.path.basename(img_path)]

    fake = types.ModuleType("classifier")
    fake.classifier = classifier
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

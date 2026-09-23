"""Tests for the pipeline steps that don't need PyTorch.

classify_images() and classifier() load pretrained CNNs, so they are not
covered here; the results dictionaries below stand in for their output.
"""
import os
import sys

import pytest

from adjust_results4_isadog import adjust_results4_isadog
from calculates_results_stats import calculates_results_stats
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels

WORKSPACE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "workspace"))
PET_IMAGES = os.path.join(WORKSPACE, "pet_images")
DOGNAMES = os.path.join(WORKSPACE, "dognames.txt")


def test_get_input_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_images.py"])
    args = get_input_args()
    assert args.dir == "pet_images/"
    assert args.arch == "vgg"
    assert args.dogfile == "dognames.txt"


def test_get_input_args_overrides(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_images.py", "--dir", "imgs/",
                                      "--arch", "resnet", "--dogfile", "d.txt"])
    args = get_input_args()
    assert (args.dir, args.arch, args.dogfile) == ("imgs/", "resnet", "d.txt")


def test_get_pet_labels_formats_filenames(tmp_path):
    for name in ["Boston_terrier_02259.jpg", "cat_07.jpg",
                 "great_horned_owl_02.jpg"]:
        (tmp_path / name).touch()

    results = get_pet_labels(str(tmp_path))

    assert results == {
        "Boston_terrier_02259.jpg": ["boston terrier"],
        "cat_07.jpg": ["cat"],
        "great_horned_owl_02.jpg": ["great horned owl"],
    }


def test_get_pet_labels_on_project_images():
    results = get_pet_labels(PET_IMAGES)
    assert len(results) == 40
    assert results["Dalmatian_04068.jpg"] == ["dalmatian"]
    assert results["German_shepherd_dog_04890.jpg"] == ["german shepherd dog"]


def test_adjust_results4_isadog_sets_dog_flags():
    results = {
        "a.jpg": ["dalmatian", "dalmatian, coach dog, carriage dog", 1],
        "b.jpg": ["beagle", "tabby, tabby cat", 0],
        "c.jpg": ["cat", "chihuahua", 0],
        "d.jpg": ["cat", "tabby, tabby cat", 1],
    }

    adjust_results4_isadog(results, DOGNAMES)

    assert results["a.jpg"][3:] == [1, 1]
    assert results["b.jpg"][3:] == [1, 0]
    assert results["c.jpg"][3:] == [0, 1]
    assert results["d.jpg"][3:] == [0, 0]


def test_adjust_results4_isadog_splits_comma_separated_names(tmp_path):
    dogfile = tmp_path / "dogs.txt"
    dogfile.write_text("chihuahua\nmaltese dog, maltese terrier, maltese\n")
    results = {"m.jpg": ["maltese", "maltese terrier", 1]}

    adjust_results4_isadog(results, str(dogfile))

    assert results["m.jpg"][3:] == [1, 1]


def test_calculates_results_stats_counts_and_percentages():
    results = {
        # dog, correct breed
        "a.jpg": ["dalmatian", "dalmatian, coach dog", 1, 1, 1],
        # dog, recognized as a dog but wrong breed
        "b.jpg": ["beagle", "walker hound", 0, 1, 1],
        # not a dog, correct
        "c.jpg": ["cat", "tabby cat, cat", 1, 0, 0],
        # dog, correct breed
        "d.jpg": ["collie", "collie", 1, 1, 1],
    }

    stats = calculates_results_stats(results)

    assert stats["n_images"] == 4
    assert stats["n_dogs_img"] == 3
    assert stats["n_notdogs_img"] == 1
    assert stats["n_match"] == 3
    assert stats["n_correct_dogs"] == 3
    assert stats["n_correct_breed"] == 2
    assert stats["pct_match"] == pytest.approx(75.0)
    assert stats["pct_correct_dogs"] == pytest.approx(100.0)
    assert stats["pct_correct_breed"] == pytest.approx(200.0 / 3)
    assert stats["pct_correct_notdogs"] == pytest.approx(100.0)


def test_calculates_results_stats_handles_no_dogs():
    results = {"c.jpg": ["cat", "tabby cat, cat", 1, 0, 0]}

    stats = calculates_results_stats(results)

    assert stats["pct_correct_dogs"] == 0.0
    assert stats["pct_correct_breed"] == 0.0

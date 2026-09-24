"""Tests for the pipeline steps: argument parsing, pet labels, classifying,
dog flags and statistics. See conftest.py for the fake classifier."""
import os
import sys

import pytest

from adjust_results4_isadog import adjust_results4_isadog, load_dognames
from calculates_results_stats import calculates_results_stats, pct
from get_input_args import get_input_args
from get_pet_labels import get_pet_labels, label_from_filename

from conftest import WORKSPACE

PET_IMAGES = os.path.join(WORKSPACE, "pet_images")
DOGNAMES = os.path.join(WORKSPACE, "dognames.txt")


# --- get_input_args ---------------------------------------------------------

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


def test_get_input_args_optional_extras(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_images.py"])
    args = get_input_args()
    assert (args.topk, args.csv) == (0, None)

    monkeypatch.setattr(sys, "argv", ["check_images.py", "--arch", "efficientnet",
                                      "--topk", "3", "--csv", "out.csv"])
    args = get_input_args()
    assert (args.arch, args.topk, args.csv) == ("efficientnet", 3, "out.csv")


@pytest.mark.parametrize("topk", ["-1", "1001"])
def test_get_input_args_rejects_out_of_range_topk(monkeypatch, capsys, topk):
    # There are only 1000 ImageNet classes; catch this before the model runs.
    monkeypatch.setattr(sys, "argv", ["check_images.py", "--topk", topk])
    with pytest.raises(SystemExit):
        get_input_args()
    assert "--topk" in capsys.readouterr().err


def test_get_input_args_rejects_unknown_arch(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_images.py", "--arch", "vgg19"])
    with pytest.raises(SystemExit):
        get_input_args()
    assert "invalid choice" in capsys.readouterr().err


# --- get_pet_labels ---------------------------------------------------------

@pytest.mark.parametrize("filename, label", [
    ("Boston_terrier_02259.jpg", "boston terrier"),
    ("cat_07.jpg", "cat"),
    ("great_horned_owl_02.jpg", "great horned owl"),
    ("German_Shepherd_01.JPG", "german shepherd"),
    # No numeric id: the extension used to stay glued to the last word.
    ("cat.jpg", "cat"),
])
def test_label_from_filename(filename, label):
    assert label_from_filename(filename) == label


def test_get_pet_labels_formats_filenames(tmp_path):
    for name in ["Boston_terrier_02259.jpg", "cat_07.jpg", "great_horned_owl_02.jpg"]:
        (tmp_path / name).touch()

    assert get_pet_labels(str(tmp_path)) == {
        "Boston_terrier_02259.jpg": ["boston terrier"],
        "cat_07.jpg": ["cat"],
        "great_horned_owl_02.jpg": ["great horned owl"],
    }


def test_get_pet_labels_skips_hidden_and_non_image_files(tmp_path):
    for name in ["beagle_01.jpg", ".DS_Store", "notes.txt", "._beagle_01.jpg"]:
        (tmp_path / name).touch()

    assert get_pet_labels(str(tmp_path)) == {"beagle_01.jpg": ["beagle"]}


def test_get_pet_labels_prints_no_false_warning(tmp_path, capsys):
    # Regression: a for/else used to print "already exists" on every run.
    (tmp_path / "beagle_01.jpg").touch()
    get_pet_labels(str(tmp_path))
    assert "already exists" not in capsys.readouterr().out


def test_get_pet_labels_on_project_images():
    results = get_pet_labels(PET_IMAGES)
    assert len(results) == 40
    assert list(results) == sorted(results)
    assert results["Dalmatian_04068.jpg"] == ["dalmatian"]
    assert results["German_shepherd_dog_04890.jpg"] == ["german shepherd dog"]


# --- classify_images --------------------------------------------------------

@pytest.mark.parametrize("pet_label, classifier_label, expected", [
    ("dalmatian", "dalmatian, coach dog, carriage dog", True),
    ("cat", "tabby, tabby cat", True),
    ("german shepherd dog", "german shepherd, german shepherd dog, alsatian", True),
    # Regression: substring matching used to accept these.
    ("cat", "polecat, fitch, foulmart, foumart, mustela putorius", False),
    ("", "tabby cat", False),
])
def test_labels_match(fake_classifier, import_fresh, pet_label, classifier_label, expected):
    classify_images = import_fresh("classify_images")
    assert classify_images.labels_match(pet_label, classifier_label) is expected


def test_classify_images_appends_label_and_match(fake_classifier, import_fresh):
    labels, calls = fake_classifier
    labels.update({"cat_01.jpg": "  Tabby, tabby cat ", "cat_02.jpg": "Polecat, fitch"})
    classify_images = import_fresh("classify_images")
    results = {"cat_01.jpg": ["cat"], "cat_02.jpg": ["cat"]}

    # No trailing slash: paths are built with os.path.join.
    predictions = classify_images.classify_images("some/dir", results, "resnet")

    assert results == {"cat_01.jpg": ["cat", "tabby, tabby cat", 1],
                       "cat_02.jpg": ["cat", "polecat, fitch", 0]}
    assert predictions == {"cat_01.jpg": [("tabby, tabby cat", 1.0)],
                           "cat_02.jpg": [("polecat, fitch", 1.0)]}
    assert sorted(calls) == [(os.path.join("some/dir", "cat_01.jpg"), "resnet"),
                             (os.path.join("some/dir", "cat_02.jpg"), "resnet")]


def test_classify_images_batches_and_keeps_order(fake_classifier, import_fresh, monkeypatch):
    labels, _ = fake_classifier
    names = ["dog_{:02d}.jpg".format(i) for i in range(5)]
    labels.update({name: "Beagle" for name in names})
    classify_images = import_fresh("classify_images")
    batches = []
    real_predict_batch = classify_images.predict_batch

    def spy(paths, model, k=1, batch_size=16):
        batches.append((len(paths), batch_size))
        return real_predict_batch(paths, model, k, batch_size)

    monkeypatch.setattr(classify_images, "predict_batch", spy)
    results = {name: ["dog"] for name in names}

    classify_images.classify_images("d", results, "vgg", batch_size=2)

    # All images go to predict_batch in one call; it splits them into batches.
    assert batches == [(5, 2)]
    assert list(results) == names
    assert all(results[name][1:] == ["beagle", 0] for name in names)


def test_classify_images_keeps_top_k_guesses(fake_classifier, import_fresh):
    labels, _ = fake_classifier
    labels["pug_01.jpg"] = [("Boxer", 0.6), ("Pug, pug-dog", 0.3), ("Bull mastiff", 0.1)]
    classify_images = import_fresh("classify_images")
    results = {"pug_01.jpg": ["pug"]}

    predictions = classify_images.classify_images("d", results, "vgg", top_k=2)

    # The top guess is the classifier label, so this is not a match...
    assert results["pug_01.jpg"] == ["pug", "boxer", 0]
    # ...but the right answer is the second guess.
    assert predictions["pug_01.jpg"] == [("boxer", 0.6), ("pug, pug-dog", 0.3)]


# --- adjust_results4_isadog -------------------------------------------------

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


def test_load_dognames_splits_every_line_including_the_first(tmp_path):
    # Regression: the first line used to be stored whole and never split.
    dogfile = tmp_path / "dogs.txt"
    dogfile.write_text("maltese dog, maltese terrier, maltese\nchihuahua\n\n")

    assert load_dognames(str(dogfile)) == {
        "maltese dog, maltese terrier, maltese", "maltese dog",
        "maltese terrier", "maltese", "chihuahua"}


def test_dognames_includes_labradoodle():
    # A labradoodle isn't an ImageNet class, but it is a dog.
    assert "labradoodle" in load_dognames(DOGNAMES)


def test_load_dognames_prints_nothing(capsys):
    load_dognames(DOGNAMES)
    assert capsys.readouterr().out == ""


# --- calculates_results_stats -----------------------------------------------

def test_pct():
    assert pct(1, 4) == 25.0
    assert pct(3, 0) == 0.0


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


def test_calculates_results_stats_counts_not_dogs_called_dogs_as_wrong():
    # Regression: every non-dog image used to count as correct.
    results = {
        "c.jpg": ["cat", "tabby cat, cat", 1, 0, 0],
        "f.jpg": ["fox", "chihuahua", 0, 0, 1],
    }

    stats = calculates_results_stats(results)

    assert stats["n_correct_notdogs"] == 1
    assert stats["pct_correct_notdogs"] == pytest.approx(50.0)


def test_calculates_results_stats_breed_uses_match_flag():
    # Regression: breed accuracy used a substring test ("pug" in "pug-dog")
    # instead of the match flag at index 2.
    results = {"p.jpg": ["pug", "pug-dog, pug", 0, 1, 1]}

    assert calculates_results_stats(results)["n_correct_breed"] == 0


def test_calculates_results_stats_handles_empty_groups(capsys):
    stats = calculates_results_stats({"c.jpg": ["cat", "tabby cat, cat", 1, 0, 0]})

    assert stats["pct_correct_dogs"] == 0.0
    assert stats["pct_correct_breed"] == 0.0
    assert capsys.readouterr().out == ""

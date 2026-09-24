"""Tests for plot_results.py: the charts render, read the right values, and
the scatter labels don't overlap."""
import contextlib
import csv
import io
import itertools

import matplotlib

matplotlib.use("Agg")  # render off-screen

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

import plot_results as pr  # noqa: E402
from export_results import CSV_COLUMNS  # noqa: E402
from print_results import print_model_speed, print_results  # noqa: E402

BASE = {"n_images": 40, "n_dogs_img": 30, "n_notdogs_img": 10, "n_match": 30,
        "n_correct_dogs": 30, "n_correct_notdogs": 10, "n_correct_breed": 27,
        "pct_match": 82.5, "pct_correct_dogs": 100.0, "pct_correct_notdogs": 90.0,
        "pct_correct_breed": 90.0}
# Clustered like real results: resnet, resnet50 and efficientnet close together.
SPEEDS = {"resnet": (0.046, 90.0), "alexnet": (0.022, 80.0), "vgg": (0.272, 93.33),
          "resnet50": (0.063, 93.33), "efficientnet": (0.040, 90.0)}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def write_output(path, model, stats, sec_per_image=None):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_results({}, stats, model)
        if sec_per_image is not None:
            print_model_speed(10_000_000, sec_per_image * stats["n_images"], stats["n_images"])
    path.write_text(buf.getvalue())


@pytest.fixture
def output_files(tmp_path):
    files = []
    for model, (sec, breed) in SPEEDS.items():
        path = tmp_path / (model + "_pet-images.txt")
        write_output(path, model, dict(BASE, pct_correct_breed=breed), sec)
        files.append(str(path))
    return files


@pytest.mark.parametrize("value, text", [(100.0, "100%"), (82.5, "82.5%"),
                                         (93.333, "93.3%"), (0.0, "0%")])
def test_percent_label(value, text):
    assert pr._percent_label(value) == text


def test_plot_accuracy(output_files):
    fig = pr.plot_accuracy(output_files, "Test")

    assert len(fig.axes) == len(pr.ACCURACY_PANELS)
    breed_panel = fig.axes[2]
    assert breed_panel.get_title(loc="left") == "Dog breeds named correctly"
    widths = [bar.get_width() for bar in breed_panel.patches]
    assert widths == [breed for _, breed in SPEEDS.values()]
    labels = [text.get_text() for text in breed_panel.texts]
    assert labels == ["90%", "80%", "93.3%", "93.3%", "90%"]
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == list(SPEEDS)


def test_plot_speed_vs_accuracy_labels_every_point_without_overlap(output_files):
    fig = pr.plot_speed_vs_accuracy(output_files)
    ax = fig.axes[0]
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    labels = [t for t in ax.texts if t.get_text()]
    assert sorted(t.get_text() for t in labels) == sorted(SPEEDS)
    boxes = [t.get_window_extent(renderer) for t in labels]
    for a, b in itertools.combinations(boxes, 2):
        assert not a.overlaps(b)


def test_plot_speed_vs_accuracy_skips_models_without_speed(tmp_path):
    path = tmp_path / "vgg_pet-images.txt"
    write_output(path, "vgg", BASE)  # no speed line, like the replayed outputs

    fig = pr.plot_speed_vs_accuracy([str(path)])

    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "No measured speeds yet: run the batch scripts" in texts


def write_csv(path, rows):
    with open(path, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def test_plot_mistakes(tmp_path):
    Image.new("RGB", (40, 20), "white").save(tmp_path / "beagle_01.jpg")
    Image.new("RGB", (20, 40), "white").save(tmp_path / "cat_01.jpg")
    write_csv(tmp_path / "r.csv", [
        ("vgg", "beagle_01.jpg", "beagle", "walker hound, walker foxhound", "0.55", "0", "1", "1",
         "walker hound, walker foxhound (55.0%); beagle (40.0%); basset (3.0%)"),
        ("vgg", "cat_01.jpg", "cat", "tabby, tabby cat", "0.81", "1", "0", "0",
         "tabby, tabby cat (81.0%)"),
    ])

    fig = pr.plot_mistakes(str(tmp_path / "r.csv"), str(tmp_path))

    titles = [ax.get_title(loc="left") for ax in fig.axes if ax.get_title(loc="left")]
    assert titles == ["real: beagle\nwalker hound (55.0%)\nbeagle (40.0%)\nbasset (3.0%)"]


def test_plot_mistakes_none_wrong(tmp_path):
    write_csv(tmp_path / "r.csv", [("vgg", "cat_01.jpg", "cat", "tabby cat", "0.81", "1", "0", "0",
                                    "tabby cat (81.0%)")])
    assert pr.plot_mistakes(str(tmp_path / "r.csv"), str(tmp_path)) is None

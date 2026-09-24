"""Tests for the printed output: print_results, print_model_tables and the
runtime formatting in check_images."""
import csv
import os
import sys

import pytest

from export_results import CSV_COLUMNS, write_results_csv
from print_model_tables import PET_IMAGE_FILES, parse_results_file, print_models_table
from print_results import print_model_speed, print_results, print_top_predictions

from conftest import WORKSPACE

RESULTS = {
    "a.jpg": ["dalmatian", "dalmatian, coach dog", 1, 1, 1],
    "b.jpg": ["beagle", "walker hound", 0, 1, 1],
    "c.jpg": ["cat", "chihuahua", 0, 0, 1],
    "d.jpg": ["collie", "tabby cat", 0, 1, 0],
}
STATS = {
    "n_images": 4, "n_dogs_img": 3, "n_notdogs_img": 1, "n_match": 1,
    "n_correct_dogs": 2, "n_correct_notdogs": 0, "n_correct_breed": 1,
    "pct_match": 25.0, "pct_correct_dogs": 200 / 3, "pct_correct_breed": 100 / 3,
    "pct_correct_notdogs": 0.0,
}


def write_results(path, model, stats):
    """Writes a file in the format check_images.py produces."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print("Command Line Arguments: ...")
        print_results({}, stats, model)
    path.write_text(buf.getvalue())


# --- print_results ----------------------------------------------------------

def test_print_results_summary(capsys):
    print_results(RESULTS, STATS, "vgg")
    out = capsys.readouterr().out

    assert "*** Results Summary for CNN Model Architecture VGG ***" in out
    assert "N Images            :   4" in out
    assert "N Not-Dog Images    :   1" in out
    assert "% Correct Dogs      :  66.67" in out
    assert "% Correct Not-a-Dog :   0.00" in out
    assert "INCORRECT" not in out


def test_print_results_lists_mistakes(capsys):
    print_results(RESULTS, STATS, "vgg", True, True)
    out = capsys.readouterr().out

    dogs_section = out.split("INCORRECT Dog/NOT Dog Assignments:")[1].split("INCORRECT Dog Breed")[0]
    assert "cat" in dogs_section and "chihuahua" in dogs_section
    assert "collie" in dogs_section and "tabby cat" in dogs_section
    assert "beagle" not in dogs_section

    breed_section = out.split("INCORRECT Dog Breed Assignment:")[1]
    assert "beagle" in breed_section and "walker hound" in breed_section
    assert "dalmatian" not in breed_section


# --- print_model_tables -----------------------------------------------------

def test_parse_results_file_round_trips_print_results(tmp_path):
    path = tmp_path / "vgg.txt"
    write_results(path, "vgg", STATS)

    model, stats = parse_results_file(str(path))

    assert model == "vgg"
    assert stats["n_images"] == 4
    assert stats["pct_correct_dogs"] == pytest.approx(66.67)
    assert set(stats) == {"n_images", "n_dogs_img", "n_notdogs_img", "pct_match",
                          "pct_correct_dogs", "pct_correct_breed", "pct_correct_notdogs"}


def test_print_models_table(tmp_path, capsys):
    files = []
    for model in ("resnet", "alexnet", "vgg"):
        path = tmp_path / (model + ".txt")
        write_results(path, model, dict(STATS, pct_match={"resnet": 10.0}.get(model, 25.0)))
        files.append(str(path))
    capsys.readouterr()

    assert print_models_table(files) is True
    lines = capsys.readouterr().out.splitlines()

    assert lines[0] == "N Images            |   4"
    rows = [line for line in lines if line.startswith(("resnet", "alexnet", "vgg"))]
    assert [r.split()[0] for r in rows] == ["resnet", "alexnet", "vgg"]
    # Columns: model, 4 percentages, params, sec/image, runtime.
    assert rows[0].split("|")[4].strip() == "10.00%"
    assert rows[1].split("|")[2].strip() == "66.67%"
    # These files have no size/speed/runtime lines.
    assert [cell.strip() for cell in rows[0].split("|")[5:]] == ["n/a", "n/a", "n/a"]


def test_print_models_table_size_speed_and_runtime(tmp_path, capsys):
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_results({}, STATS, "vgg")
        print_model_speed(138357544, 8.0, 40)
        print("\n** Total Elapsed Runtime: 0:01:05")
    path = tmp_path / "vgg.txt"
    path.write_text(buf.getvalue())

    model, stats = parse_results_file(str(path))
    assert stats["params_m"] == pytest.approx(138.4)
    assert stats["sec_per_image"] == pytest.approx(0.2)
    assert stats["runtime"] == 65

    print_models_table([str(path)])
    row = capsys.readouterr().out.splitlines()[-1]
    assert [cell.strip() for cell in row.split("|")[5:]] == ["138.4", "0.200", "0:01:05"]


def test_print_models_table_skips_missing_files(tmp_path, capsys):
    path = tmp_path / "vgg.txt"
    write_results(path, "vgg", STATS)
    capsys.readouterr()

    assert print_models_table([str(tmp_path / "resnet50.txt"), str(path)]) is True
    rows = [line for line in capsys.readouterr().out.splitlines() if "|" in line]
    assert [r.split()[0] for r in rows[4:]] == ["vgg"]


def test_print_models_table_all_files_missing(tmp_path, capsys, caplog):
    assert print_models_table([str(tmp_path / "nope.txt")]) is False
    assert capsys.readouterr().out == ""
    assert "nope.txt" in caplog.text


def test_pet_image_files_cover_every_architecture():
    assert PET_IMAGE_FILES == ["resnet_pet-images.txt", "alexnet_pet-images.txt",
                               "vgg_pet-images.txt", "resnet50_pet-images.txt",
                               "efficientnet_pet-images.txt"]


def test_saved_output_files_parse():
    # The committed *_pet-images.txt / *_uploaded-images.txt files must stay
    # readable by print_model_tables.py.
    for name in os.listdir(WORKSPACE):
        if name.endswith(("_pet-images.txt", "_uploaded-images.txt")):
            model, stats = parse_results_file(os.path.join(WORKSPACE, name))
            assert model == name.split("_")[0]
            # 3 counts, 4 percentages, the model size and the total runtime.
            # (The replayed runs have no measured seconds per image.)
            assert len(stats) == 9 and "params_m" in stats and "runtime" in stats


# --- check_images -----------------------------------------------------------

@pytest.mark.parametrize("seconds, text", [
    (0, "0:00:00"), (32.9, "0:00:32"), (61, "0:01:01"), (3 * 3600 + 5, "3:00:05"),
])
def test_format_runtime(fake_classifier, import_fresh, seconds, text):
    check_images = import_fresh("check_images")
    assert check_images.format_runtime(seconds) == text


def run_check_images(monkeypatch, import_fresh, labels, image_dir, *args):
    """Runs check_images.main() on image_dir with the fake classifier."""
    names = [name for name in labels]
    for name in names:
        (image_dir / name).touch()
    monkeypatch.setattr(sys, "argv", ["check_images.py", "--dir", str(image_dir),
                                      "--dogfile", os.path.join(WORKSPACE, "dognames.txt"),
                                      *args])
    import_fresh("check_images").main()


def test_check_images_end_to_end(fake_classifier, import_fresh, monkeypatch, tmp_path, capsys):
    labels, _ = fake_classifier
    labels.update({
        "Beagle_01.jpg": [("Walker hound, Walker foxhound", 0.55), ("beagle", 0.40)],
        "Dalmatian_01.jpg": [("dalmatian, coach dog, carriage dog", 0.97), ("pointer", 0.01)],
        "cat_01.jpg": [("tabby, tabby cat", 0.81), ("tiger cat", 0.12)],
    })
    images = tmp_path / "images"
    images.mkdir()
    out_csv = tmp_path / "out.csv"

    run_check_images(monkeypatch, import_fresh, labels, images,
                     "--arch", "resnet50", "--topk", "2", "--csv", str(out_csv))
    out = capsys.readouterr().out

    assert "*** Results Summary for CNN Model Architecture RESNET50 ***" in out
    assert "Parameters (M)      :      1.2" in out
    assert "Seconds per Image   :" in out
    assert "% Correct Breed     :  50.00" in out
    # Top guesses are shown for the one mismatch only.
    top = out.split("Top 2 guesses for images whose labels don't match:")[1]
    assert "Beagle_01.jpg (real: beagle)" in top
    assert " 40.00%  beagle" in top
    assert "Dalmatian_01.jpg" not in top

    with open(out_csv, newline="") as infile:
        rows = list(csv.DictReader(infile))
    assert [r["filename"] for r in rows] == ["Beagle_01.jpg", "Dalmatian_01.jpg", "cat_01.jpg"]
    assert rows[1]["confidence"] == "0.9700"
    assert rows[1]["labels_match"] == "1"


def test_check_images_without_extras(fake_classifier, import_fresh, monkeypatch, tmp_path, capsys):
    labels, _ = fake_classifier
    labels["cat_01.jpg"] = "Polecat"
    run_check_images(monkeypatch, import_fresh, labels, tmp_path)

    out = capsys.readouterr().out
    assert "Top " not in out and "Saved per-image results" not in out


# --- print_top_predictions / write_results_csv -------------------------------

PREDICTIONS = {
    "a.jpg": [("dalmatian, coach dog", 0.9)],
    "b.jpg": [("walker hound", 0.5), ("beagle", 0.3), ("basset", 0.1)],
    "c.jpg": [("chihuahua", 0.6), ("cat", 0.2)],
    "d.jpg": [("tabby cat", 0.7)],
}


def test_print_top_predictions(capsys):
    print_top_predictions(RESULTS, PREDICTIONS, 2)
    lines = capsys.readouterr().out.strip().splitlines()

    assert lines[0] == "Top 2 guesses for images whose labels don't match:"
    assert lines[1:4] == ["b.jpg (real: beagle)", "   50.00%  walker hound", "   30.00%  beagle"]
    assert not any(line.startswith("a.jpg") for line in lines)


def test_print_top_predictions_all_match(capsys):
    print_top_predictions({"a.jpg": RESULTS["a.jpg"]}, PREDICTIONS, 3)
    assert capsys.readouterr().out == ""


def test_write_results_csv(tmp_path):
    path = tmp_path / "r.csv"
    write_results_csv(str(path), RESULTS, PREDICTIONS, "vgg")

    with open(path, newline="") as infile:
        rows = list(csv.reader(infile))
    assert tuple(rows[0]) == CSV_COLUMNS
    assert rows[2] == ["vgg", "b.jpg", "beagle", "walker hound", "0.5000", "0", "1", "1",
                       "walker hound (50.0%); beagle (30.0%); basset (10.0%)"]
    assert len(rows) == 5


def test_print_model_speed(capsys):
    print_model_speed(61100840, 3.0, 40)
    assert capsys.readouterr().out.splitlines()[1:] == [
        "*** Model Size and Speed ***",
        "Parameters (M)      :     61.1",
        "Seconds per Image   :    0.075"]


def test_print_model_speed_no_images(capsys):
    print_model_speed(5288548, 0.0, 0)
    assert "Seconds per Image   :    0.000" in capsys.readouterr().out

"""Tests for the printed output: print_results, print_model_tables and the
runtime formatting in check_images."""
import os

import pytest

from print_model_tables import parse_results_file, print_models_table
from print_results import print_results

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
    assert rows[0].split("|")[-1].strip() == "10.00%"
    assert rows[1].split("|")[2].strip() == "66.67%"


def test_print_models_table_missing_file(tmp_path, capsys, caplog):
    assert print_models_table([str(tmp_path / "nope.txt")]) is False
    assert capsys.readouterr().out == ""
    assert "nope.txt" in caplog.text


def test_saved_output_files_parse():
    # The committed *_pet-images.txt / *_uploaded-images.txt files must stay
    # readable by print_model_tables.py.
    for name in os.listdir(WORKSPACE):
        if name.endswith(("_pet-images.txt", "_uploaded-images.txt")):
            model, stats = parse_results_file(os.path.join(WORKSPACE, name))
            assert model == name.split("_")[0]
            assert len(stats) == 7


# --- check_images -----------------------------------------------------------

@pytest.mark.parametrize("seconds, text", [
    (0, "0:00:00"), (32.9, "0:00:32"), (61, "0:01:01"), (3 * 3600 + 5, "3:00:05"),
])
def test_format_runtime(fake_classifier, import_fresh, seconds, text):
    check_images = import_fresh("check_images")
    assert check_images.format_runtime(seconds) == text

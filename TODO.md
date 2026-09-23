# TODO: code improvements

Improvements found while reviewing the project. The cleanup on this branch
kept program behavior the same, so none of these are fixed yet. Items are
grouped by priority; file references are relative to `workspace/`.

## Bugs (wrong results or misleading output)

- [ ] **`calculates_results_stats.py`: `n_correct_notdogs` counts every non-dog image.**
      The `else` branch increments the counter without checking that the
      classifier also said "not a dog" (`results_dic[key][4] == 0`), so
      `pct_correct_notdogs` is always 100%. The same miscount also hides
      not-dog images misclassified as dogs from the "INCORRECT Dog/NOT Dog"
      list in `print_results.py`, because that list is gated on
      `n_correct_dogs + n_correct_notdogs != n_images`.
- [ ] **`get_pet_labels.py`: a false "already exists" warning prints on every run.**
      The `else:` is attached to the `for` loop (a `for ... else`), not the
      `if`, so it runs whenever the loop finishes. Every saved output file
      starts with a bogus `** Warning: Key= ... already exists` line.
      Indent the `else` under the `if` (or drop it, since `listdir` never
      returns duplicate names).
- [ ] **`calculates_results_stats.py`: breed accuracy uses a substring test.**
      `n_correct_breed` checks `pet_label in classifier_label` rather than the
      match flag at index 2, which can disagree with `n_match`. Use
      `results_dic[key][3] == 1 and results_dic[key][2] == 1`.
- [ ] **`classify_images.py`: label matching is a raw substring test.**
      `truth in model_label` treats `"cat"` as matching `"polecat, fitch, ..."`.
      Split the classifier label on
      commas and compare whole names (or match on word boundaries).
- [ ] **`adjust_results4_isadog.py`: a first line with several names isn't split.**
      The first line of the dog file is added whole and never split on
      commas; only later lines are. Load every line the same way.
- [ ] **`print_model_tables.py`: parsing depends on line order.**
      `table_dic[key].append(...)` fails with `KeyError` if a `pct_*` line
      appears before `pct_match`, and the columns are chosen by list index.
      Store the values by name (e.g. `table_dic[key]['pct_match']`).

## Robustness

- [ ] Build paths with `os.path.join(images_dir, key)` instead of
      `images_dir + key`, which breaks when `--dir` has no trailing slash.
- [ ] Skip hidden or non-image files in `get_pet_labels` (e.g. `.DS_Store`),
      which would otherwise be sent to the classifier and crash it.
- [ ] Validate `--arch` with `choices=['resnet', 'alexnet', 'vgg']` in
      `get_input_args` so a typo gives a clear error, not a `KeyError`.
- [ ] `classifier.py` opens `imagenet1000_clsid_to_human.txt` relative to the
      current directory, so every script must be run from inside
      `workspace/`. Resolve the path from `__file__`.
- [ ] `print_model_tables.py` hard-codes the `'alexnet'` key for the image
      counts; use whichever model was read first.

## Maintainability and style

- [ ] Replace the positional list in `results_dic` (`[label, clf_label, match,
      is_dog, clf_is_dog]`) with a small dataclass or named tuple, so code
      reads `r.pet_is_dog` instead of `results_dic[key][3]`.
- [ ] Simplify `adjust_results4_isadog`: read the dog names into a `set`
      with a single loop that splits on commas; drop the empty-dict special
      case and the "Key is already in" prints.
- [ ] Simplify `get_pet_labels` to one loop:
      `" ".join(w for w in name.lower().split("_") if w.isalpha())`.
- [ ] Replace the `#Debug ZeroDivisionError prevented` prints in
      `calculates_results_stats.py` with a helper such as
      `pct(part, whole)` that returns 0.0 when `whole == 0`.
- [ ] Format the runtime with zero padding (`"%d:%02d:%02d"`); it currently
      prints `0:0:32`.
- [ ] Print friendlier statistic names in `print_results` (e.g.
      `% Correct Dogs` rather than `pct_correct_dogs`). If you change these
      lines, update `print_model_tables.py` too, since it parses them.
- [ ] Move the main flow of `print_model_tables.py` into a `main()` and let
      `check_images.py` skip the tables when the batch output files don't
      exist yet, instead of printing "Missing; exiting".
- [ ] Use `logging` instead of `print` for warnings and diagnostics.
- [ ] Turn on full flake8 (PEP 8) in CI once the style issues above are fixed;
      CI currently checks only syntax errors and undefined names.

## Dependencies

- [ ] `classifier.py` uses `pretrained=True`, which newer torchvision
      deprecates; switch to `weights=models.ResNet18_Weights.DEFAULT` etc.
- [ ] Remove the PyTorch < 0.4 `Variable` code path in `classifier.py`.
- [ ] Load only the requested model, not all three, at import time.
- [ ] Use `torch.no_grad()` for inference and cache the transform pipeline
      rather than rebuilding it for every image.
- [ ] Add a `requirements.txt` (torch, torchvision, Pillow) with pinned versions.

## Testing

- [ ] Add a test for `classify_images` that monkeypatches `classifier`, so it
      runs without PyTorch.
- [ ] Add a test for `print_results` and `print_model_tables` output
      (pytest's `capsys`).
- [ ] Add regression tests for each bug above as it is fixed.

## Data

- [ ] `uploaded_images/`: `German_Shepherd_01.jpg`/`_02.jpg` and
      `Labradoodle_01.jpg`/`_02.jpg` are the same size and look like identical
      copies. The assignment asks for the second image to be a horizontally
      flipped version.

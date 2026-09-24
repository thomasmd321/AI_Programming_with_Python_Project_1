# Project History

The original 2020 submission is preserved on the `original-code` branch. In
2026 it was cleaned up and extended through these pull requests:

| PR | What changed |
|----|--------------|
| #1 | Cleanup, comments, README, TODO list, tests and CI. Fixed: non-dog accuracy was always 100%; substring label matching; a false warning on every run; the first dog-names line wasn't split; fragile table parsing |
| #2 | Top-k confidence (`--topk`), CSV export (`--csv`), ResNet-50 and EfficientNet-B0, GPU support, Colab notebook; mirrored the duplicate uploaded images; added `labradoodle` to the dog names |
| #3 | Model size and speed comparison, batched inference, charts, CI on Python 3.10/3.12/3.13, Dependabot |
| #4, #5, #7 | Dependabot updates: `actions/checkout` v7, `actions/setup-python` v7, Pillow 12.3 |
| #8 | MIT license; Dependabot stops raising minimum versions |
| #9 | Fixes from a full review: timing included model loading; cut-short output files crashed the table; a test that would fail on real results; `--topk` over 1000 |

## Notable findings along the way

- **Non-dog accuracy was always 100%.** Every non-dog image was counted as
  correct without checking the classifier. Four of the six saved runs
  disagreed with Udacity's own check; after the fix all six agree.
- **The "flipped" uploaded images weren't flipped.** The `_02` photos were
  the `_01` pixels with an EXIF "rotate 180°" tag, which image viewers apply
  but the classifier ignored. They are now true mirror images, and the
  classifier applies EXIF orientation.
- **Labradoodles counted as non-dogs** because the breed isn't an ImageNet
  class. Adding it to `dognames.txt` moved not-a-dog accuracy on the uploaded
  images from 60% to 100%.

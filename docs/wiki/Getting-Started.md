# Getting Started

## Option 1: Google Colab (no setup)

The quickest way to run everything with the real models:

1. Open the [notebook in Colab](https://colab.research.google.com/github/thomasmd321/AI_Programming_with_Python_Project_1/blob/master/notebooks/run_on_colab.ipynb).
2. Optional: *Runtime → Change runtime type → GPU* for a faster run.
3. *Runtime → Run all*.

The notebook clones the repo, installs requirements, runs a smoke test, runs
every model on both image folders, shows the results as tables and charts, and
downloads the output files as `results.zip`. To keep the results, unzip them
into `workspace/` in your clone, then commit and push.

## Option 2: Your own machine

Requirements: Python 3.10+, with PyTorch, torchvision, Pillow and matplotlib.

```bash
git clone https://github.com/thomasmd321/AI_Programming_with_Python_Project_1.git
cd AI_Programming_with_Python_Project_1
pip install -r requirements.txt
cd workspace
```

Run the commands from inside `workspace/`: the default image folder,
dog-names file and batch scripts use paths relative to it.

```bash
# Classify the course images with one model
python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt

# Every model on both folders, then the comparison tables
sh run_models_batch.sh
sh run_models_batch_uploaded.sh
```

The pretrained weights download automatically the first time each model is
used (VGG-16 is about 528 MB). A GPU is used if one is available.

## Try your own images

Put photos in `workspace/uploaded_images/` and name each one after what it
shows: words separated by underscores, then an optional number, e.g.
`Golden_retriever_01.jpg` or `coffee_mug_02.jpg`. Hidden files and files that
aren't images are skipped. Then run `sh run_models_batch_uploaded.sh`.

If the photo is of a dog breed that isn't in `dognames.txt`, add it there, so
the pet counts as a dog (see [[Troubleshooting]]).

Next: [[How It Works]] · [[Command Line Reference]]

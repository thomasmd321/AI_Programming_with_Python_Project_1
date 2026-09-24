# Classifying Pet Images

Project 1 of Udacity's *AI Programming with Python* Nanodegree, by Thomas
Stewart. The program classifies pet images with convolutional neural networks
(CNNs) pretrained on ImageNet. It checks whether each model tells dogs from
non-dogs and names the right breed, then compares the models on accuracy,
size and speed.

The true identity of each pet comes from its filename
(`Boston_terrier_02259.jpg` → `boston terrier`), and the program compares
that label with each model's prediction.

## At a glance

On the 40 course images (30 dogs, 10 non-dogs):

| Model   | Not-a-dog correct | Dogs correct | Breeds correct | Parameters | Runtime |
|---------|------------------:|-------------:|---------------:|-----------:|--------:|
| ResNet  |  90% | 100% | 90.0% |  11.7M | 0:00:05 |
| AlexNet | 100% | 100% | 80.0% |  61.1M | 0:00:03 |
| VGG     | 100% | 100% | 93.3% | 138.4M | 0:00:32 |

**VGG** is the most accurate model, but it's also by far the largest and slowest.
See [[Results]] for the full numbers and how they were produced.

## Pages

- [[Getting Started]]: install, run, or use Google Colab
- [[How It Works]]: the pipeline, step by step
- [[Command Line Reference]]: options, batch scripts and output files
- [[Results]]: accuracy, size and speed for each model
- [[Models]]: the five CNN architectures
- [[Development]]: tests, CI, code layout and contributing
- [[Troubleshooting]]: common problems and fixes
- [[Project History]]: what changed since the original submission

## Links

- [Repository](https://github.com/thomasmd321/AI_Programming_with_Python_Project_1) · [README](https://github.com/thomasmd321/AI_Programming_with_Python_Project_1#readme) · [TODO list](https://github.com/thomasmd321/AI_Programming_with_Python_Project_1/blob/master/TODO.md)
- [Open the notebook in Colab](https://colab.research.google.com/github/thomasmd321/AI_Programming_with_Python_Project_1/blob/master/notebooks/run_on_colab.ipynb)
- License: MIT for Thomas Stewart's code; see [NOTICE.md](https://github.com/thomasmd321/AI_Programming_with_Python_Project_1/blob/master/NOTICE.md)
  for what it doesn't cover.

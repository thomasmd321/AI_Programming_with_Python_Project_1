# Troubleshooting

**`FileNotFoundError: dognames.txt` or `pet_images/`**
Run the commands from inside `workspace/`, or pass full paths with `--dir`
and `--dogfile`.

**The first run of a model is slow**
The pretrained weights are downloaded the first time each model is used
(VGG-16 is about 528 MB) and cached afterwards. The speed figures exclude
this download.

**`argument --arch: invalid choice`**
Use one of `resnet`, `alexnet`, `vgg`, `resnet50`, `efficientnet`.

**`--topk must be between 0 and 1000`**
ImageNet has 1000 classes, so there are at most 1000 guesses to show.

**"Skipping …: it has no complete results summary"**
That model's run was cut short, so its output file is incomplete. Re-run that
model, or delete the file; the table and charts skip it in the meantime.

**A dog counts as "not a dog"**
Its breed isn't in `dognames.txt`, so the pet label isn't recognised as a dog.
Add the breed on its own line in lower case, the way `labradoodle` was added.

**A photo looks upright but is classified sideways**
This was fixed: images are now turned upright using their EXIF orientation
tag before classification.

**Some files in my image folder are ignored**
Hidden files (starting with `.`) and files that aren't `.jpg`, `.jpeg`,
`.png`, `.bmp` or `.gif` are skipped on purpose.

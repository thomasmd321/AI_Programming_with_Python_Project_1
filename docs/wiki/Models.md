# Models

All five models come from torchvision, pretrained on ImageNet (1000 classes),
and are used as-is, without retraining.

| `--arch` | Model | Parameters | Notes |
|----------|-------|-----------:|-------|
| `resnet` | ResNet-18 | 11.7M | Small and fast; part of the Udacity comparison |
| `alexnet` | AlexNet | 61.1M | The 2012 classic; fastest here, least accurate on breeds |
| `vgg` | VGG-16 | 138.4M | Most accurate on breeds; largest and slowest |
| `resnet50` | ResNet-50 | 25.6M | Extra: a deeper ResNet |
| `efficientnet` | EfficientNet-B0 | 5.3M | Extra: the smallest model |

## How `classifier.py` uses them

- **Loading:** only the model you ask for is loaded, the first time it's
  used, and then kept in memory. torchvision 0.13+ loads weights with
  `weights=...DEFAULT`; older versions fall back to `pretrained=True`.
- **Preprocessing:** the standard ImageNet transform: resize to 256, crop to
  224×224, normalise with the ImageNet mean and standard deviation. EXIF
  rotation is applied first.
- **Batching:** images go through the model 16 at a time
  (`predict_batch`), with gradients turned off. Measured on a 4-core CPU,
  batching was 1.1–1.5× faster than one image at a time for most models
  (VGG about 10% slower); the gain is usually larger on a GPU.
- **GPU:** used automatically when available.

## Adding a model

1. Add it to `ARCHITECTURES` in `workspace/classifier.py`:
   `'name': ('torchvision_builder', 'WeightsEnumName')`.
2. Add the same name to `ARCHITECTURES` in `workspace/get_input_args.py`
   (a test checks the two lists match).
3. Add it to the `for arch in ...` loop in both batch scripts.

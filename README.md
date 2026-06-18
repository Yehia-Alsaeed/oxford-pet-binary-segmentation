# Oxford Pet Binary Segmentation

This repository contains a PyTorch deep learning notebook for binary semantic segmentation on the Oxford-IIIT Pet dataset. The task is to classify each pixel as either pet foreground or background using the dataset images and trimap annotations.

The notebook implements, trains, evaluates, and compares three segmentation models:

- **FCN-ResNet18**
- **SegNet-VGG16**
- **HRNet-W18**

The final comparison is based on test-set segmentation metrics, model size, inference speed, and training efficiency.

## Project Highlights

- Builds a complete binary segmentation pipeline for Oxford-IIIT Pet images.
- Converts trimap annotations into foreground/background masks.
- Uses a fixed train/validation/test split for consistent evaluation.
- Applies image resizing, normalization, and training-time augmentation.
- Trains pretrained-backbone segmentation models in PyTorch.
- Evaluates models with mIoU, pet IoU, Dice/F1, pixel accuracy, precision, and recall.
- Compares accuracy, parameter count, inference time, and epoch training time.

## Models

| Model | Backbone | Notes |
| --- | --- | --- |
| FCN-ResNet18 | ResNet18 | Lightweight fully convolutional baseline with skip-style feature fusion. |
| SegNet-VGG16 | VGG16 | Encoder-decoder segmentation model using max-unpooling indices. |
| HRNet-W18 | HRNet-W18 | High-resolution backbone with multi-scale feature fusion through `timm`. |

## Results

Final test metrics from the saved experiment artifacts:

| Model | Test mIoU | Pet IoU | Dice/F1 | Pixel Accuracy | Parameters | Inference / Image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FCN-ResNet18 | 0.9243 | 0.9124 | 0.9542 | 0.9617 | 11.55M | 0.1919s |
| SegNet-VGG16 | 0.9122 | 0.8980 | 0.9462 | 0.9553 | 29.46M | 2.3331s |
| HRNet-W18 | 0.9306 | 0.9198 | 0.9582 | 0.9650 | 11.44M | 0.0633s |

HRNet-W18 achieved the strongest test mIoU and Dice/F1 score in this experiment while also producing the fastest measured inference time per image.

## Visual Overview

### Architecture Comparison

<img src="docs/assets/model-architecture-comparison.svg" alt="Architecture comparison for FCN-ResNet18, SegNet-VGG16, and HRNet-W18" width="100%">

### Qualitative Predictions

The following examples are real prediction grids exported from the experiment notebook. Each grid shows input images, ground-truth binary masks, predicted masks, and prediction overlays.

**FCN-ResNet18**

<img src="docs/assets/fcn-resnet18-predictions.png" alt="FCN-ResNet18 qualitative segmentation predictions" width="100%">

**SegNet-VGG16**

<img src="docs/assets/segnet-vgg16-predictions.png" alt="SegNet-VGG16 qualitative segmentation predictions" width="100%">

**HRNet-W18**

<img src="docs/assets/hrnet-w18-predictions.png" alt="HRNet-W18 qualitative segmentation predictions" width="100%">

## Methodology

The notebook follows this workflow:

1. Resolve the Oxford-IIIT Pet dataset location and load fixed split CSVs.
2. Build a custom PyTorch dataset for image-trimap pairs.
3. Convert trimaps into binary masks.
4. Apply image normalization and training augmentations.
5. Train FCN-ResNet18, SegNet-VGG16, and HRNet-W18.
6. Save the best checkpoints locally during training.
7. Evaluate each model on the test split.
8. Generate qualitative prediction grids and comparison plots.
9. Save the final model comparison table.

## Repository Structure

```text
pet_segmentation_models.ipynb       Main notebook with preprocessing, models, training, evaluation, and comparison
preprocessing_artifacts/            Fixed train/validation/test split CSVs
results_artifacts/model_results.csv Saved final comparison metrics
data/README.md                      Dataset placement instructions
requirements.txt                    Python dependencies
.gitignore                          Excludes datasets, checkpoints, caches, and generated large artifacts
```

Large local artifacts such as the full Oxford-IIIT Pet dataset and `.pt` model checkpoints are intentionally not committed.

## Dataset Setup

Download the Oxford-IIIT Pet dataset and place it in this layout:

```text
data/
  oxford_pet/
    oxford_pet/
      images/
      annotations/
        trimaps/
        trainval.txt
        test.txt
```

The included split CSVs are expected under `preprocessing_artifacts/`:

```text
preprocessing_artifacts/
  train_split.csv
  val_split.csv
  test_split.csv
```

## Setup

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook pet_segmentation_models.ipynb
```

For GPU training, install a PyTorch build that matches your CUDA version from the official PyTorch installation selector.

## Notes

- The notebook was developed for local/GPU execution.
- Training from scratch can take a long time depending on hardware.
- Checkpoints are generated under `checkpoints/` and are ignored by Git.
- Results may vary slightly across hardware, random seeds, and dependency versions.

## Skills Demonstrated

- PyTorch model implementation and training loops
- Semantic segmentation preprocessing and evaluation
- Transfer learning with pretrained computer-vision backbones
- Custom dataset and dataloader design
- Metric-driven model comparison
- Experiment artifact management

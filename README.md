# Prompted Segmentation for Drywall QA

Text-conditioned segmentation: given an image and a prompt, the model outputs a binary mask.

| Prompt | Dataset |
|---|---|
| `segment crack` | Cracks (Roboflow, `fyp-ny1jt/cracks-3ii36`, forked) |
| `segment taping area` | Drywall-Join-Detect (Roboflow, `objectdetect-pu6rn/drywall-join-detect`, version 2) |

Full results, visuals and failure notes are in [REPORT.md](REPORT.md).

## Files

| File | Purpose |
|---|---|
| `drywall_segmentation_v2.ipynb` | Complete pipeline: download, split, train, tune, test, export, figures |
| `predictions/` | Test-set masks (PNG, single-channel, values 0/255, original image size) |
| `results.json` | Seed, hyperparameters, split counts, thresholds, metrics, timings, model size |
| `examples.png`, `failures.png` | Visual examples (original / GT / prediction) |
| `REPORT.md` | Report |

Prediction filenames: `<dataset>_<source image name>__<prompt>.png`, for example
`crack_cracking0161_jpg.rf.6922...__segment_crack.png` and `tape_IMG_8202_JPG_jpg.rf.d2f0...__segment_taping_area.png`.

## Reproduce

1. Open `drywall_segmentation_v2.ipynb` in Google Colab and select **Runtime > Change runtime type > T4 GPU**.
2. Create a free Roboflow account, fork the Cracks dataset into your workspace, and create a version (Versions > Create New Version, no augmentation). The original Cracks page has no downloadable version.
3. In the config cell set `API_KEY` and the workspace/project/version values.
4. Run all cells top to bottom (about 32 minutes of training on a T4).

## Settings

| Setting | Value |
|---|---|
| Model | CLIPSeg, `CIDAS/clipseg-rd64-refined` (150.7M parameters) |
| Fine-tuning | All parameters trainable |
| Loss | BCE + soft Dice |
| Optimizer | AdamW, lr 3e-5, weight decay 1e-4, cosine schedule |
| Epochs / batch size | 20 / 16 (mixed precision) |
| Input size | 352 x 352, masks resized back to the original size for scoring |
| Augmentation | Horizontal and vertical flip, color jitter; prompt phrasing sampled at random during training |
| **Seed** | **42** (Python, NumPy, PyTorch, data loader workers, and the data split) |
| Split | 80/10/10 by source photo (see below) |
| Thresholds | Chosen on the validation set per prompt: crack 0.5, taping 0.25 |

Training prompts per class: crack ("segment crack", "segment wall crack", "segment the crack", "crack"); taping ("segment taping area", "segment joint/tape", "segment drywall seam", "segment drywall joint"). Evaluation always uses the canonical prompt, "segment crack" or "segment taping area".

## Data handling

- **Grouped split.** The Roboflow exports contain several augmented copies of each photo (`name.rf.<hash>.jpg`). All copies of one photo are placed in the same split, so test images are unseen scenes. A random split would leak near-duplicates into the test set.
- **Crack training cap.** Crack training is capped at 2,000 images so that cracks do not dominate the 815 taping training images. Validation and test use all remaining crack images of their groups.
- **Labels.** Crack labels are polygons. Taping labels are bounding boxes, converted to filled rectangles. Taping scores therefore measure agreement with box-shaped labels.
- **Test images** include Roboflow's augmentations (rotations, black cutout squares, grayscale). They were not removed.

## Metrics

Per image: IoU = |P and G| / |P or G|, Dice = 2|P and G| / (|P| + |G|). An image whose prediction and ground truth are both empty scores 1. mIoU and Dice are the means over test images.

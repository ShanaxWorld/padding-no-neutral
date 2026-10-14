# Padding Is Not Neutral: Batch Composition Silently Changes Object Detections

Code, notebooks and per-image results for the paper (under review at the *International Journal of Computer Vision*).

When images of different sizes are batched, detection libraries pad them to a common canvas. We show that this padding changes the confident detections of **25–98% of COCO images** across thirteen public detectors (DETR family, Faster/Mask R-CNN, RetinaNet, FCOS, YOLOS, Grounding DINO, Ultralytics YOLO11/YOLO26). COCO AP moves by at most 0.4 points, so the change is invisible in standard evaluation. The batch-mates' content is irrelevant; only the padding matters.

This repository provides:

- **`padfix`**: `self` (own-canvas masking), a training-free fix that makes every image in a padded batch reproduce its batch-1 output; exact-shape bucketing; and the per-image stability metric.
- **`notebooks/`**: the Kaggle notebooks that produced every number in the paper.
- **`results/`**: the per-image and per-run CSVs behind every table and figure.
- **`figures/make_figures.py`**: regenerates every data figure from `results/`.

## Quick check (a few minutes)

On Kaggle: upload `examples/check_padfix_kaggle.ipynb`, turn on the T4 GPU and Internet, and click *Run All* (it downloads COCO val2017 if it is not attached). Locally:

```bash
pip install torch torchvision numpy pillow
python examples/check_exactness.py --images /path/to/coco/val2017 --n 64
```

Output on a Kaggle T4 (Faster R-CNN, first 64 COCO val images):

```
random batches of 8, default : 39.1% of images change
random batches of 8, + self  : 0.0% of images change
exact-shape buckets           : 0.0% of images change
```

## Using `self` in your code

```python
import torchvision
from padfix import enable_self_torchvision

model = torchvision.models.detection.retinanet_resnet50_fpn_v2(weights="DEFAULT").eval()
fix = enable_self_torchvision(model)      # install once
out = model(list_of_image_tensors)        # each image now gets its batch-1 output
```

For Hugging Face DETR and Conditional DETR, use `enable_self_hf(model)` and call the model with `pixel_values` and `pixel_mask` as usual. `fix.set(())` disables the fix and `fix.set(fix.SELF)` re-enables it.

**Supported:** torchvision Faster R-CNN, Mask R-CNN, Keypoint R-CNN, RetinaNet and FCOS; Hugging Face DETR and Conditional DETR. **Not supported:** window-attention and plain-ViT backbones (Grounding DINO, YOLOS). For those, and for Ultralytics YOLO, use exact-shape buckets (`padfix.exact_shape_buckets`), which need no model changes.

**Porting to other code bases.** Every value computed inside an image's batch-1 canvas must see exactly the inputs it would see at batch size 1. The paper (Sect. 6.2) turns this into five rules: canvas, neighbourhood operators, statistics, canvas-derived quantities, and outputs.

## Where each result comes from

| Paper | Notebook | Results |
|---|---|---|
| Table 1, Figs. 2, 4, 5 (twelve models, full COCO val) | `final_fullval_group1*.ipynb`, `final_fullval_group2.ipynb` | `summary_group*.csv`, `per_image_group*.csv`, `fragility_stats.csv` |
| Table 3 and Fig. 12 (remedies, throughput) | `final_fullval_group1*.ipynb` | `summary_group1.csv`, `per_image_group1.csv` |
| Fig. 11 (YOLOS box rescale) | `final_fullval_group2.ipynb` (first tested in `batchdep_exp5.ipynb`) | `summary_group2.csv` |
| Figs. 1 and 3 (qualitative) | `qualitative_figures.ipynb` | — |
| Content irrelevance (Sect. 5.1), Fig. 9 (locality), Fig. 13 (batching strategies) | `batchdep_exp1.ipynb` | `B_per_detection.csv`, `A_summary.csv` |
| Fig. 8 (switch), Fig. 10 (four leaks), nominal-stride fix (Sect. 6.1) | `batchdep_exp2.ipynb` | `A_sweep.csv`, `exp_extracted.csv` |
| Design of `self` (500-image study) | `batchdep_exp3.ipynb` | — |
| Newer models (500 images), instance masks, fp16 | `batchdep_exp4.ipynb` | — |
| Figs. 14 and 15 (auto-labelling, serving) | `batchdep_exp5.ipynb` | `exp_extracted.csv` |
| Table 2 (Pascal VOC, Ultralytics YOLO) | `exp6_strengthening.ipynb` | `exp6_*.csv` |
| Table D1 (Ultralytics validator) | `exp6c_official_validator.ipynb` | `exp6_C_val.csv` |
| Fig. 6 (batch size), pose, CPU vs. GPU | `exp7_reviewer_answers.ipynb` | `exp7_*.csv` |
| First sanity checks (determinism, 500 images) | `batchdep_day0.ipynb` | — |

`exp_extracted.csv` holds numbers transcribed from notebook outputs whose CSVs were overwritten by later runs; each row names its source notebook.

## Reproducing

All experiments ran on a free Kaggle notebook with one NVIDIA T4 GPU (Settings → Accelerator → GPU T4, then *Save & Run All*), with the COCO 2017 dataset attached. Pinned versions: `transformers==5.17.0`, `ultralytics==8.4.172`, and torch/torchvision as preinstalled (2.10/0.25; 2.11/0.26 for Exp 7). Each notebook writes its CSVs and a `manifest.json` with the environment and a claims check.

## Licence

Code: MIT. COCO, Pascal VOC and the pretrained weights are subject to their own licences.

## Citation

```bibtex
@article{hossain2026padding,
  title   = {Padding Is Not Neutral: Batch Composition Silently Changes Object Detections},
  author  = {Hossain, S. M. Shah Nawaz},
  journal = {Under review},
  year    = {2026}
}
```

# NeuroDxFM

NeuroDxFM is a multimodal 3D brain-imaging foundation model for presymptomatic neurodegenerative disease detection across acquisition protocols and clinical populations. The implementation combines protocol-counterfactual contrastive learning, elapsed-time-conditioned future representation prediction, modality-dropout reconstruction, anatomical regional grounding and amyloid anchoring in a shared volumetric encoder.

## Environment

The fixed environment uses Python 3.11, PyTorch 2.1.2 and CUDA 12.1.

```bash
conda env create -f environment.yml
conda activate neurodxfm
python -m pip install -e .
```

The container path is:

```bash
docker build -t neurodxfm .
```

## Data

Official access locations that responded during release validation are collected in `dataset_links.txt`. UK Biobank, ADNI, AIBL, NACC, PPMI and DIAN require an application or data-use agreement. HCP-Aging uses ConnectomeDB terms. OpenNeuro licenses are dataset-specific. No restricted imaging data are distributed here.

Create `data/manifest.csv` with one row per visit. Required columns are `subject_id`, `visit_id`, `cohort`, `t1`, `diagnosis`, `protocol`, `months`, `apoe4`, `site`, `vendor`, `field_strength` and `sequence`. Optional columns are `flair`, `fdg_pet`, `amyloid_pet`, `csf`, `amyloid` and a JSON array named `anatomy` containing 68 regional volumes. All partitions are subject-level and stratified by diagnosis and APOE ε4 status.

Volumes are expected as float32 `.npy` or tensor `.pt` files after skull stripping, affine registration to MNI152 at 1 mm isotropic resolution, bias-field correction and white-matter peak normalization to 110. Inputs are center cropped or padded to 160 × 192 × 160.

## Pretraining

The primary run uses eight NVIDIA A100 80GB accelerators, FP16, 200 epochs and an effective batch of 256. Per-device batch is 8 with four gradient-accumulation steps. The first 100 epochs train protocol invariance, modality reconstruction, anatomical grounding and amyloid anchoring. The trajectory head joins at epoch 101. Peak memory is approximately 38GB per device. The reported pretraining budget is 1,200 GPU-hours; the complete experimental matrix, including fine-tuning, evaluation and parameter search, totals 2,048 GPU-hours.

```bash
bash scripts/train.sh
```

The primary configuration follows the Methods training paragraph: AdamW learning rate 3e-4, weight decay 0.05, betas 0.9 and 0.999, cosine decay and ten warm-up epochs. `configs/sa4_selected.yaml` records the distinct 2e-4 learning rate and 1e-4 weight decay printed in Supplementary Table SA-4. This discrepancy is kept explicit.

## Experiments

Objective-removal and architecture runs are represented by files under `configs/`. The backbone alternatives are DINOv2-3D, Vision Mamba-3D and ResNet-3D. The evaluation package provides bootstrap confidence intervals, DeLong comparison, Holm correction, subgroup reporting, prevalence-adjusted predictive values, calibration error, concordance index and site-discriminator analysis.

The principal acceptance targets are ADNI MCI conversion AUC 0.887, T1-only cross-site AUCs of 0.941 on ADNI, 0.924 on OASIS-3, 0.908 on AIBL and 0.931 on NACC, amyloid-positive cognitively-normal sensitivity 0.728 at specificity 0.80, PPMI prodromal AUC 0.812 and DIAN lead time 27.4 months. Seed-level results use 20 seeds and are compared with the tolerances encoded in `neurodxfm.reporting`.

## Validation

```bash
pytest -q
ruff check .
mypy --strict code/neurodxfm
```

The suite covers tensor shapes, losses, metrics, preprocessing, subject partition integrity, modality fusion, architecture variants, configuration parsing, gradients and integration behavior.

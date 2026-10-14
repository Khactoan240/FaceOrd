# FaceOrd: Landmark-guided ordinal multi-task learning for facial age and gender prediction

FaceOrd predicts one of eight ordered age groups and binary gender from a face image. It combines three components:

- **Landmark-guided global–local representation.** A ConvNeXt-S global branch encodes the aligned face. Six regions defined by 68 facial landmarks (left/right eye–eyebrow, nose, mouth, left/right cheek–jaw) pass through a shared regional encoder and a lightweight Transformer. A learnable, sample-dependent gate fuses the global and local features.
- **Ordinal age head.** Seven cumulative thresholds model P(y > k). The age loss has three terms: an ordinal binary cross-entropy, a monotonicity penalty and a Gaussian soft label-distribution loss. The reconstructed group masses are clamped and renormalised into a valid distribution.
- **Uncertainty-aware multi-task loss.** Learnable homoscedastic uncertainties balance the age and gender losses.

| Group | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| Age (years) | 0–2 | 3–6 | 7–13 | 14–21 | 22–33 | 34–44 | 45–53 | ≥54 |

Gender labels: 0 = male, 1 = female.

## Repository structure

```text
faceord/                    FaceOrd model, data pipeline, training and evaluation
  config.py                 all hyper-parameters (defaults = final configuration)
  age_mapping.py            exact-age and Adience category -> age group, gender labels
  landmarks.py              2D-FAN (default) or dlib landmarks, landmark cache, perturbations
  preprocessing.py          resize, CLAHE + gamma, affine alignment, region crops, augmentation
  dataset.py                UTKFace dataset
  splits.py                 stratified outer folds with inner validation, stored as JSON
  models.py                 FaceOrd and its ablation switches
  losses.py                 ordinal, monotonicity, label-distribution and uncertainty losses
  data_cleaner.py           per-epoch DataCleaner resampling
  experiment.py             model/loss factories, ablation variants (a)-(f)
  engine.py                 training loop, checkpoint selection, cross-validation
  calibration.py            temperature scaling on the validation subset (calibrated ECE)
  checkpoint.py             checkpoint save/load
  metrics.py                Acc, F1, AUC, mAP, Brier, ECE, ordinal violation diagnostics
  logger.py                 console log files in logs/
  kfold.py                  five-fold comparison (FaceOrd and comparison models)
  ablation.py               cumulative ablation
  tau_sweep.py              tau sensitivity (nested inner folds)
  loss_sweep.py             lambda_mono / lambda_dist sensitivity (nested inner folds)
  weighting_tune.py         fixed task weights tuned on validation data
  eval_from_predictions.py  metrics from saved out-of-fold predictions
  landmark_robustness.py    landmark jitter / dropout evaluation
  eval_cross_dataset.py     direct transfer to MORPH-II, Adience and FairFace
  flops.py                  parameters, FLOPs and model-only latency
  benchmark_pipeline.py     end-to-end latency including landmark estimation
  visualize.py              Grad-CAM, t-SNE, attention maps, failure grid
  infer.py                  prediction for a single image
baselines/
  models.py                 VGG19, ResNet50, DenseNet169, OptimizedVGG19, Swin-T,
                            ConvNeXt-S/T, ViT-B/16 face classifier
  official_adapters.py      adapters for the official MiVOLO and FaceXFormer code
analysis/
  compare_results.py        paired t-tests, 95% CI, Cohen's d_z, Holm correction
  subgroups.py              gender / race / age-group results, confusion matrices
  ordinal_diagnostics.py    monotonicity-violation rates of the cumulative outputs
  plot_training_dynamics.py sigma trajectories and tau sensitivity figure
  failure_cases/            selection and figure of representative failure cases
configs/                    one JSON configuration per experiment
scripts/
  prepare_external_annotations.py   MORPH-II / Adience / FairFace annotation files
tests/                      unit tests
run_experiment.py           prints (or runs) the commands of an experiment group
```

## Installation

The experiments used Python 3.10, PyTorch 2.5.1 and CUDA 12.1 on an NVIDIA RTX 3090.

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-paper.txt
```

| File | Contents |
|---|---|
| `requirements-paper.txt` | exact training environment (PyTorch 2.5.1, face-alignment 1.4.1) |
| `requirements.txt` + `requirements-landmarks.txt` | same packages with looser version ranges (e.g. CPU or macOS) |
| `requirements-eval.txt` | metrics and statistics from saved predictions only (no PyTorch) |
| `requirements-official-baselines.txt` | extra packages for the official MiVOLO code |
| `requirements-dlib.txt` | optional dlib landmark backend |
| `requirements-dev.txt` | pytest |

On first use, `face-alignment` downloads the pretrained 2D-FAN and SFD face-detector weights, and torchvision downloads the ImageNet backbones.

## Data

The datasets are not redistributed. Obtain them from their providers and follow their licenses.

- **UTKFace** (aligned & cropped faces). Place the images in `data/UTKFace/`, or pass `--data`. The original file names (`age_gender_race_date.jpg`) provide the labels. Images are sorted by file name, and this order defines `sample_idx` in every split and prediction file.
- **MORPH-II** and **Adience** (and optionally **FairFace**) are used only for cross-dataset transfer (see below).

### Landmark cache

The 68 landmarks are estimated once with the fixed pretrained 2D-FAN (face-alignment 1.4.1) and reused by every experiment:

```bash
python -m faceord.landmarks --data data/UTKFace --out data/landmarks_utkface.npz --device cuda
```

Landmarks are computed on the resized 224×224 CLAHE/gamma image. The largest face found by the SFD detector is used. If no face is detected, 2D-FAN runs with the whole image as the face box. The cache records which case applied to each image, and the backend and gamma it was built with. During training, the flip and rotation augmentations are applied to the cached landmarks as well, so the global image and the regional patches stay consistent. Without a cache, landmarks are estimated on the fly, which is much slower.

**dlib backend (optional).** Landmarks can also come from the dlib HOG face detector and its 68-point shape predictor. Install `requirements-dlib.txt` and download [`shape_predictor_68_face_landmarks.dat`](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) to `data/` (`DLIB_PREDICTOR`). Then build a separate cache and set `"LANDMARK_BACKEND": "dlib"` and `LANDMARK_CACHE` in the configuration:

```bash
python -m faceord.landmarks --backend dlib --data data/UTKFace --out data/landmarks_utkface_dlib.npz
```

## Usage

Run all commands from the repository root. Each script lists its options with `--help`.

### Training protocol

- Five outer folds, stratified by age group (split seed 42) and shared by all methods. Use one split file for every run, e.g. `--splits results/shared_splits.json`.
- Folds 1–5 train with seeds 3407–3411, the same for every method. The seed drives initialisation, shuffling and augmentation. Deterministic PyTorch/CUDA execution is enabled where supported.
- 10% of each outer training portion is a stratified validation subset. The checkpoint with the lowest validation age + gender negative log-likelihood is kept. The same subset fits a temperature for the age head, which gives the calibrated ECE (`ece_calibrated`) of the test fold; all other metrics use the uncalibrated probabilities. The outer test fold is used only for the final evaluation.
- 30 epochs, batch size 32, gradient clipping at 5.0, mixed precision on CUDA.
- FaceOrd uses AdamW (learning rate 3e-4, weight decay 1e-4) with a 3-epoch linear warm-up and cosine decay to 1e-6. The first two ConvNeXt-S stages are frozen during warm-up. Further settings: τ = 0.5, λ_mono = 0.1, λ_dist = 0.5, square-root age-group weighting.
- DataCleaner runs from epoch 4 on the training subset only. Easy samples (both tasks correct, confidence ≥ 0.90) are replaced every epoch: the 20% most confident of them are swapped for resampled near-miss samples (age error ≤ 1 group). The training-set size stays the same.
- Comparison models: VGG19, ResNet50 and DenseNet169 use SGD (momentum 0.9, learning rate 1e-4) with cosine decay. OptimizedVGG19 uses AdamW (3e-4). MiVOLO-Face, FaceXFormer, Swin-T and ConvNeXt-S use AdamW (3e-4) with warm-up.

### Main five-fold comparison

```bash
python -m faceord.kfold --config configs/faceord.json --splits results/shared_splits.json
python -m faceord.kfold --model resnet50 --config configs/resnet50.json --splits results/shared_splits.json
```

`python run_experiment.py --experiment baselines` prints the commands for all comparison models. `--execute` runs them. The models are `vgg19`, `resnet50`, `densenet169`, `optimized_vgg19` (with and without DataCleaner), `mivolo_face`, `facexformer`, `swin_t` and `convnext_small`. `convnext_tiny` and `vit_face_approx` are also available.

Every run writes to its `SAVE_DIR` (e.g. `results/faceord/`):

| File | Contents |
|---|---|
| `metrics.json` | per-fold and mean ± std metrics (including `ece_calibrated` and the fitted temperature), configuration, split fingerprint |
| `oof_predictions.csv` / `.xlsx` | every out-of-fold prediction: group probabilities `age_prob_k`, raw cumulative outputs `age_c_k` = P(y > k), gender probability |
| `fold_<i>/best.pt` | selected checkpoint with configuration and train/val/test indices |
| `fold_<i>/history.json` | per-epoch loss, validation metrics, σ_age and σ_gender |
| `fold_<i>/cleaner/epoch_<n>.json` | DataCleaner indices and counts per epoch |
| `run_metadata.json` | configuration and software/hardware versions |

**MiVOLO-Face and FaceXFormer** use the official implementations. Clone them first (`pip install -r requirements-official-baselines.txt`):

```bash
git clone https://github.com/WildChlamydia/MiVOLO external/MiVOLO
git clone https://github.com/Kartik-3004/facexformer external/facexformer
```

Alternatively, set `FACEORD_MIVOLO_SOURCE` / `FACEORD_FACEXFORMER_SOURCE`. MiVOLO is used face-only (`MIVOLO_VARIANT`, default `mivolo_d3_224`). Both models are trained with new 8-way age and 2-way gender outputs on the same folds and seeds.

### Ablation and sensitivity analysis

```bash
# cumulative ablation (a)-(f): flat -> ordinal -> label distribution -> regions -> attention + gate -> uncertainty
python -m faceord.ablation --splits results/shared_splits.json --output results/ablation

# tau, lambda_mono, lambda_dist: three inner folds (30 epochs) inside each outer training portion
python -m faceord.tau_sweep --outer-splits results/shared_splits.json --output results/tau_sweep
python -m faceord.loss_sweep --parameter LAMBDA_MONO --values 0 0.05 0.1 0.2 --output results/lambda_mono
python -m faceord.loss_sweep --parameter LAMBDA_DIST --values 0 0.25 0.5 1.0 --output results/lambda_dist

# fixed task weights tuned on the validation subsets (comparison with uncertainty weighting)
python -m faceord.weighting_tune --age-weights 0.5 1 2 --output results/weighting_tune
```

The per-fold selections can be used for the final training with `faceord.kfold --tau-selection results/tau_sweep/selected_tau_by_outer_fold.json` or `--hyperparameter-selection <dir>/selected_settings.json`.

### Evaluation and statistics

```bash
python -m faceord.eval_from_predictions --predictions results/faceord/oof_predictions.csv --splits results/shared_splits.json
python -m analysis.compare_results --reference results/faceord/metrics.json \
    --others results/resnet50/metrics.json results/facexformer/metrics.json
python -m analysis.subgroups --predictions results/faceord/oof_predictions.csv --splits results/shared_splits.json
python -m analysis.ordinal_diagnostics --predictions results/faceord/oof_predictions.csv --splits results/shared_splits.json
python -m analysis.plot_training_dynamics --run-dir results/faceord --tau-summary results/tau_sweep/tau_sweep_summary.json
```

`compare_results` reports paired t-tests on the fold AUCs with 95% confidence intervals and Cohen's d_z. The Holm correction is applied separately to the age and gender comparisons. Standard deviations across folds use ddof = 1.

### Landmark robustness

```bash
python -m faceord.landmark_robustness --checkpoints results/faceord/fold_*/best.pt \
    --splits results/shared_splits.json --data data/UTKFace --jitters 0.01 0.03 0.05 --dropouts 0.1 0.2 0.3
```

Each fold checkpoint is evaluated on its own test fold. The perturbations act on the aligned landmarks before the regions are cropped. Jitter is Gaussian noise with std = level × D, where D is the landmark bounding-box diagonal (`--scale interocular` uses the eye distance instead). Dropout removes ⌊68 × level⌋ points.

### Cross-dataset transfer

UTKFace checkpoints are evaluated directly on MORPH-II and Adience, without fine-tuning or calibration. The eight native Adience categories map one-to-one onto the eight groups. MORPH-II exact ages use the age bins above. FairFace is supported as an additional target; its age ranges are mapped approximately to the groups (`faceord/age_mapping.py`).

```bash
# annotations: from the original Adience fold files, or from a file,age,gender CSV (MORPH-II, FairFace)
python scripts/prepare_external_annotations.py --dataset adience \
    --adience-folds /data/adience/fold_*_data.txt --images /data/adience --output /data/adience/annotations.csv
python scripts/prepare_external_annotations.py --dataset morph \
    --input /data/morph/labels.csv --output /data/morph/annotations.csv
python scripts/prepare_external_annotations.py --dataset fairface \
    --input /data/fairface/fairface_label_val.csv --output /data/fairface/annotations.csv

# landmark cache of the target images (optional, recommended)
python -m faceord.landmarks --data /data/adience --recursive --out /data/adience/landmarks.npz --device cuda

python -m faceord.eval_cross_dataset --dataset adience --checkpoints results/faceord/fold_*/best.pt \
    --annotations /data/adience/annotations.csv --images /data/adience --landmarks /data/adience/landmarks.npz
```

Each checkpoint is evaluated separately, and the summary reports the mean ± std over checkpoints. Use either the outer-fold checkpoints or separately trained source-domain models (`python run_experiment.py --experiment final`, one model per seed 3407–3411).

### Computational cost

```bash
python -m faceord.flops --config configs/faceord.json --device cuda          # parameters, FLOPs, model-only latency
python -m faceord.benchmark_pipeline --checkpoint results/faceord/fold_1/best.pt \
    --images data/UTKFace/<image>.jpg --device cuda --landmark-device cuda      # end-to-end latency
```

`flops` excludes landmark detection and preprocessing. `benchmark_pipeline` times every stage separately: decoding, CLAHE/gamma, 2D-FAN, alignment, region extraction, inference and ordinal post-processing.

### Visualisation and failure cases

```bash
python -m faceord.visualize --ckpt results/faceord/fold_1/best.pt \
    --compare-ckpt results/vgg19/fold_1/best.pt --compare-name VGG19 --mode all

python -m analysis.failure_cases.select_failure_cases --pred results/faceord/oof_predictions.xlsx
# inspect results/failure_cases/candidates/*.png, then copy
# analysis/failure_cases/selected_cases.example.json to selected_cases.json and fill in six cases
python -m analysis.failure_cases.make_failure_figure --pred results/faceord/oof_predictions.xlsx
```

`visualize` produces the Grad-CAM (FaceOrd next to a comparison model), t-SNE, attention and failure-grid figures from the checkpoint's test fold. For predictions of a dlib-based run, pass `--landmark-backend dlib --landmark-cache <dlib cache>` to both failure-case scripts.

### Single-image inference

```bash
python -m faceord.infer --checkpoint results/faceord/fold_1/best.pt --image face.jpg
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Citation

If you use this code, please cite the FaceOrd paper (see `CITATION.cff`).

## License

The source code is released under the MIT License (see `LICENSE`). UTKFace, MORPH-II, Adience, FairFace, the pretrained 2D-FAN and dlib models and the MiVOLO / FaceXFormer code are subject to their own licenses and terms of use.

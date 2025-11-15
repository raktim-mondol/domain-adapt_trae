# BMA MIL Classifier

End-to-end Multiple Instance Learning (MIL) for BMA classification with options for bag-level and pile-level training, configurable pooling strategies, and domain adaptation variants (Supervised DA, Unsupervised DA, Semi-Supervised DA). Implements patch-based processing using a ViT-R50 feature extractor and attention aggregation.

## Features
- Patch extraction from 4032×3024 images into 12 patches per image
- ViT-R50 feature extractor via `timm` with selective layer unfreezing
- Attention-based MIL aggregation and classifier head
- Bag-level training with pile-level evaluation; optional pile-level training
- Pooling methods for pile aggregation: mean, max, attention, majority voting
- Domain adaptation options: DA (adversarial + MMD + orthogonality), UDA (entropy + consistency + EMA teacher), SSDA (labeled + unlabeled target)
- Stratified standard split and robust k-fold splitting for small-class scenarios
- Early stopping, LR scheduler, weighted loss, logging, checkpointing, plots

## Repository Structure
```
configs/
  config.py               # Global configuration and hyperparameters
data/
  BWM_label_data.csv      # Example source domain labels
  CVM_label_data.csv      # Example target domain labels
  images/sample_image.JPG # Example image
scripts/
  train.py                # Main training CLI
  evaluate_kfold.py       # K-fold evaluation runner
  show_fold_splits.py     # Visualize/print fold splits
  test_da_dummy.py        # DA smoke test
  test_uda_dummy.py       # UDA smoke test
  test_ssda_dummy.py      # SSDA smoke test
  test_single_pile.py     # Single-pile debug/training
src/
  __init__.py
  augmentation.py         # Histogram, geometric, color, noise aug pipeline
  feature_extractor.py    # ViT-R50 feature extractor via timm
  data/
    dataset.py            # Bag-level dataset and helper
    pile_dataset.py       # Pile-level dataset and collate
    patch_extractor.py    # 12-patch extraction per image
  models/
    bma_mil_model.py      # Attention aggregator + classifier
    domain_discriminator.py# Gradient reversal + domain discriminator
  losses/
    mmd.py, entropy.py, consistency.py, orthogonal.py
  utils/
    training.py           # Training loops (bag + DA/UDA/SSDA)
    pile_training.py      # Pile-level training/validation
    pooling.py            # Aggregation methods + AttentionPooling
    evaluation.py         # Evaluation utilities
    early_stopping.py     # Early stopping utility
    ema.py                # EMA teacher updates
    logging_utils.py      # Logging setup
test/
  ...                     # Unit, integration, GPU tests and batch runner
requirements.txt          # Python dependencies
setup.py                  # Package metadata and console script entry
.gitignore
```

## Installation
- Python >= 3.8
- Install dependencies:
```
pip install -r requirements.txt
```
- Optional: install the package for console script usage:
```
pip install -e .
```
This provides the `bma-train` entry point.

## Data Format
Input CSVs must include the following columns:
- `pile`: pile identifier
- `image_path`: filename of the image (relative to `IMAGE_DIR`)
- `BMA_label`: integer class label (1-indexed in data; converted to 0-indexed internally)

Place all referenced images in a single folder and set `IMAGE_DIR` accordingly.

## Configuration
Edit `configs/config.py` to set paths, hyperparameters and modes. Key fields:
- Paths: `DATA_PATH`, `IMAGE_DIR`, `SOURCE_*`, `TARGET_*`
- Model: `FEATURE_DIM`, `IMAGE_HIDDEN_DIM`, `FEATURE_EXTRACTOR_MODEL`, `TRAINABLE_FEATURE_LAYERS`
- Training: `BATCH_SIZE`, `NUM_EPOCHS`, `LEARNING_RATE`, `WEIGHT_DECAY`, `DROPOUT_RATE`
- Scheduler: `USE_LR_SCHEDULER`, `LR_SCHEDULER_TYPE` (reduce_on_plateau), related parameters
- Level: `TRAINING_LEVEL` in `{bag, pile}`; `POOLING_METHOD` for pile-level training
- Class imbalance: `USE_WEIGHTED_LOSS`
- Splits: `SPLIT_MODE` in `{standard, kfold}`, ratios and `NUM_FOLDS`
- Device: `DEVICE` auto-detects CUDA if available
- Logging/outputs: `BEST_MODEL_PATH`, `TRAINING_PLOT_PATH`, `LOG_DIR`
- Augmentation: histogram method; enable/disable geometric/color/noise; counts via `INCLUDE_ORIGINAL_AND_AUGMENTED` and `NUM_AUGMENTATION_VERSIONS`
- Domain Adaptation knobs (when enabled): `DA_ENABLED`, `UDA_ENABLED`, `SSDA_ENABLED`, loss weights and warmups

Windows paths are supported; use raw strings like `r'C:\Users\You\Images'`.

## Training Modes
- Bag-level training (default): trains per image (bag), validates on piles by aggregating bag predictions.
- Pile-level training: trains per pile, processing all bags from a pile in each sample.

Aggregation methods for pile-level evaluation/training are implemented in `src/utils/pooling.py`.

## Domain Adaptation Variants
- Supervised DA (`DA_ENABLED`): adversarial domain discriminator with gradient reversal, MMD alignment, orthogonality constraint.
- Unsupervised DA (`UDA_ENABLED`): adds entropy minimization, consistency with EMA teacher, pseudo-labeling with thresholding.
- Semi-Supervised DA (`SSDA_ENABLED`): mixes labeled and unlabeled target data; combines supervised loss with entropy/consistency and pseudo-loss.

Control via CLI `--da_mode` or by setting flags in `configs/config.py`.

## Quickstart
Run the standard training script:
```
python scripts/train.py --level bag
```

Enable domain adaptation via command-line:
```
# Supervised DA (source + target labeled)
python scripts/train.py --da_mode supervised --level bag \
  --source_csv data/BWM_label_data.csv --source_images D:/SCANDY/Data/BWM_Data \
  --target_csv data/CVM_label_data.csv --target_images D:/SCANDY/Data/CVM_Data

# Unsupervised DA (source labeled, target unlabeled)
python scripts/train.py --da_mode unsupervised --level bag \
  --source_csv data/BWM_label_data.csv --source_images D:/SCANDY/Data/BWM_Data \
  --target_csv data/CVM_label_data.csv --target_images D:/SCANDY/Data/CVM_Data

# Semi-Supervised DA (target labeled + unlabeled CSVs)
python scripts/train.py --da_mode semi --level bag \
  --source_csv data/BWM_label_data.csv --source_images D:/SCANDY/Data/BWM_Data \
  --target_csv_labeled data/CVM_label_data.csv --target_csv_unlabeled data/CVM_label_data.csv \
  --target_images D:/SCANDY/Data/CVM_Data
```

Switch to pile-level training:
```
python scripts/train.py --level pile
```

Standard vs k-fold splitting is controlled in `configs/config.py` via `SPLIT_MODE`.

## Cross-Validation
Use k-fold cross-validation runner:
```
python scripts/evaluate_kfold.py
```
Fold creation is robust to small-class scenarios and prints per-fold class distributions and results.

## Outputs
- Checkpoints: `models/best_bma_mil_model*.pth`
- Training plot: `results/training_history.png` (or per-fold variants)
- Logs: under `logs/`

## Testing
Run smoke tests for modes:
```
python scripts/test_da_dummy.py
python scripts/test_uda_dummy.py
python scripts/test_ssda_dummy.py
python scripts/test_single_pile.py
```

Run unit/integration tests:
```
pytest
```
GPU checks (Windows): `test/run_gpu_test.bat`

## Notes
- Labels in CSV are expected to be 1-indexed; the code converts to 0-indexed internally.
- Images must be 4032×3024 or will be resized; 12 patches are extracted per image and resized to 224×224.
- When the feature extractor is fully frozen, feature computation runs under `no_grad` for efficiency.

## License
MIT License. See metadata in `setup.py`.
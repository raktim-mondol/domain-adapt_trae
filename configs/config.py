"""
Configuration file for BMA MIL Classifier
"""

import torch


class Config:
    # Data parameters
    DATA_PATH = 'data/CVM_label_data.csv'
    # IMAGE_DIR can be:
    # - Relative path: 'data/images'
    # - Absolute Windows path: r'C:\Users\YourName\Pictures\pile_images'
    # - Absolute Windows path: 'C:/Users/YourName/Pictures/pile_images'
    # All images referenced in the CSV should be in this single folder
    IMAGE_DIR = r'D:\SCANDY\Data\CVM_Data'
    SOURCE_DATA_PATH = 'data/BWM_label_data.csv'
    SOURCE_IMAGE_DIR = r'D:\SCANDY\Data\BWM_Data'
    TARGET_DATA_PATH = 'data/CVM_label_data.csv'
    TARGET_IMAGE_DIR = r'D:\SCANDY\Data\CVM_Data'
    DA_MODE = 'none'
    NUM_CLASSES = 3
    QLD1_DATA_PATH = DATA_PATH
    QLD1_IMAGE_DIR = IMAGE_DIR
    QLD2_DATA_PATH = DATA_PATH
    QLD2_IMAGE_DIR = IMAGE_DIR

    # Image processing
    ORIGINAL_SIZE = (4032, 3024)
    PATCH_SIZE = 1008
    TARGET_SIZE = 224
    NUM_PATCHES_PER_IMAGE = 12
    MAX_IMAGES_PER_PILE = 100000

    # Model architecture
    FEATURE_DIM = 768  # ViT-R50 feature dimension
    IMAGE_HIDDEN_DIM = 512
    PILE_HIDDEN_DIM = 256

    # Training parameters
    BATCH_SIZE = 6
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    DROPOUT_RATE = 0.3
    
    # Optimizer and Scheduler
    USE_ADAMW = True  # Use AdamW optimizer (improved Adam with better weight decay)
    USE_LR_SCHEDULER = True  # Enable learning rate scheduler
    LR_SCHEDULER_TYPE = 'reduce_on_plateau'  # 'reduce_on_plateau' or 'cosine_annealing'
    
    # ReduceLROnPlateau parameters
    LR_SCHEDULER_MODE = 'min'  # 'min' for loss, 'max' for accuracy
    LR_SCHEDULER_FACTOR = 0.5  # Reduce LR by this factor
    LR_SCHEDULER_PATIENCE = 5  # Number of epochs with no improvement before reducing LR
    LR_SCHEDULER_MIN_LR = 1e-7  # Minimum learning rate
    LR_SCHEDULER_THRESHOLD = 1e-4  # Threshold for measuring improvement
    
    # Training level
    TRAINING_LEVEL = 'bag'  # 'bag' or 'pile'
    # 'bag': Train on individual images (bags), validate on piles (aggregated)
    # 'pile': Train on entire piles, validate on piles (both pile-level)
    
    # Pile-level aggregation method (only used when TRAINING_LEVEL = 'pile')
    POOLING_METHOD = 'mean'  # 'mean', 'max', 'attention', or 'majority'
    # NOTE: This setting is IGNORED for bag-level training
    # For bag-level training: all 3 methods (mean, max, majority) are automatically evaluated during testing
    # For pile-level training: this determines which method is used during training and validation
    
    # Class imbalance handling
    USE_WEIGHTED_LOSS = True

    DA_ENABLED = False
    LAMBDA_ADV = 1.0
    LAMBDA_MMD = 0.5
    LAMBDA_ORTH = 0.01
    GRL_COEFF = 1.0
    MMD_BANDWIDTHS = [0.5, 1.0, 2.0, 4.0]
    DA_RAMP_EPOCHS = 5
    UDA_ENABLED = False
    LAMBDA_ENT = 0.1
    LAMBDA_CONS = 0.5
    PSEUDO_THRESHOLD = 0.9
    UDA_WARMUP_EPOCHS = 5
    USE_EMA_TEACHER = True
    EMA_DECAY = 0.99
    SSDA_ENABLED = False
    QLD2_LABELED_DATA_PATH = DATA_PATH
    QLD2_UNLABELED_DATA_PATH = DATA_PATH
    SSDA_WARMUP_EPOCHS = 5

    # Data split (pile-level)
    SPLIT_MODE = 'kfold'  # 'standard' or 'kfold'
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.1
    TEST_RATIO = 0.2
    RANDOM_STATE = 42
    
    # Cross-validation (pile-level)
    NUM_FOLDS = 3  # Only used when SPLIT_MODE = 'kfold'

    # Feature extractor (integrated into model)
    FEATURE_EXTRACTOR_MODEL = 'vit_base_r50_s16_224.orig_in21k'
    
    # Feature extractor training mode:
    # TRAINABLE_FEATURE_LAYERS:
    #   - 0: Fully frozen (no gradients)
    #   - -1: Fully trainable (all layers)
    #   - N (1-12 for ViT): Last N blocks/layers trainable, rest frozen
    TRAINABLE_FEATURE_LAYERS = 2  # -1=all trainable, 0=frozen, N=last N layers trainable

    # Paths
    BEST_MODEL_PATH = 'models/best_bma_mil_model.pth'
    TRAINING_PLOT_PATH = 'results/training_history.png'
    LOG_DIR = 'logs'

    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    #DEVICE = 'cpu'

    # Checkpointing / Resume
    RESUME_TRAINING = True
    CHECKPOINT_PATH = BEST_MODEL_PATH
    # Early stopping
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10
    EARLY_STOPPING_MIN_DELTA = 0.001
    
    # Evaluation
    EVAL_ON_PILE_LEVEL = True  # Aggregate bag predictions to pile level during validation
    
    # Logging
    ENABLE_LOGGING = True
    LOG_LEVEL = 'INFO'
    
    # Data Augmentation Parameters
    HISTOGRAM_METHOD = 'clahe'
    
    # Augmentation strategy
    INCLUDE_ORIGINAL_AND_AUGMENTED = True  # If True: includes original patches
                                            # If False: only augmented patches
    NUM_AUGMENTATION_VERSIONS = 3           # Number of augmented versions per patch
                                            # 1: 12 patches (standard)
                                            # 2: 24 patches (12 original + 12 augmented) if INCLUDE_ORIGINAL_AND_AUGMENTED=True
                                            # 3: 36 patches (12 original + 36 augmented) if INCLUDE_ORIGINAL_AND_AUGMENTED=True
                                            #    OR 36 patches (only augmented) if INCLUDE_ORIGINAL_AND_AUGMENTED=False
    
    # Enable/disable augmentation types
    ENABLE_GEOMETRIC_AUG = True
    ENABLE_COLOR_AUG = False
    ENABLE_NOISE_AUG = False
    
    # Geometric augmentation parameters
    ROTATION_RANGE = 15
    ZOOM_RANGE = (0.9, 2.5)
    #SHEAR_RANGE = 10
    HORIZONTAL_FLIP = True
    #VERTICAL_FLIP = True
    #GEOMETRIC_PROB = 0.5
    
    # Color augmentation parameters
    BRIGHTNESS_RANGE = (0.8, 1.2)
    CONTRAST_RANGE = (0.8, 1.2)
    SATURATION_RANGE = (0.8, 1.2)
    HUE_RANGE = (-0.1, 0.1)
    COLOR_PROB = 0.5
    
    # Noise and blur parameters
    NOISE_STD = 0.01
    BLUR_SIGMA = (0.1, 2.0)
    NOISE_PROB = 0.3

"""
Quick test script to evaluate a single pile with saved model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from configs.config import Config
from src.models.bma_mil_model import BMA_MIL_Classifier
from src.feature_extractor import FeatureExtractor
from src.data.dataset import BMADataset
from src.utils.pooling import aggregate_all_methods, AttentionPooling
from src.augmentation import ComposedAugmentation
from tqdm import tqdm


def test_single_pile():
    config = Config()

    # Load data
    df = pd.read_csv(config.DATA_PATH)
    df['BMA_label'] = df['BMA_label'] - 1  # Convert to 0-indexed

    # Get first pile
    first_pile = df['pile'].iloc[0]
    test_df = df[df['pile'] == first_pile].reset_index(drop=True)

    print(f"Testing with pile: {first_pile}")
    print(f"Number of images in pile: {len(test_df)}")
    print(f"True label: {test_df['BMA_label'].iloc[0]}")

    # Prepare image data list
    image_data_list = []
    for _, row in test_df.iterrows():
        image_data_list.append({
            'image_path': row['image_path'],
            'pile_id': row['pile'],
            'pile_label': row['BMA_label']
        })

    # Create CLAHE-only augmentation (preprocessing for test)
    test_augmentation = ComposedAugmentation(
        histogram_method=config.HISTOGRAM_METHOD,
        enable_geometric=False,
        enable_color=False,
        enable_noise=False,
        is_training=False,
        target_size=config.TARGET_SIZE
    )

    # Create dataset with CLAHE preprocessing
    test_dataset = BMADataset(
        image_data_list=image_data_list,
        image_dir=config.IMAGE_DIR,
        augmentation=test_augmentation,
        is_training=False,
        include_original_and_augmented=False,
        num_augmentation_versions=1
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Load model (use fold 1 model)
    model_path = 'models/best_bma_mil_model_fold1.pth'

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    print(f"Loading model: {model_path}")

    # Create feature extractor and model
    feature_extractor = FeatureExtractor(
        device=config.DEVICE,
        trainable_layers=config.TRAINABLE_FEATURE_LAYERS
    )

    model = BMA_MIL_Classifier(
        feature_extractor=feature_extractor.model,
        feature_dim=config.FEATURE_DIM,
        hidden_dim=config.IMAGE_HIDDEN_DIM,
        num_classes=config.NUM_CLASSES,
        dropout=config.DROPOUT_RATE,
        trainable_layers=config.TRAINABLE_FEATURE_LAYERS
    ).to(config.DEVICE)

    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Collect predictions
    pile_predictions = []

    print(f"\nProcessing {len(test_dataset)} images...")

    with torch.no_grad():
        for bags, labels, pile_ids, image_paths in tqdm(test_loader, desc='Testing'):
            bags = bags.to(config.DEVICE)

            for i in range(bags.shape[0]):
                bag = bags[i]
                logits, _ = model(bag)
                pred_probs = torch.softmax(logits, dim=0)
                pile_predictions.append(pred_probs.cpu().numpy())

    # Aggregate predictions
    pile_predictions_tensor = torch.tensor(np.array(pile_predictions), dtype=torch.float32).to(config.DEVICE)

    attention_model = AttentionPooling(num_classes=config.NUM_CLASSES).to(config.DEVICE)
    attention_model.eval()

    with torch.no_grad():
        aggregated = aggregate_all_methods(pile_predictions_tensor, attention_model=attention_model)

    print(f"\n{'='*60}")
    print(f"RESULTS FOR PILE: {first_pile}")
    print(f"{'='*60}")
    print(f"True Label: Class {test_df['BMA_label'].iloc[0]}")
    print(f"Number of images: {len(test_dataset)}")
    print(f"\nPredictions by pooling method:")
    print(f"{'-'*60}")

    for method in ['mean', 'max', 'majority']:
        if method in aggregated:
            agg_probs = aggregated[method]
            if isinstance(agg_probs, torch.Tensor):
                pred_class = torch.argmax(agg_probs).item()
                probs = agg_probs.cpu().numpy()
            else:
                pred_class = np.argmax(agg_probs)
                probs = agg_probs

            print(f"\n{method.upper()} POOLING:")
            print(f"  Predicted Class: {pred_class}")
            print(f"  Probabilities: {probs}")
            print(f"  Correct: {'✓' if pred_class == test_df['BMA_label'].iloc[0] else '✗'}")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    test_single_pile()

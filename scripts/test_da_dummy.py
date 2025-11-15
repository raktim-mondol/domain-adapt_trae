import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.models import BMA_MIL_Classifier
from src.utils.training import train_model_da
from configs.config import Config


class DummyExtractor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, feature_dim)

    def forward(self, x):
        h = self.act(self.conv(x))
        h = self.pool(h).view(x.size(0), -1)
        return self.fc(h)


class DummyLabeledDataset(Dataset):
    def __init__(self, n=20, num_patches=12, size=64):
        self.n = n
        self.num_patches = num_patches
        self.size = size

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        bag = torch.randn(self.num_patches, 3, self.size, self.size)
        label = torch.randint(0, Config.NUM_CLASSES, (1,)).item()
        pile_id = f'pile_{idx}'
        image_path = f'img_{idx}.png'
        return bag, label, pile_id, image_path


def main():
    Config.BEST_MODEL_PATH = 'best_dummy_da.pth'
    feature_dim = Config.FEATURE_DIM
    hidden_dim = Config.IMAGE_HIDDEN_DIM
    extractor = DummyExtractor(feature_dim)
    model = BMA_MIL_Classifier(
        feature_extractor=extractor,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=Config.NUM_CLASSES,
        dropout=Config.DROPOUT_RATE,
        trainable_layers=-1
    )
    model = model.to('cpu')
    src_ds = DummyLabeledDataset(n=20)
    tgt_ds = DummyLabeledDataset(n=20)
    val_ds = DummyLabeledDataset(n=10)
    src_loader = DataLoader(src_ds, batch_size=2, shuffle=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)
    train_losses, val_accuracies, val_f1_scores = train_model_da(
        model=model,
        train_loader_source=src_loader,
        train_loader_target=tgt_loader,
        val_loader_target=val_loader,
        num_epochs=2,
        learning_rate=1e-3,
        class_weights=None,
        fold=None
    )
    print('Train losses:', train_losses)
    print('Val accuracies:', val_accuracies)
    print('Val F1:', val_f1_scores)


if __name__ == '__main__':
    main()


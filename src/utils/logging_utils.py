"""
Logging and result saving utilities
"""

import os
import logging
from datetime import datetime


def setup_logging(log_dir='logs', mode='standard'):
    """Setup logging configuration"""
    from configs.config import Config
    
    if not Config.ENABLE_LOGGING:
        return None
    
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'bma_mil_{mode}_{timestamp}.log')
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Device: {Config.DEVICE}")
    
    return logger


def save_results_to_file(results_data, accuracy, f1_scores, confusion_matrix, mode='standard'):
    """Save training results to a text file"""
    from configs.config import Config
    
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = os.path.join(Config.LOG_DIR, f'results_{mode}_{timestamp}.txt')
    
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"BMA MIL Classifier - Results Summary\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        f.write("Configuration:\n")
        f.write(f"  Epochs: {Config.NUM_EPOCHS}\n")
        f.write(f"  Batch Size: {Config.BATCH_SIZE}\n")
        f.write(f"  Learning Rate: {Config.LEARNING_RATE}\n")
        f.write(f"  Device: {Config.DEVICE}\n\n")
        
        if mode == 'cross_validation':
            f.write(f"Cross-Validation Results ({Config.NUM_FOLDS} Folds):\n")
            f.write("="*60 + "\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            
            f.write("Per-Class F1 Scores:\n")
            for class_idx, f1 in enumerate(f1_scores):
                f.write(f"  BMA Class {class_idx + 1}: {f1:.4f}\n")
            
            f.write("\nAverage Confusion Matrix:\n")
            f.write("Predicted ->\n")
            f.write("True v    BMA1   BMA2   BMA3   BMA4\n")
            for i in range(Config.NUM_CLASSES):
                row_str = f"BMA {i+1}:  "
                for j in range(Config.NUM_CLASSES):
                    row_str += f"{confusion_matrix[i, j]:5.1f}  "
                f.write(row_str + "\n")
        else:
            f.write("Test Set Results:\n")
            f.write("="*60 + "\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n")
            f.write(f"Weighted F1 Score: {results_data['weighted_f1']:.4f}\n\n")
            
            f.write("Per-Class F1 Scores:\n")
            for class_idx, f1 in enumerate(f1_scores):
                f.write(f"  BMA Class {class_idx + 1}: {f1:.4f}\n")
            
            f.write("\nConfusion Matrix:\n")
            f.write("Predicted ->\n")
            f.write("True v    BMA1  BMA2  BMA3  BMA4\n")
            for i in range(Config.NUM_CLASSES):
                row_str = f"BMA {i+1}:  "
                for j in range(Config.NUM_CLASSES):
                    row_str += f"{confusion_matrix[i, j]:4d}  "
                f.write(row_str + "\n")
    
    print(f"\nResults saved to: {results_file}")
    if logging.getLogger(__name__).hasHandlers():
        logging.info(f"Results saved to: {results_file}")


def setup_logger(log_file):
    """Setup a simple logger for scripts"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def save_training_curves(history, output_path, mode='standard'):
    """Save training curves to file"""
    import matplotlib.pyplot as plt

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    if 'val_acc' in history:
        axes[1].plot(history['val_acc'], label='Val Acc', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to: {output_path}")

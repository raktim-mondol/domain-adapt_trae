"""
Early stopping utility for training
"""

import logging


class EarlyStopping:
    """Early stopping to stop training when validation accuracy doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_acc, epoch):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                msg = f'EarlyStopping counter: {self.counter}/{self.patience} (best: {self.best_score:.4f} at epoch {self.best_epoch})'
                print(msg)
                if logging.getLogger(__name__).hasHandlers():
                    logging.info(msg)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            
        return self.early_stop

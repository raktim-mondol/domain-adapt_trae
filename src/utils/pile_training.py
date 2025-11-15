"""
Pile-level training utilities
Training and validation both happen at pile level
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from .early_stopping import EarlyStopping


def train_one_epoch_pile_level(model, train_loader, optimizer, criterion, device, epoch, num_epochs, fold=None, pooling_method='mean', attention_model=None):
    """
    Train for one epoch at PILE LEVEL.
    Each pile is a training sample, contains multiple bags (images).
    Loss is computed per pile, gradients flow through all bags in pile.
    
    Args:
        pooling_method: One of ['mean', 'max', 'attention', 'majority']
        attention_model: AttentionPooling model (required if method='attention')
    """
    from .pooling import aggregate_pile_predictions
    
    model.train()
    if attention_model is not None and pooling_method == 'attention':
        attention_model.train()
    
    epoch_loss = 0.0
    all_preds = []
    all_labels = []
    
    fold_str = f" (Fold {fold})" if fold is not None else ""
    method_str = f" ({pooling_method.capitalize()} Pooling)"
    train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}{fold_str} - Training{method_str}', 
                     leave=False, unit='pile')
    
    for batch_idx, (pile_bags_batch, labels, pile_ids, num_bags) in enumerate(train_pbar):
        # pile_bags_batch: List of lists, each inner list contains bags for one pile
        # labels: Tensor [batch_size] - one label per pile
        
        labels = labels.to(device).long()
        batch_size = len(pile_bags_batch)
        
        optimizer.zero_grad()
        
        # Process each pile in the batch
        pile_logits = []
        
        for pile_idx in range(batch_size):
            pile_bags = pile_bags_batch[pile_idx]  # List of bags for this pile
            pile_bag_probs = []
            
            # Forward pass through each bag in the pile
            for bag in pile_bags:
                bag = bag.to(device)  # [num_patches, 3, H, W]
                logits, _ = model(bag)  # [num_classes]
                bag_probs = torch.softmax(logits, dim=0)  # [num_classes]
                pile_bag_probs.append(bag_probs)
            
            # Aggregate bag probabilities for this pile using specified method
            pile_agg_probs, _ = aggregate_pile_predictions(
                pile_bag_probs, method=pooling_method, attention_model=attention_model
            )
            
            # Convert back to logits for loss computation
            # Using log for numerical stability
            pile_logit = torch.log(pile_agg_probs + 1e-10)  # [num_classes]
            pile_logits.append(pile_logit)
        
        # Stack all pile logits
        pile_logits = torch.stack(pile_logits)  # [batch_size, num_classes]
        
        # Compute loss at pile level
        loss = criterion(pile_logits, labels)
        
        # Backward pass - gradients flow through all bags in all piles
        loss.backward()
        optimizer.step()
        
        # Track metrics
        epoch_loss += loss.item()
        preds = torch.argmax(pile_logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        train_pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{epoch_loss/(batch_idx+1):.4f}'
        })
    
    avg_loss = epoch_loss / len(train_loader)
    pile_acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, pile_acc


def validate_pile_level(model, val_loader, device, epoch, num_epochs, fold=None, criterion=None, pooling_method='mean', attention_model=None):
    """
    Validate at PILE LEVEL.
    Same aggregation as training using specified pooling method.
    
    Args:
        pooling_method: One of ['mean', 'max', 'attention', 'majority']
        attention_model: AttentionPooling model (required if method='attention')
        criterion: Loss criterion (optional, for computing validation loss)
    """
    from .pooling import aggregate_pile_predictions
    
    model.eval()
    if attention_model is not None:
        attention_model.eval()
    
    all_preds = []
    all_labels = []
    val_loss = 0.0
    num_batches = 0
    
    fold_str = f" (Fold {fold})" if fold is not None else ""
    method_str = f" ({pooling_method.capitalize()} Pooling)"
    val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs}{fold_str} - Validation{method_str}', 
                   leave=False, unit='pile')
    
    with torch.no_grad():
        for pile_bags_batch, labels, pile_ids, num_bags in val_pbar:
            labels = labels.to(device).long()
            batch_size = len(pile_bags_batch)
            
            # Process each pile
            pile_logits = []
            for pile_idx in range(batch_size):
                pile_bags = pile_bags_batch[pile_idx]
                pile_bag_probs = []
                
                # Forward pass through each bag
                for bag in pile_bags:
                    bag = bag.to(device)
                    logits, _ = model(bag)
                    bag_probs = torch.softmax(logits, dim=0)
                    pile_bag_probs.append(bag_probs)
                
                # Aggregate using specified method
                pile_agg_probs, _ = aggregate_pile_predictions(
                    pile_bag_probs, method=pooling_method, attention_model=attention_model
                )
                pred = torch.argmax(pile_agg_probs).item()
                
                all_preds.append(pred)
                all_labels.append(labels[pile_idx].item())
                
                # Compute pile logit for loss computation if criterion provided
                if criterion is not None:
                    pile_logit = torch.log(pile_agg_probs + 1e-10)
                    pile_logits.append(pile_logit)
            
            # Compute validation loss if criterion provided
            if criterion is not None and len(pile_logits) > 0:
                pile_logits_tensor = torch.stack(pile_logits)
                loss = criterion(pile_logits_tensor, labels)
                val_loss += loss.item()
                num_batches += 1
    
    pile_acc = accuracy_score(all_labels, all_preds)
    pile_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Calculate average validation loss
    avg_val_loss = val_loss / num_batches if num_batches > 0 and criterion is not None else None
    
    return pile_acc, pile_f1, all_preds, all_labels, avg_val_loss


def train_model_pile_level(model, train_loader, val_loader, num_epochs=50, learning_rate=1e-4,
                           class_weights=None, fold=None, resume_state=None):
    """
    Train the BMA MIL classifier at PILE LEVEL.
    - Training happens at pile level (each pile is a training sample)
    - Validation also at pile level
    - Gradients computed from pile-level loss, flow through all bags
    """
    from configs.config import Config
    from .pooling import AttentionPooling
    
    logger = logging.getLogger(__name__)
    device = next(model.parameters()).device
    
    # Get pooling method from config (default to 'mean')
    pooling_method = getattr(Config, 'POOLING_METHOD', 'mean')
    
    # Initialize attention pooling model if needed
    attention_model = None
    if pooling_method == 'attention':
        attention_model = AttentionPooling(num_classes=Config.NUM_CLASSES).to(device)
        print(f"Using Attention Pooling with trainable attention weights")
        if logger.hasHandlers():
            logger.info("Attention Pooling enabled - trainable attention weights")
    
    # Print device information
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n{'='*60}")
        print(f"GPU Training Enabled - PILE LEVEL")
        print(f"{'='*60}")
        print(f"GPU Device: {gpu_name}")
        print(f"Total GPU Memory: {gpu_memory:.2f} GB")
        print(f"Model is on device: {device}")
        print(f"Training Level: PILE (weights updated per pile)")
        print(f"{'='*60}\n")
        if logger.hasHandlers():
            logger.info(f"Pile-Level GPU Training - Device: {gpu_name}, Memory: {gpu_memory:.2f} GB")
    else:
        print(f"\n[WARNING] Training on CPU - PILE LEVEL")
        if logger.hasHandlers():
            logger.info("Pile-Level Training on CPU")
    
    # Setup optimizer and loss
    # Include attention model parameters if using attention pooling
    params_to_optimize = list(model.parameters())
    if attention_model is not None:
        params_to_optimize += list(attention_model.parameters())
    
    if Config.USE_ADAMW:
        optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=Config.WEIGHT_DECAY)
        optimizer_name = "AdamW"
    else:
        optimizer = torch.optim.Adam(params_to_optimize, lr=learning_rate, weight_decay=Config.WEIGHT_DECAY)
        optimizer_name = "Adam"
    
    print(f"Using {optimizer_name} optimizer (Pile-Level, LR={learning_rate}, Weight Decay={Config.WEIGHT_DECAY})")
    if logger.hasHandlers():
        logger.info(f"Pile-Level Optimizer: {optimizer_name}, LR={learning_rate}, WD={Config.WEIGHT_DECAY}")
    
    # Setup learning rate scheduler
    scheduler = None
    if Config.USE_LR_SCHEDULER:
        if Config.LR_SCHEDULER_TYPE == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=Config.LR_SCHEDULER_MODE,
                factor=Config.LR_SCHEDULER_FACTOR,
                patience=Config.LR_SCHEDULER_PATIENCE,
                min_lr=Config.LR_SCHEDULER_MIN_LR,
                threshold=Config.LR_SCHEDULER_THRESHOLD
            )
            print(f"Using ReduceLROnPlateau scheduler (mode={Config.LR_SCHEDULER_MODE}, patience={Config.LR_SCHEDULER_PATIENCE})")
            if logger.hasHandlers():
                logger.info(f"Scheduler: ReduceLROnPlateau, mode={Config.LR_SCHEDULER_MODE}, patience={Config.LR_SCHEDULER_PATIENCE}")
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted CrossEntropyLoss (Pile-Level)")
        if logger.hasHandlers():
            logger.info("Using weighted CrossEntropyLoss (Pile-Level)")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss (Pile-Level)")
        if logger.hasHandlers():
            logger.info("Using standard CrossEntropyLoss (Pile-Level)")
    
    # Training history
    train_losses = []
    val_losses = []
    val_pile_accuracies = []
    val_pile_f1_scores = []
    best_val_acc = 0.0
    
    # Early stopping
    early_stopping = None
    if Config.USE_EARLY_STOPPING:
        early_stopping = EarlyStopping(
            patience=Config.EARLY_STOPPING_PATIENCE,
            min_delta=Config.EARLY_STOPPING_MIN_DELTA
        )
        print(f"Early stopping enabled: patience={Config.EARLY_STOPPING_PATIENCE}")
        if logger.hasHandlers():
            logger.info(f"Early stopping: patience={Config.EARLY_STOPPING_PATIENCE}")
    
    # Resume training if checkpoint provided
    start_epoch = 0
    if resume_state is not None:
        start_epoch = resume_state['epoch'] + 1
        best_val_acc = resume_state.get('best_val_acc', 0.0)
        train_losses = resume_state.get('train_losses', [])
        val_losses = resume_state.get('val_losses', [])
        val_pile_accuracies = resume_state.get('val_accuracies', [])
        val_pile_f1_scores = resume_state.get('val_f1_scores', [])
        
        try:
            optimizer.load_state_dict(resume_state['optimizer_state_dict'])
            print(f"Resumed from epoch {start_epoch}, best acc: {best_val_acc:.4f}")
        except Exception as e:
            print(f"[WARNING] Could not load optimizer state: {e}")
        
        if scheduler is not None and 'scheduler_state_dict' in resume_state:
            try:
                scheduler.load_state_dict(resume_state['scheduler_state_dict'])
            except Exception as e:
                print(f"[WARNING] Could not load scheduler state: {e}")
    
    # Training loop
    fold_str = f" (Fold {fold})" if fold is not None else ""
    epoch_pbar = tqdm(range(start_epoch, num_epochs), 
                     desc=f'Training Progress{fold_str}', 
                     unit='epoch', initial=start_epoch, total=num_epochs)
    
    for epoch in epoch_pbar:
        # Train for one epoch at pile level
        train_loss, train_pile_acc = train_one_epoch_pile_level(
            model, train_loader, optimizer, criterion, device, epoch, num_epochs, fold,
            pooling_method=pooling_method, attention_model=attention_model
        )
        train_losses.append(train_loss)
        
        # Validate at pile level (with validation loss)
        pile_acc, pile_f1, _, _, val_loss = validate_pile_level(
            model, val_loader, device, epoch, num_epochs, fold, criterion=criterion,
            pooling_method=pooling_method, attention_model=attention_model
        )
        val_pile_accuracies.append(pile_acc)
        val_pile_f1_scores.append(pile_f1)
        val_losses.append(val_loss if val_loss is not None else 0.0)
        
        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Use validation loss for ReduceLROnPlateau
                scheduler.step(val_loss if val_loss is not None else train_loss)
        
        # Get current learning rate and log if changed
        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != old_lr:
            lr_msg = f'Learning rate reduced: {old_lr:.2e} -> {current_lr:.2e}'
            print(f"\n{lr_msg}")
            if logger.hasHandlers():
                logger.info(lr_msg)
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}' if val_loss is not None else 'N/A',
            'Pile Acc': f'{pile_acc:.4f}',
            'Pile F1': f'{pile_f1:.4f}',
            'LR': f'{current_lr:.2e}'
        })
        
        # Logging
        msg = f'Epoch {epoch+1}/{num_epochs}{fold_str} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Pile Acc: {pile_acc:.4f}, Pile F1: {pile_f1:.4f}, LR: {current_lr:.2e}'
        if logger.hasHandlers():
            logger.info(msg)
        
        # Save best model
        if pile_acc > best_val_acc:
            best_val_acc = pile_acc
            model_path = Config.BEST_MODEL_PATH if fold is None else f'models/best_bma_mil_model_fold{fold}.pth'
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_acc': best_val_acc,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'val_accuracies': val_pile_accuracies,
                'val_f1_scores': val_pile_f1_scores
            }
            
            # Save scheduler state if available
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, model_path)
            
            msg = f'[BEST] New best model saved (Pile Acc: {best_val_acc:.4f})'
            print(msg)
            if logger.hasHandlers():
                logger.info(msg)
        
        # Early stopping
        if early_stopping is not None:
            early_stopping(pile_acc, model)
            if early_stopping.early_stop:
                counter_msg = f"EarlyStopping counter: {early_stopping.counter}/{early_stopping.patience} (best: {best_val_acc:.4f} at epoch {epoch+1-early_stopping.counter})"
            else:
                counter_msg = f"EarlyStopping counter: {early_stopping.counter}/{early_stopping.patience} (best: {best_val_acc:.4f} at epoch {epoch+1-early_stopping.counter})"
            
            if early_stopping.early_stop:
                print(counter_msg)
                if logger.hasHandlers():
                    logger.info(counter_msg)
                stop_msg = f"Early stopping at epoch {epoch+1}. Best: {best_val_acc:.4f} at epoch {epoch+1-early_stopping.counter}"
                print(stop_msg)
                if logger.hasHandlers():
                    logger.info(stop_msg)
                break
            elif early_stopping.counter > 0:
                print(counter_msg)
                if logger.hasHandlers():
                    logger.info(counter_msg)
    
    return train_losses, val_pile_accuracies, val_pile_f1_scores


"""
Enhanced Training utilities with advanced domain adaptation techniques
"""

import torch
import torch.nn as nn
import itertools
import numpy as np
import logging
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from .early_stopping import EarlyStopping
from src.models.domain_discriminator import GradientReversal, DomainDiscriminator
from src.losses.mmd import mmd_loss
from src.losses.orthogonal import orthogonal_constraint
from src.losses.entropy import entropy_loss_from_logits
from src.losses.consistency import consistency_mse
from src.losses.enhanced_da_losses import (
    coral_loss, cka_loss, adaptive_pseudo_labeling, 
    AdaptiveLossWeighting, ProgressiveDAScheduler,
    ConditionalDomainAdversarialLoss, compute_domain_gap
)
from .ema import update_ema
import copy


def train_one_epoch_enhanced_da(model, loader_s, loader_t, optimizer, cls_criterion, dom_criterion,
                               device, epoch, num_epochs, bandwidths, lambda_adv, lambda_mmd, lambda_orth,
                               grl, domain_discriminator, 
                               lambda_coral=0.1, lambda_cka=0.1, use_coral=False, use_cka=False,
                               use_cdann=False, num_classes=3, cdan_loss=None):
    """
    Enhanced training epoch with improved domain adaptation techniques
    """
    model.train()
    epoch_loss = 0.0
    all_preds_s = []
    all_labels_s = []
    all_preds_t = []
    all_labels_t = []
    
    it_s = itertools.cycle(loader_s)
    it_t = itertools.cycle(loader_t)
    steps = max(len(loader_s), len(loader_t))
    
    loss_cls_sum = 0.0
    loss_adv_sum = 0.0
    loss_mmd_sum = 0.0
    loss_orth_sum = 0.0
    loss_coral_sum = 0.0
    loss_cka_sum = 0.0
    loss_cdann_sum = 0.0
    
    for step in range(steps):
        bags_s, labels_s, _, _ = next(it_s)
        bags_t, labels_t, _, _ = next(it_t)
        
        bags_s = bags_s.to(device)
        bags_t = bags_t.to(device)
        labels_s = labels_s.to(device).long()
        labels_t = labels_t.to(device).long()
        
        optimizer.zero_grad()
        
        # Process source data
        logits_s_list = []
        feats_s_list = []
        for i in range(bags_s.shape[0]):
            logits_s, _, z_s = model.forward_with_features(bags_s[i])
            logits_s_list.append(logits_s)
            feats_s_list.append(z_s)
        
        # Process target data
        logits_t_list = []
        feats_t_list = []
        for i in range(bags_t.shape[0]):
            logits_t, _, z_t = model.forward_with_features(bags_t[i])
            logits_t_list.append(logits_t)
            feats_t_list.append(z_t)
        
        logits_s_batch = torch.stack(logits_s_list)
        logits_t_batch = torch.stack(logits_t_list)
        z_s_batch = torch.stack(feats_s_list)
        z_t_batch = torch.stack(feats_t_list)
        
        # Classification loss (source + target)
        loss_cls = cls_criterion(logits_s_batch, labels_s) + cls_criterion(logits_t_batch, labels_t)
        
        # Domain adversarial loss
        z_concat = torch.cat([z_s_batch, z_t_batch], dim=0)
        d_labels = torch.cat([
            torch.zeros(z_s_batch.size(0), device=device),
            torch.ones(z_t_batch.size(0), device=device)
        ], dim=0)
        
        z_rev = grl(z_concat)
        d_logits = domain_discriminator(z_rev)
        loss_adv = dom_criterion(d_logits, d_labels)
        
        # MMD loss
        loss_mmd = mmd_loss(z_s_batch, z_t_batch, bandwidths)
        
        # Orthogonal constraint
        loss_orth = orthogonal_constraint(model.classifier, domain_discriminator)
        
        # Additional enhanced losses
        loss_coral = torch.tensor(0.0, device=device)
        if use_coral:
            loss_coral = coral_loss(z_s_batch, z_t_batch)
        
        loss_cka = torch.tensor(0.0, device=device)
        if use_cka:
            loss_cka = cka_loss(z_s_batch, z_t_batch)
        
        loss_cdann = torch.tensor(0.0, device=device)
        if use_cdann and cdan_loss is not None:
            # Use source features and logits for CDAN
            loss_cdann = cdan_loss(z_s_batch, logits_s_batch)
        
        # Total loss with enhanced components
        loss_total = (loss_cls + 
                     lambda_adv * loss_adv + 
                     lambda_mmd * loss_mmd + 
                     lambda_orth * loss_orth +
                     lambda_coral * loss_coral +
                     lambda_cka * loss_cka +
                     loss_cdann)
        
        loss_total.backward()
        optimizer.step()
        
        epoch_loss += loss_total.item()
        loss_cls_sum += loss_cls.item()
        loss_adv_sum += loss_adv.item()
        loss_mmd_sum += loss_mmd.item()
        loss_orth_sum += loss_orth.item()
        loss_coral_sum += loss_coral.item()
        loss_cka_sum += loss_cka.item()
        loss_cdann_sum += loss_cdann.item()
        
        # Track metrics
        preds_s = torch.argmax(logits_s_batch, dim=1)
        preds_t = torch.argmax(logits_t_batch, dim=1)
        all_preds_s.extend(preds_s.cpu().numpy())
        all_labels_s.extend(labels_s.cpu().numpy())
        all_preds_t.extend(preds_t.cpu().numpy())
        all_labels_t.extend(labels_t.cpu().numpy())
    
    avg_loss = epoch_loss / steps
    bag_acc_s = accuracy_score(all_labels_s, all_preds_s)
    bag_acc_t = accuracy_score(all_labels_t, all_preds_t)
    avg_cls = loss_cls_sum / steps
    avg_adv = loss_adv_sum / steps
    avg_mmd = loss_mmd_sum / steps
    avg_orth = loss_orth_sum / steps
    avg_coral = loss_coral_sum / steps
    avg_cka = loss_cka_sum / steps
    avg_cdann = loss_cdann_sum / steps
    
    return (avg_loss, bag_acc_s, bag_acc_t, avg_cls, avg_adv, avg_mmd, avg_orth, 
            avg_coral, avg_cka, avg_cdann)


def train_one_epoch_enhanced_uda(model, loader_s, loader_t, optimizer, cls_criterion, dom_criterion,
                                device, epoch, num_epochs, bandwidths, lambda_adv, lambda_mmd, lambda_orth,
                                lambda_ent, lambda_cons, grl, domain_discriminator, teacher, 
                                threshold=0.9, use_adaptive_pseudo=False, 
                                lambda_coral=0.1, lambda_cka=0.1, use_coral=False, use_cka=False):
    """
    Enhanced UDA training epoch with improved pseudo-labeling and feature alignment
    """
    model.train()
    epoch_loss = 0.0
    
    it_s = itertools.cycle(loader_s)
    it_t = itertools.cycle(loader_t)
    steps = max(len(loader_s), len(loader_t))
    
    for step in range(steps):
        bags_s, labels_s, _, _ = next(it_s)
        bags_t, _, _, _ = next(it_t)
        
        bags_s = bags_s.to(device)
        labels_s = labels_s.to(device).long()
        bags_t = bags_t.to(device)
        
        optimizer.zero_grad()
        
        # Process source data
        logits_s_list = []
        feats_s_list = []
        for i in range(bags_s.shape[0]):
            logits_s, _, z_s = model.forward_with_features(bags_s[i])
            logits_s_list.append(logits_s)
            feats_s_list.append(z_s)
        
        # Process target data
        logits_t_list = []
        feats_t_list = []
        for i in range(bags_t.shape[0]):
            logits_t, _, z_t = model.forward_with_features(bags_t[i])
            logits_t_list.append(logits_t)
            feats_t_list.append(z_t)
        
        logits_s_batch = torch.stack(logits_s_list)
        logits_t_batch = torch.stack(logits_t_list)
        z_s_batch = torch.stack(feats_s_list)
        z_t_batch = torch.stack(feats_t_list)
        
        # Source classification loss
        loss_cls = cls_criterion(logits_s_batch, labels_s)
        
        # Domain adversarial loss
        z_concat = torch.cat([z_s_batch, z_t_batch], dim=0)
        d_labels = torch.cat([
            torch.zeros(z_s_batch.size(0), device=device),
            torch.ones(z_t_batch.size(0), device=device)
        ], dim=0)
        
        z_rev = grl(z_concat)
        d_logits = domain_discriminator(z_rev)
        loss_adv = dom_criterion(d_logits, d_labels)
        
        # Target entropy loss
        loss_ent = entropy_loss_from_logits(logits_t_batch)
        
        # Consistency loss with teacher model
        probs_t = torch.softmax(logits_t_batch, dim=-1)
        with torch.no_grad():
            teacher.eval()
            teacher_logits_list = []
            for i in range(bags_t.shape[0]):
                tl, _ = teacher(bags_t[i])
                teacher_logits_list.append(tl)
            teacher_logits = torch.stack(teacher_logits_list)
            teacher_probs = torch.softmax(teacher_logits, dim=-1)
        
        loss_cons = consistency_mse(probs_t, teacher_probs)
        
        # MMD loss
        loss_mmd_total = mmd_loss(z_s_batch, z_t_batch, bandwidths)
        
        # Enhanced pseudo-labeling
        if use_adaptive_pseudo:
            pseudo_labels, mask_t = adaptive_pseudo_labeling(logits_t_batch, threshold=threshold)
        else:
            probs_t = torch.softmax(logits_t_batch, dim=-1)
            max_probs, pseudo_labels = torch.max(probs_t, dim=-1)
            mask_t = max_probs >= threshold
        
        loss_pseudo = torch.tensor(0.0, device=device)
        if mask_t.any():
            loss_pseudo = cls_criterion(logits_t_batch[mask_t], pseudo_labels[mask_t])
        
        # Additional enhanced losses
        loss_coral = torch.tensor(0.0, device=device)
        if use_coral:
            loss_coral = coral_loss(z_s_batch, z_t_batch)
        
        loss_cka = torch.tensor(0.0, device=device)
        if use_cka:
            loss_cka = cka_loss(z_s_batch, z_t_batch)
        
        # Total loss with enhanced components
        loss_total = (loss_cls + 
                     lambda_adv * loss_adv + 
                     lambda_mmd * loss_mmd_total + 
                     lambda_orth * orthogonal_constraint(model.classifier, domain_discriminator) + 
                     lambda_ent * loss_ent + 
                     lambda_cons * loss_cons + 
                     loss_pseudo +
                     lambda_coral * loss_coral +
                     lambda_cka * loss_cka)
        
        loss_total.backward()
        optimizer.step()
        
        # Update teacher model with EMA
        update_ema(model, teacher, decay=0.99)
        
        epoch_loss += loss_total.item()
    
    avg_loss = epoch_loss / steps
    return avg_loss


def train_model_enhanced_da(model, train_loader_source, train_loader_target, val_loader_target,
                           num_epochs=50, learning_rate=1e-4, class_weights=None, fold=None,
                           resume_state=None,
                           use_coral=True, use_cka=True, use_cdann=False,
                           lambda_coral=0.1, lambda_cka=0.1, lambda_cdann=0.1):
    """
    Enhanced domain adaptation training with multiple improved techniques
    """
    from configs.config import Config
    from .training import validate_pile_level  # Import from original training module
    logger = logging.getLogger(__name__)
    device = next(model.parameters()).device
    
    # Setup optimizer
    if Config.USE_ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY)
    
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
    
    # Setup criteria
    if class_weights is not None:
        cls_criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        cls_criterion = nn.CrossEntropyLoss()
    
    dom_criterion = nn.BCEWithLogitsLoss()
    grl = GradientReversal(coeff=Config.GRL_COEFF)
    domain_discriminator = DomainDiscriminator(in_dim=Config.IMAGE_HIDDEN_DIM, dropout=Config.DROPOUT_RATE).to(device)
    
    # Setup CDAN if enabled
    cdan_loss = None
    if use_cdann:
        cdan_loss = ConditionalDomainAdversarialLoss(
            feature_dim=Config.IMAGE_HIDDEN_DIM, 
            num_classes=Config.NUM_CLASSES
        ).to(device)
    
    # Setup adaptive loss weighting
    num_losses = 4  # cls, adv, mmd, orth (additional losses handled separately)
    adaptive_weighting = AdaptiveLossWeighting(num_losses)
    
    # Setup progressive DA scheduler
    progressive_scheduler = ProgressiveDAScheduler(total_epochs=num_epochs)
    
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
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            verbose=True
        )
    
    start_epoch = 0
    fold_str = f" (Fold {fold})" if fold is not None else ""
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc=f'Training Progress{fold_str}', 
                     unit='epoch', initial=start_epoch, total=num_epochs)
    
    for epoch in epoch_pbar:
        # Adjust domain adaptation parameters based on progressive scheduler
        da_weight = progressive_scheduler.get_domain_weight(epoch)
        lambda_adv = Config.LAMBDA_ADV * da_weight
        lambda_mmd = Config.LAMBDA_MMD * da_weight
        lambda_orth = Config.LAMBDA_ORTH * da_weight
        
        # Update GRL coefficient
        grl.coeff = Config.GRL_COEFF * da_weight
        
        # Train for one epoch with enhanced DA
        (train_loss, train_bag_acc_s, train_bag_acc_t, avg_cls, avg_adv, avg_mmd, avg_orth, 
         avg_coral, avg_cka, avg_cdann) = train_one_epoch_enhanced_da(
            model, train_loader_source, train_loader_target, optimizer, cls_criterion, dom_criterion,
            device, epoch, num_epochs, Config.MMD_BANDWIDTHS, lambda_adv, lambda_mmd, lambda_orth,
            grl, domain_discriminator,
            lambda_coral=lambda_coral, lambda_cka=lambda_cka, 
            use_coral=use_coral, use_cka=use_cka, 
            use_cdann=use_cdann, num_classes=Config.NUM_CLASSES, 
            cdan_loss=cdan_loss
        )
        
        train_losses.append(train_loss)
        
        # Validate on pile level
        pile_acc, pile_f1, _, _, val_loss = validate_pile_level(
            model, val_loader_target, device, epoch, num_epochs, fold, 
            criterion=cls_criterion, pooling_method='mean'
        )
        
        val_pile_accuracies.append(pile_acc)
        val_pile_f1_scores.append(pile_f1)
        val_losses.append(val_loss if val_loss is not None else 0.0)
        
        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loss is not None else train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update progress bar with enhanced metrics
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}' if val_loss is not None else 'N/A',
            'Pile Acc': f'{pile_acc:.4f}',
            'Pile F1': f'{pile_f1:.4f}',
            'LR': f'{current_lr:.2e}',
            'L_cls': f'{avg_cls:.4f}',
            'L_adv': f'{avg_adv:.4f}',
            'L_mmd': f'{avg_mmd:.4f}',
            'L_orth': f'{avg_orth:.4f}',
            'L_coral': f'{avg_coral:.4f}',
            'L_cka': f'{avg_cka:.4f}',
            'L_cdann': f'{avg_cdann:.4f}'
        })
        
        # Save best model
        if pile_acc > best_val_acc:
            best_val_acc = pile_acc
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
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, Config.BEST_MODEL_PATH)
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(pile_acc, epoch + 1):
                break
    
    return train_losses, val_pile_accuracies, val_pile_f1_scores


def train_model_enhanced_uda(model, train_loader_source, train_loader_target, val_loader_target,
                            num_epochs=50, learning_rate=1e-4, class_weights=None, fold=None,
                            resume_state=None,
                            use_adaptive_pseudo=True, threshold=0.9,
                            use_coral=True, use_cka=True,
                            lambda_coral=0.1, lambda_cka=0.1):
    """
    Enhanced UDA training with improved pseudo-labeling and feature alignment
    """
    from configs.config import Config
    from .training import validate_pile_level  # Import from original training module
    logger = logging.getLogger(__name__)
    device = next(model.parameters()).device
    
    # Setup optimizer
    if Config.USE_ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY)
    
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
    
    # Setup criteria
    if class_weights is not None:
        cls_criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        cls_criterion = nn.CrossEntropyLoss()
    
    dom_criterion = nn.BCEWithLogitsLoss()
    grl = GradientReversal(coeff=Config.GRL_COEFF)
    domain_discriminator = DomainDiscriminator(in_dim=Config.IMAGE_HIDDEN_DIM, dropout=Config.DROPOUT_RATE).to(device)
    
    # Create teacher model
    teacher = copy.deepcopy(model).to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Setup progressive DA scheduler
    progressive_scheduler = ProgressiveDAScheduler(total_epochs=num_epochs)
    
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
            min_delta=Config.EARLY_STOPPING_MIN_DELTA,
            verbose=True
        )
    
    start_epoch = 0
    fold_str = f" (Fold {fold})" if fold is not None else ""
    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc=f'Training Progress{fold_str}', 
                     unit='epoch', initial=start_epoch, total=num_epochs)
    
    for epoch in epoch_pbar:
        # Adjust parameters based on progressive schedule
        ramp = min(1.0, (epoch + 1) / max(1, getattr(Config, 'UDA_WARMUP_EPOCHS', 5)))
        lambda_adv = Config.LAMBDA_ADV * ramp
        lambda_mmd = Config.LAMBDA_MMD * ramp
        lambda_orth = Config.LAMBDA_ORTH
        lambda_ent = Config.LAMBDA_ENT
        lambda_cons = Config.LAMBDA_CONS
        grl.coeff = Config.GRL_COEFF * ramp
        
        # Adaptive threshold for pseudo-labeling
        current_threshold = threshold if epoch + 1 > getattr(Config, 'UDA_WARMUP_EPOCHS', 5) else 0.0
        
        # Train for one epoch with enhanced UDA
        train_loss = train_one_epoch_enhanced_uda(
            model, train_loader_source, train_loader_target, optimizer, cls_criterion, dom_criterion,
            device, epoch, num_epochs, Config.MMD_BANDWIDTHS, lambda_adv, lambda_mmd, lambda_orth,
            lambda_ent, lambda_cons, grl, domain_discriminator, teacher,
            threshold=current_threshold, use_adaptive_pseudo=use_adaptive_pseudo,
            lambda_coral=lambda_coral, lambda_cka=lambda_cka, 
            use_coral=use_coral, use_cka=use_cka
        )
        
        train_losses.append(train_loss)
        
        # Validate on pile level
        pile_acc, pile_f1, _, _, val_loss = validate_pile_level(
            model, val_loader_target, device, epoch, num_epochs, fold, 
            criterion=cls_criterion, pooling_method='mean'
        )
        
        val_pile_accuracies.append(pile_acc)
        val_pile_f1_scores.append(pile_f1)
        val_losses.append(val_loss if val_loss is not None else 0.0)
        
        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss if val_loss is not None else train_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}' if val_loss is not None else 'N/A',
            'Pile Acc': f'{pile_acc:.4f}',
            'Pile F1': f'{pile_f1:.4f}',
            'LR': f'{current_lr:.2e}'
        })
        
        # Save best model
        if pile_acc > best_val_acc:
            best_val_acc = pile_acc
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
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, Config.BEST_MODEL_PATH)
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(pile_acc, epoch + 1):
                break
    
    return train_losses, val_pile_accuracies, val_pile_f1_scores


# We'll import the original validation function from the training module
# This avoids duplicating code and potential inconsistencies
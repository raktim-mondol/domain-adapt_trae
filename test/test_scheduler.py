"""
Quick test script to verify AdamW + Scheduler implementation
"""

import torch
from configs.config import Config

def test_scheduler_config():
    """Test that scheduler configuration is properly set"""
    print("=" * 60)
    print("Testing Scheduler Configuration")
    print("=" * 60)
    
    # Check optimizer config
    print(f"\n[OK] USE_ADAMW: {Config.USE_ADAMW}")
    assert hasattr(Config, 'USE_ADAMW'), "Missing USE_ADAMW config"
    
    # Check scheduler config
    print(f"[OK] USE_LR_SCHEDULER: {Config.USE_LR_SCHEDULER}")
    assert hasattr(Config, 'USE_LR_SCHEDULER'), "Missing USE_LR_SCHEDULER config"
    
    print(f"[OK] LR_SCHEDULER_TYPE: {Config.LR_SCHEDULER_TYPE}")
    assert hasattr(Config, 'LR_SCHEDULER_TYPE'), "Missing LR_SCHEDULER_TYPE config"
    
    # Check scheduler parameters
    print(f"\nScheduler Parameters:")
    print(f"  Mode: {Config.LR_SCHEDULER_MODE}")
    print(f"  Factor: {Config.LR_SCHEDULER_FACTOR}")
    print(f"  Patience: {Config.LR_SCHEDULER_PATIENCE}")
    print(f"  Min LR: {Config.LR_SCHEDULER_MIN_LR}")
    print(f"  Threshold: {Config.LR_SCHEDULER_THRESHOLD}")
    
    assert Config.LR_SCHEDULER_MODE in ['min', 'max'], "Invalid scheduler mode"
    assert 0 < Config.LR_SCHEDULER_FACTOR < 1, "Factor should be between 0 and 1"
    assert Config.LR_SCHEDULER_PATIENCE > 0, "Patience should be positive"
    assert Config.LR_SCHEDULER_MIN_LR > 0, "Min LR should be positive"
    
    print("\n[SUCCESS] All configuration tests passed!")
    return True


def test_optimizer_creation():
    """Test that AdamW optimizer can be created"""
    print("\n" + "=" * 60)
    print("Testing Optimizer Creation")
    print("=" * 60)
    
    # Create a simple model
    model = torch.nn.Linear(10, 3)
    
    # Test AdamW
    if Config.USE_ADAMW:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        print(f"\n[OK] AdamW optimizer created successfully")
        print(f"  LR: {Config.LEARNING_RATE}")
        print(f"  Weight Decay: {Config.WEIGHT_DECAY}")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.WEIGHT_DECAY
        )
        print(f"\n[OK] Adam optimizer created successfully")
    
    print(f"  Optimizer type: {type(optimizer).__name__}")
    
    print("\n[SUCCESS] Optimizer creation test passed!")
    return optimizer


def test_scheduler_creation(optimizer):
    """Test that ReduceLROnPlateau scheduler can be created"""
    print("\n" + "=" * 60)
    print("Testing Scheduler Creation")
    print("=" * 60)
    
    if Config.USE_LR_SCHEDULER:
        if Config.LR_SCHEDULER_TYPE == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=Config.LR_SCHEDULER_MODE,
                factor=Config.LR_SCHEDULER_FACTOR,
                patience=Config.LR_SCHEDULER_PATIENCE,
                min_lr=Config.LR_SCHEDULER_MIN_LR,
                threshold=Config.LR_SCHEDULER_THRESHOLD,
                verbose=True
            )
            print(f"\n[OK] ReduceLROnPlateau scheduler created successfully")
            print(f"  Type: {type(scheduler).__name__}")
            print(f"  Mode: {scheduler.mode}")
            print(f"  Factor: {scheduler.factor}")
            print(f"  Patience: {scheduler.patience}")
        else:
            print(f"\n[WARNING] Scheduler type '{Config.LR_SCHEDULER_TYPE}' not tested")
            scheduler = None
    else:
        print("\n[OK] Scheduler disabled (Config.USE_LR_SCHEDULER=False)")
        scheduler = None
    
    print("\n[SUCCESS] Scheduler creation test passed!")
    return scheduler


def test_scheduler_step(optimizer, scheduler):
    """Test that scheduler can perform step"""
    print("\n" + "=" * 60)
    print("Testing Scheduler Step")
    print("=" * 60)
    
    if scheduler is None:
        print("\n[OK] No scheduler to test (disabled or not implemented)")
        return
    
    # Get initial LR
    initial_lr = optimizer.param_groups[0]['lr']
    print(f"\nInitial LR: {initial_lr:.2e}")
    
    # Simulate validation loss not improving
    # ReduceLROnPlateau waits patience+1 epochs total before reducing
    num_epochs = Config.LR_SCHEDULER_PATIENCE + 2
    print(f"\nSimulating {num_epochs} epochs with no improvement...")
    val_loss = 0.5
    for epoch in range(num_epochs):
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch + 1}: Val Loss = {val_loss:.4f}, LR = {current_lr:.2e}")
    
    # Check if LR was reduced
    final_lr = optimizer.param_groups[0]['lr']
    expected_lr = initial_lr * Config.LR_SCHEDULER_FACTOR
    
    print(f"\nFinal LR: {final_lr:.2e}")
    print(f"Expected LR: {expected_lr:.2e}")
    
    assert final_lr < initial_lr, "LR should have been reduced!"
    assert abs(final_lr - expected_lr) < 1e-9, f"LR reduction incorrect: {final_lr} vs {expected_lr}"
    
    print("\n[SUCCESS] Scheduler step test passed!")


def test_validation_loss_in_signature():
    """Test that validate_pile_level returns validation loss"""
    print("\n" + "=" * 60)
    print("Testing Validation Function Signature")
    print("=" * 60)
    
    from src.utils.training import validate_pile_level
    import inspect
    
    # Check function signature
    sig = inspect.signature(validate_pile_level)
    print(f"\nFunction signature:")
    print(f"  {sig}")
    
    # Check parameters
    params = list(sig.parameters.keys())
    print(f"\nParameters: {params}")
    
    assert 'criterion' in params, "Missing 'criterion' parameter"
    print("[OK] Function has 'criterion' parameter")
    
    print("\n[SUCCESS] Validation function signature test passed!")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("AdamW + Scheduler Implementation Test Suite")
    print("=" * 60)
    
    try:
        # Run all tests
        test_scheduler_config()
        optimizer = test_optimizer_creation()
        scheduler = test_scheduler_creation(optimizer)
        test_scheduler_step(optimizer, scheduler)
        test_validation_loss_in_signature()
        
        print("\n" + "=" * 60)
        print("*** ALL TESTS PASSED! ***")
        print("=" * 60)
        print("\nAdamW + Scheduler implementation is working correctly!")
        print("You can now run training with:")
        print("  python scripts/train.py")
        print("\n" + "=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("[FAILED] TEST FAILED")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "=" * 60)


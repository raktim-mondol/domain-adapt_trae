"""
Quick script to verify GPU training is working
"""
import torch
from configs.config import Config

print("="*60)
print("GPU Training Check")
print("="*60)

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"Current device: {torch.cuda.current_device()}")

# Check config
print(f"\nConfig.DEVICE: {Config.DEVICE}")

# Test tensor creation on GPU
if Config.DEVICE == 'cuda':
    try:
        test_tensor = torch.randn(100, 100).to(Config.DEVICE)
        print(f"\n[OK] Test tensor created on GPU: {test_tensor.device}")
        
        # Test computation
        result = torch.matmul(test_tensor, test_tensor)
        print(f"[OK] Matrix multiplication on GPU successful")
        
        # Check memory usage
        print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        print(f"\n{'='*60}")
        print("[SUCCESS] GPU training is properly configured!")
        print("="*60)
        
    except Exception as e:
        print(f"\n[ERROR] Error using GPU: {e}")
        print("Training will fall back to CPU")
else:
    print(f"\n[WARNING] GPU not available, training will use CPU")

print()


"""
Simple test script to verify environment setup
"""

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        
        import torchvision
        print(f"✓ Torchvision: {torchvision.__version__}")
        
        import timm
        print(f"✓ Timm: {timm.__version__}")
        
        import pandas as pd
        print(f"✓ Pandas: {pd.__version__}")
        
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
        
        import sklearn
        print(f"✓ Scikit-learn: {sklearn.__version__}")
        
        import matplotlib
        print(f"✓ Matplotlib: {matplotlib.__version__}")
        
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
        
        import tqdm
        print(f"✓ TQDM: {tqdm.__version__}")
        
        print("\n✓ All packages imported successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error importing packages: {e}")
        return False


def test_basic_operations():
    """Test basic operations with key packages"""
    print("\nTesting basic operations...")
    
    try:
        # Test PyTorch
        import torch
        x = torch.randn(3, 4)
        y = torch.randn(4, 5)
        z = torch.matmul(x, y)
        print(f"✓ PyTorch operation: {x.shape} @ {y.shape} = {z.shape}")
        
        # Test NumPy
        import numpy as np
        arr = np.random.rand(3, 4)
        print(f"✓ NumPy array shape: {arr.shape}")
        
        # Test Pandas
        import pandas as pd
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        print(f"✓ Pandas DataFrame shape: {df.shape}")
        
        print("\n✓ Basic operations successful!")
        return True
        
    except Exception as e:
        print(f"✗ Error in basic operations: {e}")
        return False


def main():
    print("="*50)
    print("ENVIRONMENT SETUP VERIFICATION")
    print("="*50)
    
    success1 = test_imports()
    success2 = test_basic_operations()
    
    print("\n" + "="*50)
    if success1 and success2:
        print("✓ ENVIRONMENT SETUP SUCCESSFUL!")
        print("All required packages are available and functional.")
    else:
        print("✗ ENVIRONMENT SETUP FAILED!")
        print("Some packages are missing or not functioning properly.")
    print("="*50)


if __name__ == "__main__":
    main()
# GPU Testing Script with PyTorch

This comprehensive GPU testing script evaluates your GPU performance using intensive PyTorch operations.

## Features

- **GPU Detection**: Automatically detects CUDA availability and GPU specifications
- **Memory Monitoring**: Real-time GPU memory usage tracking
- **Intensive Operations**: 
  - Large matrix multiplications
  - Neural network training
  - Convolutional operations
  - Memory stress testing
  - Performance benchmarking
- **Comprehensive Testing**: Multiple test scenarios to fully utilize GPU capabilities

## Requirements

- Python 3.7+
- CUDA-compatible GPU (optional, will run on CPU if not available)
- Conda environment named 'mine2' (as per your setup)

## Installation & Usage

### Method 1: Using the batch file (Windows)
```bash
run_gpu_test.bat
```

### Method 2: Manual execution
```bash
# Activate your conda environment
conda activate mine2

# Install requirements
pip install -r requirements.txt

# Run the test
python gpu_test.py
```

## What the Script Tests

### 1. Basic Operations
- Large tensor creation and manipulation
- Matrix multiplication performance
- Element-wise operations

### 2. Neural Network Training
- Complex multi-layer network training
- Batch processing with real gradients
- Memory usage during training

### 3. Convolutional Operations
- Deep CNN forward passes
- Different image sizes and batch sizes
- Throughput measurements

### 4. Memory Stress Test
- Progressive memory allocation
- Maximum memory capacity testing
- Memory cleanup verification

### 5. Performance Benchmarks
- Matrix multiplication timing
- FFT operations
- Convolution performance
- Statistical analysis of operation times

## Output Information

The script provides detailed information about:
- GPU specifications and CUDA version
- Memory usage throughout testing
- Operation timing and throughput
- Performance benchmarks with statistics
- Error handling and cleanup status

## Safety Features

- Automatic memory cleanup after each test
- Memory limit protection during stress testing
- Error handling and graceful degradation
- CPU fallback if GPU is not available

## Troubleshooting

If you encounter issues:
1. Ensure CUDA drivers are properly installed
2. Verify PyTorch CUDA compatibility
3. Check available GPU memory
4. Make sure the conda environment is activated

The script will automatically fall back to CPU mode if GPU is not available.

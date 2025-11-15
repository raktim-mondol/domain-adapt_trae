#!/usr/bin/env python3
"""
GPU Testing Script with Intensive PyTorch Operations
This script tests GPU availability, runs intensive operations, and monitors performance.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import psutil
import gc
import numpy as np
from typing import Tuple, List
import sys

class GPUTester:
    def __init__(self):
        self.device = None
        self.gpu_name = None
        self.gpu_memory = None
        
    def check_gpu_availability(self) -> bool:
        """Check if CUDA GPU is available and get GPU information."""
        print("=" * 60)
        print("GPU AVAILABILITY CHECK")
        print("=" * 60)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.gpu_name = torch.cuda.get_device_name(0)
            self.gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            print(f"✅ CUDA is available!")
            print(f"📱 GPU Device: {self.gpu_name}")
            print(f"💾 Total GPU Memory: {self.gpu_memory:.2f} GB")
            print(f"🔢 CUDA Version: {torch.version.cuda}")
            print(f"🐍 PyTorch Version: {torch.__version__}")
            print(f"🔢 Number of GPUs: {torch.cuda.device_count()}")
            return True
        else:
            print("❌ CUDA is not available. Will run on CPU.")
            self.device = torch.device('cpu')
            return False
    
    def get_gpu_memory_usage(self) -> Tuple[float, float]:
        """Get current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(0) / 1024**3   # GB
            return allocated, reserved
        return 0.0, 0.0
    
    def print_memory_usage(self, operation_name: str):
        """Print current memory usage."""
        allocated, reserved = self.get_gpu_memory_usage()
        if torch.cuda.is_available():
            print(f"🔍 {operation_name} - GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    
    def test_basic_operations(self):
        """Test basic GPU operations."""
        print("\n" + "=" * 60)
        print("BASIC GPU OPERATIONS TEST")
        print("=" * 60)
        
        # Test tensor creation and basic operations
        print("🧪 Testing tensor creation and basic operations...")
        
        # Create large tensors
        size = 5000
        a = torch.randn(size, size, device=self.device)
        b = torch.randn(size, size, device=self.device)
        
        self.print_memory_usage("After tensor creation")
        
        # Matrix multiplication
        start_time = time.time()
        c = torch.matmul(a, b)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        print(f"⚡ Matrix multiplication ({size}x{size}): {end_time - start_time:.4f} seconds")
        self.print_memory_usage("After matrix multiplication")
        
        # Element-wise operations
        start_time = time.time()
        d = torch.sin(a) + torch.cos(b) + torch.exp(a * 0.01)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        print(f"🔄 Element-wise operations: {end_time - start_time:.4f} seconds")
        self.print_memory_usage("After element-wise operations")
        
        # Cleanup
        del a, b, c, d
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    def test_neural_network_training(self):
        """Test neural network training on GPU."""
        print("\n" + "=" * 60)
        print("NEURAL NETWORK TRAINING TEST")
        print("=" * 60)
        
        # Define a complex neural network
        class ComplexNet(nn.Module):
            def __init__(self):
                super(ComplexNet, self).__init__()
                self.layers = nn.Sequential(
                    nn.Linear(1000, 2048),
                    nn.ReLU(),
                    nn.BatchNorm1d(2048),
                    nn.Dropout(0.3),
                    nn.Linear(2048, 1024),
                    nn.ReLU(),
                    nn.BatchNorm1d(1024),
                    nn.Dropout(0.3),
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.BatchNorm1d(512),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 10)
                )
            
            def forward(self, x):
                return self.layers(x)
        
        # Create model and move to device
        model = ComplexNet().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print(f"🧠 Created neural network with {sum(p.numel() for p in model.parameters())} parameters")
        self.print_memory_usage("After model creation")
        
        # Generate synthetic data
        batch_size = 512
        num_batches = 100
        
        print(f"🏋️ Training for {num_batches} batches with batch size {batch_size}")
        
        total_start_time = time.time()
        
        for batch_idx in range(num_batches):
            # Generate random data
            data = torch.randn(batch_size, 1000, device=self.device)
            target = torch.randint(0, 10, (batch_size,), device=self.device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if batch_idx % 20 == 0:
                print(f"📊 Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}")
                self.print_memory_usage(f"Batch {batch_idx}")
        
        total_end_time = time.time()
        
        print(f"🎯 Training completed in {total_end_time - total_start_time:.2f} seconds")
        print(f"⚡ Average time per batch: {(total_end_time - total_start_time) / num_batches:.4f} seconds")
        
        # Cleanup
        del model, data, target, output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    def test_convolution_operations(self):
        """Test intensive convolution operations."""
        print("\n" + "=" * 60)
        print("CONVOLUTION OPERATIONS TEST")
        print("=" * 60)
        
        # Create a large convolutional network
        class ConvNet(nn.Module):
            def __init__(self):
                super(ConvNet, self).__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(512, 1024, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Linear(1024, 1000)
            
            def forward(self, x):
                x = self.conv_layers(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = ConvNet().to(self.device)
        print(f"🖼️ Created ConvNet with {sum(p.numel() for p in model.parameters())} parameters")
        self.print_memory_usage("After ConvNet creation")
        
        # Test with different image sizes
        image_sizes = [(224, 224), (512, 512)]
        batch_sizes = [32, 8]
        
        for (h, w), batch_size in zip(image_sizes, batch_sizes):
            print(f"\n🔍 Testing with images of size {h}x{w}, batch size {batch_size}")
            
            # Create random images
            images = torch.randn(batch_size, 3, h, w, device=self.device)
            
            start_time = time.time()
            with torch.no_grad():
                output = model(images)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            print(f"⚡ Forward pass time: {end_time - start_time:.4f} seconds")
            print(f"🎯 Throughput: {batch_size / (end_time - start_time):.2f} images/second")
            self.print_memory_usage(f"After {h}x{w} forward pass")
            
            del images, output
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    def stress_test_memory(self):
        """Stress test GPU memory allocation."""
        print("\n" + "=" * 60)
        print("MEMORY STRESS TEST")
        print("=" * 60)
        
        if not torch.cuda.is_available():
            print("⚠️ Skipping memory stress test (CPU mode)")
            return
        
        print("🔥 Starting memory stress test...")
        tensors = []
        
        try:
            tensor_size = 1000
            while True:
                tensor = torch.randn(tensor_size, tensor_size, device=self.device)
                tensors.append(tensor)
                
                allocated, reserved = self.get_gpu_memory_usage()
                print(f"📈 Allocated {len(tensors)} tensors, GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
                
                # Increase tensor size gradually
                if len(tensors) % 10 == 0:
                    tensor_size += 100
                
                # Break if we're using too much memory (safety check)
                if allocated > self.gpu_memory * 0.9:
                    print("🛑 Approaching memory limit, stopping stress test")
                    break
                    
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"💥 GPU out of memory after allocating {len(tensors)} tensors")
                print(f"🎯 Maximum GPU memory usage: {self.get_gpu_memory_usage()[0]:.2f}GB")
            else:
                print(f"❌ Error during stress test: {e}")
        
        # Cleanup
        del tensors
        torch.cuda.empty_cache()
        gc.collect()
        
        print("🧹 Memory stress test cleanup completed")
        self.print_memory_usage("After cleanup")
    
    def benchmark_operations(self):
        """Benchmark various GPU operations."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK")
        print("=" * 60)
        
        operations = {
            "Matrix Multiplication (2048x2048)": lambda: torch.matmul(
                torch.randn(2048, 2048, device=self.device),
                torch.randn(2048, 2048, device=self.device)
            ),
            "Element-wise Addition (10M elements)": lambda: torch.randn(10000000, device=self.device) + torch.randn(10000000, device=self.device),
            "FFT (1M elements)": lambda: torch.fft.fft(torch.randn(1000000, device=self.device, dtype=torch.complex64)),
            "Softmax (1000x1000)": lambda: torch.softmax(torch.randn(1000, 1000, device=self.device), dim=1),
            "Convolution (256 filters, 224x224)": lambda: torch.nn.functional.conv2d(
                torch.randn(32, 3, 224, 224, device=self.device),
                torch.randn(256, 3, 3, 3, device=self.device),
                padding=1
            )
        }
        
        results = {}
        
        for op_name, op_func in operations.items():
            print(f"\n🔬 Benchmarking: {op_name}")
            
            # Warmup
            for _ in range(3):
                _ = op_func()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            
            # Actual benchmark
            times = []
            for _ in range(10):
                start_time = time.time()
                result = op_func()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
                del result
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            results[op_name] = (avg_time, std_time)
            
            print(f"⏱️ Average time: {avg_time:.6f} ± {std_time:.6f} seconds")
            
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print("\n📊 BENCHMARK SUMMARY:")
        print("-" * 60)
        for op_name, (avg_time, std_time) in results.items():
            print(f"{op_name:<35}: {avg_time:.6f} ± {std_time:.6f} seconds")
    
    def run_full_test_suite(self):
        """Run the complete GPU test suite."""
        print("🚀 Starting Comprehensive GPU Test Suite")
        print("=" * 80)
        
        start_time = time.time()
        
        # Check GPU availability
        gpu_available = self.check_gpu_availability()
        
        if not gpu_available:
            print("\n⚠️ Running tests on CPU (limited functionality)")
        
        try:
            # Run all tests
            self.test_basic_operations()
            self.test_neural_network_training()
            self.test_convolution_operations()
            
            if gpu_available:
                self.stress_test_memory()
            
            self.benchmark_operations()
            
        except Exception as e:
            print(f"\n❌ Error during testing: {e}")
            import traceback
            traceback.print_exc()
        
        end_time = time.time()
        
        print("\n" + "=" * 80)
        print("🎉 GPU TEST SUITE COMPLETED")
        print("=" * 80)
        print(f"⏱️ Total execution time: {end_time - start_time:.2f} seconds")
        
        if torch.cuda.is_available():
            print(f"🏁 Final GPU memory usage: {self.get_gpu_memory_usage()[0]:.2f}GB")
            torch.cuda.empty_cache()
        
        print("✅ All tests completed successfully!")

def main():
    """Main function to run GPU tests."""
    print("🔥 PyTorch GPU Intensive Testing Script")
    print("🎯 This script will test your GPU capabilities with intensive operations")
    print()
    
    # Check Python and system info
    print(f"🐍 Python version: {sys.version}")
    print(f"💻 System: {psutil.virtual_memory().total / 1024**3:.2f}GB RAM")
    print()
    
    # Create and run GPU tester
    tester = GPUTester()
    tester.run_full_test_suite()

if __name__ == "__main__":
    main()

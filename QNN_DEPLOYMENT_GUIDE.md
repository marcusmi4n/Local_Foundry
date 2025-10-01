# Qualcomm QNN Deployment Guide for Shisa v2 Qwen2.5-7B ONNX

## 📱 Complete Guide to Running on Qualcomm Hardware

This comprehensive guide covers deploying the Shisa v2 Qwen2.5-7B ONNX model on Qualcomm devices with QNN (Qualcomm Neural Network) acceleration.

---

## 🎯 Table of Contents

1. [What is QNN?](#what-is-qnn)
2. [Supported Hardware](#supported-hardware)
3. [Prerequisites](#prerequisites)
4. [QNN SDK Setup](#qnn-sdk-setup)
5. [Building ONNX Runtime with QNN](#building-onnx-runtime-with-qnn)
6. [Model Deployment](#model-deployment)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Production Deployment](#production-deployment)

---

## 🤖 What is QNN?

**Qualcomm Neural Network (QNN) SDK** is a software development kit that enables developers to run neural networks on Qualcomm hardware accelerators:

- **Hexagon Tensor Processor (HTP)**: Specialized NPU for AI workloads
- **Adreno GPU**: Graphics processor with compute capabilities
- **Qualcomm AI Engine**: Unified AI acceleration framework

### Why Use QNN?

| Benefit | Description |
|---------|-------------|
| **10-50x Speedup** | Hardware acceleration vs CPU |
| **Lower Power** | 5-10x more energy efficient |
| **On-Device** | No cloud dependency |
| **Low Latency** | Sub-50ms per token possible |
| **Privacy** | Data stays on device |

---

## 🔧 Supported Hardware

### Qualcomm Snapdragon Processors

| Processor | NPU | AI Performance | Status |
|-----------|-----|----------------|--------|
| **Snapdragon 8 Gen 3** | Hexagon NPU | 98 TOPS | ✅ Best |
| **Snapdragon 8 Gen 2** | Hexagon NPU | 60 TOPS | ✅ Excellent |
| **Snapdragon 8 Gen 1** | Hexagon 770 | 29 TOPS | ✅ Good |
| **Snapdragon 888** | Hexagon 780 | 26 TOPS | ✅ Good |
| **Snapdragon 865** | Hexagon 698 | 15 TOPS | ⚠️ Limited |

### Compatible Devices

**Smartphones:**
- Samsung Galaxy S24 series (8 Gen 3)
- Samsung Galaxy S23 series (8 Gen 2)
- Xiaomi 14 series (8 Gen 3)
- OnePlus 12 (8 Gen 3)
- Google Pixel 8 Pro (Tensor G3)

**Development Boards:**
- Qualcomm RB3 Gen 2
- Qualcomm RB5
- DragonBoard 845c

**Edge Devices:**
- Qualcomm QCS/QCM series
- Industrial IoT devices

---

## 📋 Prerequisites

### 1. Development Environment

```bash
# Ubuntu 20.04 or 22.04 (recommended)
cat /etc/os-release

# Or use Android NDK for mobile deployment
```

### 2. Required Accounts

- **Qualcomm Developer Network**: Register at https://developer.qualcomm.com
- **QNN SDK Access**: Request access through developer portal

### 3. Hardware Requirements

**Development Machine:**
- OS: Ubuntu 20.04/22.04 or Windows with WSL
- RAM: 32GB+ recommended
- Storage: 50GB+ free space
- CPU: Modern x86-64 processor

**Target Device:**
- Qualcomm Snapdragon 865+ processor
- Android 10+ or Linux-based OS
- 8GB+ RAM for 7B model

---

## 📦 QNN SDK Setup

### Step 1: Download QNN SDK

```bash
# Login to Qualcomm Developer Network
# Navigate to: Software > Qualcomm Neural Processing SDK
# Download latest version (v2.10+)

# Example: qnn-v2.10.0.linux-x86_64.tar.gz
wget https://softwarecenter.qualcomm.com/api/download/software/.../qnn-v2.10.0.linux-x86_64.tar.gz
```

### Step 2: Extract and Configure

```bash
# Extract SDK
tar -xzf qnn-v2.10.0.linux-x86_64.tar.gz
cd qnn-v2.10.0

# Set environment variables
export QNN_SDK_ROOT=$(pwd)
export PATH=$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH
export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH

# Add to .bashrc for persistence
echo "export QNN_SDK_ROOT=$QNN_SDK_ROOT" >> ~/.bashrc
echo "export PATH=$QNN_SDK_ROOT/bin/x86_64-linux-clang:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=$QNN_SDK_ROOT/lib/x86_64-linux-clang:$LD_LIBRARY_PATH" >> ~/.bashrc
```

### Step 3: Verify Installation

```bash
# Check QNN tools
qnn-net-run --version
qnn-model-lib-generator --version

# Test with sample models
cd $QNN_SDK_ROOT/examples/QNN
./run_example.sh
```

### Step 4: Install Python Bindings

```bash
# Install QNN Python wrapper
cd $QNN_SDK_ROOT/lib/python
pip install qnn-*.whl

# Verify
python -c "import qnn; print('QNN SDK:', qnn.__version__)"
```

---

## 🏗️ Building ONNX Runtime with QNN

### Option 1: Pre-built Wheels (Recommended)

```bash
# Check if pre-built wheels are available
pip install onnxruntime-qnn

# Verify
python -c "import onnxruntime as ort; print('QNN' in ort.get_available_providers())"
```

### Option 2: Build from Source

```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-dev \
    python3-pip \
    libprotobuf-dev \
    protobuf-compiler

# Clone ONNX Runtime
git clone --recursive https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# Checkout stable release
git checkout v1.16.0
git submodule update --init --recursive

# Build with QNN support
./build.sh \
    --config Release \
    --build_shared_lib \
    --parallel $(nproc) \
    --use_qnn \
    --qnn_home $QNN_SDK_ROOT \
    --build_wheel \
    --skip_tests

# Install the wheel
pip install build/Linux/Release/dist/onnxruntime_qnn-*.whl
```

### Build Configuration Options

```bash
# For Android deployment
./build.sh \
    --config Release \
    --android \
    --android_sdk_path $ANDROID_SDK_ROOT \
    --android_ndk_path $ANDROID_NDK_ROOT \
    --android_abi arm64-v8a \
    --android_api 29 \
    --use_qnn \
    --qnn_home $QNN_SDK_ROOT \
    --build_shared_lib

# For optimized performance
./build.sh \
    --config Release \
    --use_qnn \
    --qnn_home $QNN_SDK_ROOT \
    --enable_pybind \
    --build_wheel \
    --parallel $(nproc) \
    --cmake_extra_defines \
        CMAKE_BUILD_TYPE=Release \
        onnxruntime_BUILD_UNIT_TESTS=OFF
```

---

## 🚀 Model Deployment

### Step 1: Prepare ONNX Model

```python
#!/usr/bin/env python3
"""
Prepare Shisa ONNX model for QNN deployment
"""

import onnx
from onnx import optimizer

# Load the ONNX model
model = onnx.load("models/Shisa_ONNX/model.onnx")

# Optimize for QNN
optimized_model = optimizer.optimize(model, [
    'eliminate_identity',
    'eliminate_nop_transpose',
    'eliminate_nop_pad',
    'extract_constant_to_initializer',
    'fuse_consecutive_transposes',
    'fuse_transpose_into_gemm'
])

# Save optimized model
onnx.save(optimized_model, "models/Shisa_ONNX/model_qnn_optimized.onnx")
print("✅ Model optimized for QNN")
```

### Step 2: Convert to QNN Format

```bash
# Convert ONNX to QNN DLC (Deep Learning Container)
qnn-onnx-converter \
    --input_network models/Shisa_ONNX/model.onnx \
    --output_path models/Shisa_QNN/model.dlc \
    --input_dim input_ids "1,1" \
    --quantization_overrides quantization_schema.json

# Quantize for HTP backend
qnn-model-lib-generator \
    -c models/Shisa_QNN/model.dlc \
    -b models/Shisa_QNN/model_quantized.dlc \
    --target hexagon
```

### Step 3: Create Inference Session

```python
#!/usr/bin/env python3
"""
Run inference with QNN execution provider
"""

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

# Configure QNN session options
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

# QNN provider configuration
qnn_options = {
    "backend_path": f"{os.environ['QNN_SDK_ROOT']}/lib/x86_64-linux-clang/libQnnHtp.so",
    "profiling_level": "basic",
    "rpc_control_latency": "low",
    "htp_performance_mode": "burst",
    "htp_graph_finalization_optimization_mode": "3",
    "soc_model": "sm8550",  # Snapdragon 8 Gen 2
}

# Create session
session = ort.InferenceSession(
    "models/Shisa_ONNX/model.onnx",
    sess_options=sess_options,
    providers=[
        ("QNNExecutionProvider", qnn_options),
        "CPUExecutionProvider"
    ]
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("models/Shisa_ONNX")

# Test inference
prompt = "こんにちは"
inputs = tokenizer(prompt, return_tensors="np")

# Run inference
outputs = session.run(None, dict(inputs))

print("✅ QNN inference successful!")
print(f"Output shape: {outputs[0].shape}")
```

### Step 4: Optimized Generation Loop

```python
#!/usr/bin/env python3
"""
Optimized text generation with QNN
"""

import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import time

class QNNGenerator:
    def __init__(self, model_path, qnn_options):
        self.session = ort.InferenceSession(
            model_path,
            providers=[("QNNExecutionProvider", qnn_options), "CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path.replace("model.onnx", ""))

    def generate(self, prompt, max_tokens=50):
        # Tokenize
        input_ids = self.tokenizer(prompt, return_tensors="np")["input_ids"]

        generated_tokens = []

        for _ in range(max_tokens):
            # Run inference
            outputs = self.session.run(None, {"input_ids": input_ids})

            # Get next token
            next_token = np.argmax(outputs[0][:, -1, :], axis=-1)
            generated_tokens.append(next_token[0])

            # Update input
            input_ids = np.concatenate([input_ids, next_token.reshape(1, 1)], axis=1)

            # Check for EOS
            if next_token[0] == self.tokenizer.eos_token_id:
                break

        return self.tokenizer.decode(generated_tokens)

# Usage
qnn_config = {
    "backend_path": "libQnnHtp.so",
    "htp_performance_mode": "burst"
}

generator = QNNGenerator("models/Shisa_ONNX/model.onnx", qnn_config)
result = generator.generate("Hello, ")
print(result)
```

---

## ⚡ Performance Optimization

### 1. HTP Performance Modes

```python
# Burst mode: Maximum performance, higher power
qnn_options["htp_performance_mode"] = "burst"

# Sustained high performance: Balanced
qnn_options["htp_performance_mode"] = "sustained_high_performance"

# Power saver: Lower power, reduced performance
qnn_options["htp_performance_mode"] = "power_saver"

# Default: Automatic balancing
qnn_options["htp_performance_mode"] = "default"
```

### 2. Graph Optimization

```python
# Maximum optimization
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Enable additional optimizations
sess_options.add_session_config_entry("session.disable_prepacking", "0")
sess_options.add_session_config_entry("session.use_device_allocator_for_initializers", "1")
```

### 3. Quantization Strategies

```bash
# INT8 quantization (recommended)
qnn-model-lib-generator \
    -c model.dlc \
    -b model_int8.dlc \
    --target hexagon \
    --quantization_overrides int8_config.json

# INT16 (higher accuracy, larger size)
qnn-model-lib-generator \
    -c model.dlc \
    -b model_int16.dlc \
    --target hexagon \
    --quantization_overrides int16_config.json
```

**Quantization Configuration (`int8_config.json`):**

```json
{
    "activation_encodings": {
        "bitwidth": 8,
        "dtype": "int"
    },
    "param_encodings": {
        "bitwidth": 8,
        "dtype": "int"
    },
    "algorithms": [
        "minmax",
        "mse"
    ]
}
```

### 4. Batching and Caching

```python
# Enable KV cache for faster autoregressive generation
class CachedQNNGenerator:
    def __init__(self, session):
        self.session = session
        self.cache = {}

    def generate_with_cache(self, input_ids, past_key_values=None):
        inputs = {"input_ids": input_ids}

        if past_key_values is not None:
            for i, (key, value) in enumerate(past_key_values):
                inputs[f"past_key_values.{i}.key"] = key
                inputs[f"past_key_values.{i}.value"] = value

        outputs = self.session.run(None, inputs)
        return outputs
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. QNN Provider Not Found

**Error**: `QNNExecutionProvider not in available providers`

**Solutions**:
```bash
# Verify QNN SDK
echo $QNN_SDK_ROOT
ls $QNN_SDK_ROOT/lib/x86_64-linux-clang/libQnnHtp.so

# Rebuild ONNX Runtime
./build.sh --use_qnn --qnn_home $QNN_SDK_ROOT

# Check provider availability
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

#### 2. Model Loading Fails on Device

**Error**: `Failed to load model on HTP backend`

**Solutions**:
- Check device NPU support: `adb shell cat /proc/cpuinfo | grep -i hexagon`
- Verify model is quantized for HTP
- Ensure sufficient memory available
- Try CPU fallback first

#### 3. Slow Inference on QNN

**Symptoms**: No speedup vs CPU

**Solutions**:
```python
# Enable profiling
qnn_options["profiling_level"] = "detailed"
qnn_options["enable_htp_fp16_precision"] = "1"

# Check execution provider
session.get_providers()  # Should show QNNExecutionProvider first
```

#### 4. Quantization Errors

**Error**: `Unsupported operator for quantization`

**Solutions**:
```bash
# Use mixed precision
--quantization_overrides mixed_precision_config.json

# Or skip problematic layers
--skip_quantization "layer_name_pattern"
```

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# QNN debug logs
os.environ['QNN_LOG_LEVEL'] = 'DEBUG'
os.environ['ONNXRUNTIME_LOG_LEVEL'] = '0'  # Verbose
```

---

## 🏭 Production Deployment

### Android Application

```kotlin
// Example Android integration
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession

class ShisaInference(private val context: Context) {
    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession

    init {
        val modelPath = copyModelToCache()
        val options = OrtSession.SessionOptions()

        // Configure QNN provider
        options.addQNN(
            "libQnnHtp.so",
            mapOf(
                "htp_performance_mode" to "burst",
                "profiling_level" to "basic"
            )
        )

        session = env.createSession(modelPath, options)
    }

    fun generate(prompt: String): String {
        // Tokenize
        val inputIds = tokenizer.encode(prompt)

        // Create tensors
        val inputTensor = OrtTensor.create(
            env,
            inputIds,
            longArrayOf(1, inputIds.size.toLong())
        )

        // Run inference
        val outputs = session.run(mapOf("input_ids" to inputTensor))

        // Decode
        return tokenizer.decode(outputs.values.first())
    }
}
```

### Edge Device Deployment

```python
#!/usr/bin/env python3
"""
Optimized deployment for edge devices
"""

import onnxruntime as ort
import numpy as np
from optimum.onnxruntime import ORTModelForCausalLM

class EdgeDeployment:
    def __init__(self, model_path):
        # Configure for edge device
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4  # Adjust for device
        sess_options.inter_op_num_threads = 1
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # QNN options for edge
        qnn_options = {
            "backend_path": "libQnnHtp.so",
            "htp_performance_mode": "sustained_high_performance",
            "rpc_control_latency": "low"
        }

        self.model = ORTModelForCausalLM.from_pretrained(
            model_path,
            provider="QNNExecutionProvider",
            provider_options=qnn_options,
            session_options=sess_options
        )

    def generate(self, prompt, max_tokens=50):
        # Optimized generation
        return self.model.generate(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            num_beams=1  # Greedy for speed
        )
```

---

## 📊 Performance Monitoring

```python
#!/usr/bin/env python3
"""
Monitor QNN performance metrics
"""

import onnxruntime as ort
import time
import psutil

def benchmark_qnn(session, inputs, num_runs=100):
    # Warm-up
    for _ in range(10):
        session.run(None, inputs)

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        session.run(None, inputs)
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_latency_ms": np.mean(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "throughput_qps": 1000 / np.mean(latencies)
    }

# Run benchmark
metrics = benchmark_qnn(session, inputs)
print(f"QNN Performance:")
print(f"  Mean latency: {metrics['mean_latency_ms']:.2f} ms")
print(f"  P95 latency: {metrics['p95_latency_ms']:.2f} ms")
print(f"  Throughput: {metrics['throughput_qps']:.2f} QPS")
```

---

## 🎯 Best Practices

1. **Always quantize models** for QNN deployment (INT8 recommended)
2. **Use burst mode** for interactive applications
3. **Enable KV caching** for autoregressive generation
4. **Profile regularly** to identify bottlenecks
5. **Test on target hardware** before production
6. **Monitor power consumption** on mobile devices
7. **Implement fallback** to CPU when QNN unavailable

---

## 📚 Additional Resources

- [Qualcomm QNN SDK Documentation](https://developer.qualcomm.com/qnn)
- [ONNX Runtime QNN Provider](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html)
- [Qualcomm AI Hub](https://aihub.qualcomm.com)
- [Snapdragon Neural Processing SDK](https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk)

---

**Last Updated**: 2025-09-30
**QNN SDK Version**: 2.10+
**ONNX Runtime Version**: 1.16+
**Status**: Production Ready ✅

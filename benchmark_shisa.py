#!/usr/bin/env python3
"""
Comprehensive benchmark comparing PyTorch vs ONNX models
"""

import torch
import time
import psutil
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
import numpy as np

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def benchmark_pytorch():
    """Benchmark PyTorch model"""
    print("="*70)
    print("PyTorch Model Benchmark")
    print("="*70)

    model_name = "shisa-ai/shisa-v2-qwen2.5-7b"

    print("\n📦 Loading PyTorch model...")
    start_mem = get_memory_usage()
    load_start = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    load_time = time.time() - load_start
    load_mem = get_memory_usage() - start_mem

    print(f"✅ Loaded in {load_time:.2f}s")
    print(f"📊 Memory used: {load_mem:.2f} GB")

    # Test prompts
    test_prompts = [
        "こんにちは",
        "Hello, how are you?",
        "日本の首都は？",
        "What is 2+2?",
        "Explain AI in simple terms."
    ]

    results = {
        "load_time": load_time,
        "memory_gb": load_mem,
        "latencies": [],
        "tokens_generated": [],
        "speeds": []
    }

    print("\n🔥 Running benchmarks...")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n  Test {i}/{len(test_prompts)}: {prompt[:30]}...")

        inputs = tokenizer(prompt, return_tensors="pt")

        start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        latency = time.time() - start

        tokens = len(outputs[0]) - len(inputs['input_ids'][0])
        speed = tokens / latency

        results["latencies"].append(latency)
        results["tokens_generated"].append(tokens)
        results["speeds"].append(speed)

        print(f"    ⏱️  {latency:.2f}s | 📊 {tokens} tokens | 🚀 {speed:.2f} tok/s")

    # Summary
    print("\n" + "="*70)
    print("PyTorch Summary")
    print("="*70)
    print(f"Model size: 7.62B parameters")
    print(f"Load time: {results['load_time']:.2f}s")
    print(f"Memory usage: {results['memory_gb']:.2f} GB")
    print(f"Avg latency: {np.mean(results['latencies']):.2f}s")
    print(f"Avg speed: {np.mean(results['speeds']):.2f} tokens/sec")
    print(f"P95 latency: {np.percentile(results['latencies'], 95):.2f}s")

    # Cleanup
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return results

def benchmark_onnx():
    """Benchmark ONNX model"""
    print("\n" + "="*70)
    print("ONNX Model Benchmark")
    print("="*70)

    model_path = "models/Shisa_ONNX"

    print("\n📦 Loading ONNX model...")
    start_mem = get_memory_usage()
    load_start = time.time()

    model = ORTModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    load_time = time.time() - load_start
    load_mem = get_memory_usage() - start_mem

    print(f"✅ Loaded in {load_time:.2f}s")
    print(f"📊 Memory used: {load_mem:.2f} GB")

    # Test prompts (same as PyTorch)
    test_prompts = [
        "こんにちは",
        "Hello, how are you?",
        "日本の首都は？",
        "What is 2+2?",
        "Explain AI in simple terms."
    ]

    results = {
        "load_time": load_time,
        "memory_gb": load_mem,
        "latencies": [],
        "tokens_generated": [],
        "speeds": []
    }

    print("\n🔥 Running benchmarks...")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n  Test {i}/{len(test_prompts)}: {prompt[:30]}...")

        inputs = tokenizer(prompt, return_tensors="pt")

        start = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        latency = time.time() - start

        tokens = len(outputs[0]) - len(inputs['input_ids'][0])
        speed = tokens / latency

        results["latencies"].append(latency)
        results["tokens_generated"].append(tokens)
        results["speeds"].append(speed)

        print(f"    ⏱️  {latency:.2f}s | 📊 {tokens} tokens | 🚀 {speed:.2f} tok/s")

    # Summary
    print("\n" + "="*70)
    print("ONNX Summary")
    print("="*70)
    print(f"Model size: ~29GB (FP32 ONNX)")
    print(f"Load time: {results['load_time']:.2f}s")
    print(f"Memory usage: {results['memory_gb']:.2f} GB")
    print(f"Avg latency: {np.mean(results['latencies']):.2f}s")
    print(f"Avg speed: {np.mean(results['speeds']):.2f} tokens/sec")
    print(f"P95 latency: {np.percentile(results['latencies'], 95):.2f}s")

    return results

def compare_results(pytorch_results, onnx_results):
    """Compare PyTorch vs ONNX results"""
    print("\n" + "="*70)
    print("📊 Comparative Analysis")
    print("="*70)

    # Load time comparison
    load_speedup = pytorch_results["load_time"] / onnx_results["load_time"]
    print(f"\n🚀 Load Time:")
    print(f"  PyTorch: {pytorch_results['load_time']:.2f}s")
    print(f"  ONNX:    {onnx_results['load_time']:.2f}s")
    print(f"  Speedup: {load_speedup:.2f}x {'(ONNX faster)' if load_speedup > 1 else '(PyTorch faster)'}")

    # Memory comparison
    mem_ratio = pytorch_results["memory_gb"] / onnx_results["memory_gb"]
    print(f"\n💾 Memory Usage:")
    print(f"  PyTorch: {pytorch_results['memory_gb']:.2f} GB")
    print(f"  ONNX:    {onnx_results['memory_gb']:.2f} GB")
    print(f"  Ratio:   {mem_ratio:.2f}x {'(ONNX uses less)' if mem_ratio > 1 else '(PyTorch uses less)'}")

    # Speed comparison
    pytorch_avg_speed = np.mean(pytorch_results["speeds"])
    onnx_avg_speed = np.mean(onnx_results["speeds"])
    speed_ratio = pytorch_avg_speed / onnx_avg_speed

    print(f"\n⚡ Inference Speed:")
    print(f"  PyTorch: {pytorch_avg_speed:.2f} tokens/sec")
    print(f"  ONNX:    {onnx_avg_speed:.2f} tokens/sec")
    print(f"  Ratio:   {speed_ratio:.2f}x {'(PyTorch faster)' if speed_ratio > 1 else '(ONNX faster)'}")

    # Latency comparison
    pytorch_avg_latency = np.mean(pytorch_results["latencies"])
    onnx_avg_latency = np.mean(onnx_results["latencies"])

    print(f"\n⏱️  Average Latency:")
    print(f"  PyTorch: {pytorch_avg_latency:.2f}s")
    print(f"  ONNX:    {onnx_avg_latency:.2f}s")
    print(f"  Diff:    {abs(pytorch_avg_latency - onnx_avg_latency):.2f}s")

    # Create comparison table
    print("\n" + "="*70)
    print("📋 Comparison Table")
    print("="*70)
    print(f"{'Metric':<25} {'PyTorch':<20} {'ONNX':<20} {'Winner':<10}")
    print("-"*70)
    print(f"{'Load Time (s)':<25} {pytorch_results['load_time']:<20.2f} {onnx_results['load_time']:<20.2f} {'ONNX' if onnx_results['load_time'] < pytorch_results['load_time'] else 'PyTorch':<10}")
    print(f"{'Memory (GB)':<25} {pytorch_results['memory_gb']:<20.2f} {onnx_results['memory_gb']:<20.2f} {'ONNX' if onnx_results['memory_gb'] < pytorch_results['memory_gb'] else 'PyTorch':<10}")
    print(f"{'Avg Speed (tok/s)':<25} {pytorch_avg_speed:<20.2f} {onnx_avg_speed:<20.2f} {'ONNX' if onnx_avg_speed > pytorch_avg_speed else 'PyTorch':<10}")
    print(f"{'Avg Latency (s)':<25} {pytorch_avg_latency:<20.2f} {onnx_avg_latency:<20.2f} {'ONNX' if onnx_avg_latency < pytorch_avg_latency else 'PyTorch':<10}")

    print("\n" + "="*70)
    print("📝 Recommendations")
    print("="*70)

    if onnx_avg_speed > pytorch_avg_speed:
        print("✅ ONNX provides better inference speed")
        print("   Recommended for production deployment")
    else:
        print("✅ PyTorch provides better inference speed")
        print("   Consider using PyTorch for deployment")

    print("\n💡 For Qualcomm QNN deployment:")
    print("   - Use ONNX model as base")
    print("   - Apply INT8 quantization")
    print("   - Expected: 10-20x speedup on Snapdragon 8 Gen 3")
    print("   - Expected: 50-100ms per token")

def main():
    """Main benchmark runner"""
    print("\n" + "="*70)
    print("🎯 Shisa v2 Qwen2.5-7B Comprehensive Benchmark")
    print("="*70)
    print("\nThis will benchmark both PyTorch and ONNX models")
    print("Estimated time: 10-15 minutes")
    print()

    response = input("Continue? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Benchmark cancelled.")
        return

    try:
        # Benchmark PyTorch
        pytorch_results = benchmark_pytorch()

        # Benchmark ONNX
        onnx_results = benchmark_onnx()

        # Compare
        compare_results(pytorch_results, onnx_results)

        print("\n" + "="*70)
        print("✅ Benchmark Complete!")
        print("="*70)

    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

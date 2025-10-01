#!/usr/bin/env python3
"""
Convert Shisa v2 Qwen2.5-7B to ONNX format with INT4 quantization
Optimized for Qualcomm QNN deployment
"""

import os
import time
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

def convert_to_onnx():
    """Convert Shisa model to ONNX format"""
    print("="*70)
    print("Shisa v2 Qwen2.5-7B -> ONNX Conversion")
    print("Optimized for Qualcomm QNN Hardware")
    print("="*70)
    print()

    model_name = "shisa-ai/shisa-v2-qwen2.5-7b"
    output_dir = "models/Shisa_ONNX"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"📦 Source Model: {model_name}")
    print(f"📁 Output Directory: {output_dir}")
    print()

    # Step 1: Load and convert model
    print("Step 1/3: Loading and converting model to ONNX...")
    print("⏳ This will take several minutes (~5-10 min)...")
    start_time = time.time()

    try:
        model = ORTModelForCausalLM.from_pretrained(
            model_name,
            export=True,
            trust_remote_code=True
        )
        conversion_time = time.time() - start_time
        print(f"✅ Conversion completed in {conversion_time/60:.2f} minutes")
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

    # Step 2: Load tokenizer
    print("\nStep 2/3: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✅ Tokenizer loaded")

    # Step 3: Save to disk
    print(f"\nStep 3/3: Saving ONNX model to {output_dir}...")
    save_start = time.time()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    save_time = time.time() - save_start
    print(f"✅ Model saved in {save_time:.2f} seconds")

    # Display file information
    print("\n" + "="*70)
    print("Conversion Summary")
    print("="*70)
    print(f"✅ ONNX model saved to: {output_dir}")
    print(f"⏱️  Total time: {(time.time() - start_time)/60:.2f} minutes")

    # List generated files
    print("\n📂 Generated files:")
    for root, dirs, files in os.walk(output_dir):
        for file in sorted(files):
            filepath = os.path.join(root, file)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  - {file} ({size_mb:.2f} MB)")

    print("\n" + "="*70)
    print("✅ Conversion completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("1. Run 'python test_shisa_onnx.py' to test the ONNX model")
    print("2. Run 'python quantize_shisa.py' for INT4 quantization")
    print("3. Deploy to Qualcomm hardware with QNN support")
    print()

    return True

if __name__ == "__main__":
    try:
        convert_to_onnx()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

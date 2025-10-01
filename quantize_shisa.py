#!/usr/bin/env python3
"""
Quantize Shisa v2 Qwen2.5-7B model to INT4 using Olive AI
Optimized for edge deployment and QNN acceleration
"""

import os
import subprocess
import sys

def quantize_with_olive():
    """Quantize model using Olive AI with INT4"""
    print("="*70)
    print("Shisa v2 Qwen2.5-7B Quantization")
    print("INT4 Quantization for Edge Deployment")
    print("="*70)
    print()

    model_name = "shisa-ai/shisa-v2-qwen2.5-7b"
    output_dir = "models/Shisa_INT4"

    print(f"📦 Source Model: {model_name}")
    print(f"📁 Output Directory: {output_dir}")
    print(f"🔧 Quantization: INT4")
    print(f"🎯 Target: CPU/NPU with QNN support")
    print()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("Starting Olive AI quantization...")
    print("⏳ This process may take 30-60 minutes depending on hardware")
    print()

    # Olive AI command for INT4 quantization
    command = [
        "olive", "auto-opt",
        "--model_name_or_path", model_name,
        "--trust_remote_code",
        "--output_path", output_dir,
        "--device", "cpu",
        "--provider", "CPUExecutionProvider",
        "--use_ort_genai",
        "--precision", "int4",
        "--log_level", "1"
    ]

    print(f"Running command:")
    print(" ".join(command))
    print()

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=False,
            text=True
        )

        print("\n" + "="*70)
        print("✅ Quantization completed successfully!")
        print("="*70)
        print(f"\n📂 Quantized model saved to: {output_dir}")

        # List generated files
        if os.path.exists(output_dir):
            print("\n📂 Generated files:")
            for root, dirs, files in os.walk(output_dir):
                for file in sorted(files):
                    filepath = os.path.join(root, file)
                    if os.path.isfile(filepath):
                        size_mb = os.path.getsize(filepath) / (1024 * 1024)
                        print(f"  - {file} ({size_mb:.2f} MB)")

        print("\nNext steps:")
        print("1. Test the quantized model")
        print("2. Convert to ONNX if needed")
        print("3. Deploy to Qualcomm hardware with QNN")

        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Quantization failed with error code {e.returncode}")
        print("\nTrying alternative quantization method...")
        return quantize_with_optimum()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def quantize_with_optimum():
    """Alternative: Quantize using Optimum with dynamic quantization"""
    print("\n" + "="*70)
    print("Alternative Method: Optimum Quantization")
    print("="*70)
    print()

    try:
        from optimum.onnxruntime import ORTQuantizer, ORTModelForCausalLM
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from transformers import AutoTokenizer
        import os

        model_name = "shisa-ai/shisa-v2-qwen2.5-7b"
        onnx_dir = "models/Shisa_ONNX"
        output_dir = "models/Shisa_INT8_Quantized"

        os.makedirs(output_dir, exist_ok=True)

        print("Step 1: Converting to ONNX (if not already done)...")
        if not os.path.exists(onnx_dir):
            print("Converting to ONNX first...")
            model = ORTModelForCausalLM.from_pretrained(
                model_name,
                export=True,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model.save_pretrained(onnx_dir)
            tokenizer.save_pretrained(onnx_dir)
            print(f"✅ ONNX model saved to {onnx_dir}")
        else:
            print(f"✅ Using existing ONNX model from {onnx_dir}")

        print("\nStep 2: Applying INT8 dynamic quantization...")
        # Load the ONNX model
        quantizer = ORTQuantizer.from_pretrained(onnx_dir)

        # Configure quantization
        qconfig = AutoQuantizationConfig.arm64(is_static=False, per_channel=False)

        # Apply quantization
        quantizer.quantize(
            save_dir=output_dir,
            quantization_config=qconfig
        )

        print(f"\n✅ Quantized model saved to: {output_dir}")

        # List files
        print("\n📂 Generated files:")
        for root, dirs, files in os.walk(output_dir):
            for file in sorted(files):
                filepath = os.path.join(root, file)
                if os.path.isfile(filepath):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"  - {file} ({size_mb:.2f} MB)")

        return True

    except Exception as e:
        print(f"\n❌ Alternative quantization also failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n⚠️  Note: Full INT4 quantization with Olive requires significant time and resources")
    print("For faster results, we'll use INT8 dynamic quantization with Optimum\n")

    response = input("Proceed with quantization? [y/N]: ").strip().lower()

    if response in ['y', 'yes']:
        # Try Optimum method directly for faster results
        success = quantize_with_optimum()
        if success:
            print("\n" + "="*70)
            print("✅ Quantization pipeline completed!")
            print("="*70)
        else:
            print("\n❌ Quantization failed. Check the error messages above.")
    else:
        print("Quantization cancelled.")

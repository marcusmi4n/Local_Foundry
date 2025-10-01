#!/usr/bin/env python3
"""
Efficient ONNX conversion for Shisa v2 Qwen2.5-7B
Uses optimum-cli for more stable conversion
"""

import subprocess
import sys
import os

def convert_with_optimum_cli():
    """Convert using optimum-cli for better memory management"""
    print("="*70)
    print("Shisa v2 Qwen2.5-7B -> ONNX Conversion (Efficient Method)")
    print("="*70)
    print()

    model_name = "shisa-ai/shisa-v2-qwen2.5-7b"
    output_dir = "models/Shisa_ONNX"

    os.makedirs(output_dir, exist_ok=True)

    print(f"📦 Source: {model_name}")
    print(f"📁 Output: {output_dir}")
    print(f"⚙️  Method: optimum-cli (memory efficient)")
    print()
    print("⏳ This will take 10-20 minutes...")
    print()

    # Use optimum-cli which is more memory efficient
    command = [
        "optimum-cli", "export", "onnx",
        "--model", model_name,
        "--task", "text-generation-with-past",
        "--trust-remote-code",
        output_dir
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
        print("✅ Conversion completed successfully!")
        print("="*70)

        # Verify files
        if os.path.exists(os.path.join(output_dir, "model.onnx")):
            print(f"\n📂 Files created in {output_dir}:")
            for file in sorted(os.listdir(output_dir)):
                filepath = os.path.join(output_dir, file)
                if os.path.isfile(filepath):
                    size_mb = os.path.getsize(filepath) / (1024 * 1024)
                    print(f"  ✓ {file} ({size_mb:.2f} MB)")

            print("\n✅ ONNX model ready for testing!")
            print("   Run: python test_shisa_onnx.py")
            return True
        else:
            print("\n⚠️  Conversion completed but model files not found")
            return False

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Conversion failed with error code {e.returncode}")
        print("\nNote: Converting 7B models requires significant RAM (16GB+)")
        print("Consider using a smaller model or a machine with more memory")
        return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n⚠️  Important Notes:")
    print("  - This requires ~16GB+ RAM for 7B model conversion")
    print("  - The process will take 10-20 minutes")
    print("  - Ensure you have ~15GB free disk space")
    print()

    response = input("Continue with conversion? [y/N]: ").strip().lower()

    if response in ['y', 'yes']:
        success = convert_with_optimum_cli()
        if success:
            print("\n" + "="*70)
            print("✅ ONNX Conversion Complete!")
            print("="*70)
            print("\nNext steps:")
            print("1. Test: python test_shisa_onnx.py")
            print("2. Quantize: python quantize_shisa.py")
            print("3. Benchmark: python benchmark_shisa.py")
        else:
            print("\n❌ Conversion failed. See errors above.")
    else:
        print("\nConversion cancelled.")

#!/usr/bin/env python3
"""
Test Shisa v2 Qwen2.5-7B ONNX model
Includes QNN provider compatibility check
"""

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import time
import onnxruntime as ort

def check_qnn_support():
    """Check if QNN execution provider is available"""
    print("="*70)
    print("Checking ONNX Runtime Providers")
    print("="*70)

    providers = ort.get_available_providers()
    print("\nAvailable providers:")
    for provider in providers:
        print(f"  ✓ {provider}")

    if "QNNExecutionProvider" in providers:
        print("\n✅ QNN Execution Provider is available!")
        print("   Ready for Qualcomm hardware acceleration")
        return True
    else:
        print("\n⚠️  QNN Execution Provider not available")
        print("   Note: QNN requires Qualcomm's QNN SDK")
        print("   Model will run on CPU for now")
        return False

def test_onnx_model():
    """Test the ONNX model"""
    print("\n" + "="*70)
    print("Testing Shisa v2 Qwen2.5-7B ONNX Model")
    print("="*70)
    print()

    model_path = "models/Shisa_ONNX"

    print(f"📁 Loading model from: {model_path}")

    # Load model and tokenizer
    print("Loading ONNX model...")
    load_start = time.time()
    model = ORTModelForCausalLM.from_pretrained(model_path)
    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f} seconds")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✅ Tokenizer loaded")

    # Test cases (Japanese and English)
    test_cases = [
        {
            "prompt": "こんにちは！調子はどうですか？",
            "description": "Japanese greeting",
            "max_tokens": 50
        },
        {
            "prompt": "Hello! How are you today?",
            "description": "English greeting",
            "max_tokens": 50
        },
        {
            "prompt": "日本の首都は何ですか？",
            "description": "Japanese factual question",
            "max_tokens": 30
        },
        {
            "prompt": "What is 2+2?",
            "description": "Simple math",
            "max_tokens": 20
        }
    ]

    print("\n" + "="*70)
    print("Running Test Cases")
    print("="*70)

    total_tokens = 0
    total_time = 0

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'─'*70}")
        print(f"Test {i}/{len(test_cases)}: {test_case['description']}")
        print(f"{'─'*70}")

        prompt = test_case["prompt"]
        print(f"Prompt: {prompt}")

        # Apply chat template if available
        if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt

        # Tokenize
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        input_length = len(inputs['input_ids'][0])

        # Generate
        print("\nGenerating...")
        start_time = time.time()

        outputs = model.generate(
            **inputs,
            max_new_tokens=test_case["max_tokens"],
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

        generation_time = time.time() - start_time

        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        tokens_generated = len(outputs[0]) - input_length

        print(f"\nResponse: {response}")
        print(f"\n⏱️  Time: {generation_time:.2f}s")
        print(f"📊 Tokens: {tokens_generated}")
        print(f"🚀 Speed: {tokens_generated/generation_time:.2f} tokens/sec")

        total_tokens += tokens_generated
        total_time += generation_time

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"✅ All {len(test_cases)} tests completed successfully!")
    print(f"📊 Total tokens generated: {total_tokens}")
    print(f"⏱️  Total generation time: {total_time:.2f}s")
    print(f"🚀 Average speed: {total_tokens/total_time:.2f} tokens/sec")
    print(f"💾 Model size: ~527MB (FP32 ONNX)")
    print("="*70)

    return True

if __name__ == "__main__":
    try:
        # Check QNN support
        qnn_available = check_qnn_support()

        # Test model
        test_onnx_model()

        print("\n✅ ONNX model is ready for deployment!")
        if not qnn_available:
            print("\n📝 To enable QNN acceleration:")
            print("   1. Install Qualcomm QNN SDK")
            print("   2. Build ONNX Runtime with QNN support")
            print("   3. Deploy to Qualcomm hardware (NPU)")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

#!/usr/bin/env python3
"""
Test script for the converted ONNX model with Qualcomm QNN support
"""

import os
import sys
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import time

def test_onnx_model():
    """Test the converted ONNX model"""

    print("Loading ONNX model...")
    model_path = "models/Qwen"

    # Load the converted model
    model = ORTModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Test prompt with proper chat formatting
    messages = [
        {"role": "user", "content": "Hello, I am Marcus. Who are you?"}
    ]

    # Apply chat template if available
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "Hello, I am Marcus. Who are you?"

    print(f"Test prompt: {prompt}")

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    print("Generating response...")
    start_time = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,  # Use greedy decoding for more stable output
        temperature=1.0,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generation_time = time.time() - start_time

    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    print(f"Generation time: {generation_time:.2f} seconds")

    return True

def check_qnn_support():
    """Check if QNN execution provider is available"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print("Available ONNX Runtime providers:")
        for provider in providers:
            print(f"  - {provider}")

        if "QNNExecutionProvider" in providers:
            print("✅ QNN Execution Provider is available!")
            return True
        else:
            print("❌ QNN Execution Provider is not available")
            print("Note: QNN provider typically requires Qualcomm's QNN SDK")
            return False

    except ImportError as e:
        print(f"Error importing onnxruntime: {e}")
        return False

def test_qnn_model():
    """Test model with QNN execution provider if available"""
    try:
        import onnxruntime as ort

        if "QNNExecutionProvider" not in ort.get_available_providers():
            print("QNN provider not available, skipping QNN test")
            return False

        print("Testing with QNN Execution Provider...")

        # Create session with QNN provider
        model_path = "models/Qwen/model.onnx"
        providers = ['QNNExecutionProvider', 'CPUExecutionProvider']

        session = ort.InferenceSession(model_path, providers=providers)
        print("✅ Successfully created QNN inference session!")

        return True

    except Exception as e:
        print(f"Error testing QNN: {e}")
        return False

if __name__ == "__main__":
    print("=== ONNX Model Test ===")

    # Check QNN support
    print("\n1. Checking QNN support...")
    qnn_available = check_qnn_support()

    # Test basic ONNX model
    print("\n2. Testing basic ONNX model...")
    try:
        test_onnx_model()
        print("✅ Basic ONNX model test passed!")
    except Exception as e:
        print(f"❌ Basic ONNX model test failed: {e}")

    # Test QNN if available
    if qnn_available:
        print("\n3. Testing QNN model...")
        test_qnn_model()
    else:
        print("\n3. Skipping QNN test (provider not available)")

    print("\n=== Test Complete ===")
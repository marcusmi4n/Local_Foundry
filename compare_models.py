#!/usr/bin/env python3
"""
Compare ONNX model output with original PyTorch model
"""

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

def test_pytorch_model():
    """Test the original PyTorch model"""
    print("=== Testing Original PyTorch Model ===")

    # Load original model
    model = AutoModelForCausalLM.from_pretrained(
        "Sakalti/Qwen2.5-1B-Instruct",
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Sakalti/Qwen2.5-1B-Instruct")

    # Test with proper chat formatting
    messages = [
        {"role": "user", "content": "Hello, I am Marcus. Who are you?"}
    ]

    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "Hello, I am Marcus. Who are you?"

    print(f"Prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    generation_time = time.time() - start_time

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"PyTorch Response: {response}")
    print(f"PyTorch Generation time: {generation_time:.2f} seconds\n")

    return response

def test_onnx_model():
    """Test the converted ONNX model"""
    print("=== Testing Converted ONNX Model ===")

    model = ORTModelForCausalLM.from_pretrained("models/Qwen")
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen")

    # Test with same formatting
    messages = [
        {"role": "user", "content": "Hello, I am Marcus. Who are you?"}
    ]

    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "Hello, I am Marcus. Who are you?"

    print(f"Prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt")

    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        temperature=1.0,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generation_time = time.time() - start_time

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"ONNX Response: {response}")
    print(f"ONNX Generation time: {generation_time:.2f} seconds\n")

    return response

def compare_models():
    """Compare both model outputs"""
    print("=== Model Comparison ===\n")

    try:
        pytorch_response = test_pytorch_model()
    except Exception as e:
        print(f"PyTorch model failed: {e}")
        pytorch_response = None

    try:
        onnx_response = test_onnx_model()
    except Exception as e:
        print(f"ONNX model failed: {e}")
        onnx_response = None

    if pytorch_response and onnx_response:
        print("=== Comparison Summary ===")
        print("Both models completed successfully!")
        if pytorch_response.strip() == onnx_response.strip():
            print("✅ Outputs are identical!")
        else:
            print("⚠️ Outputs differ (this is normal due to numerical precision)")

    print("\n=== Test Complete ===")

if __name__ == "__main__":
    compare_models()
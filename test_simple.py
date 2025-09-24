#!/usr/bin/env python3
"""
Simple test with different generation parameters
"""

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import time

def test_simple_generation():
    """Test with simple prompts and different parameters"""

    model = ORTModelForCausalLM.from_pretrained("models/Qwen")
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen")

    # Test different prompts and parameters
    test_cases = [
        {
            "prompt": "The capital of France is",
            "params": {"max_new_tokens": 10, "do_sample": False}
        },
        {
            "prompt": "2 + 2 =",
            "params": {"max_new_tokens": 5, "do_sample": False}
        },
        {
            "prompt": "Write a simple greeting:",
            "params": {"max_new_tokens": 15, "do_sample": True, "temperature": 0.7, "top_p": 0.9}
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n=== Test {i} ===")
        prompt = test_case["prompt"]
        params = test_case["params"]

        print(f"Prompt: {prompt}")

        inputs = tokenizer(prompt, return_tensors="pt")

        start_time = time.time()
        outputs = model.generate(
            **inputs,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **params
        )
        generation_time = time.time() - start_time

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response: {response}")
        print(f"Time: {generation_time:.2f}s")

if __name__ == "__main__":
    test_simple_generation()
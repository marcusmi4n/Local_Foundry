#!/usr/bin/env python3
"""
Test Shisa v2 Qwen2.5-7B model with PyTorch
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

def test_shisa_pytorch():
    """Test the Shisa model in PyTorch format"""
    print("=== Testing Shisa v2 Qwen2.5-7B (PyTorch) ===\n")

    model_name = "shisa-ai/shisa-v2-qwen2.5-7b"

    print(f"Loading model: {model_name}")
    print("This will download ~14GB of model files on first run...\n")

    # Load model and tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print("Loading model (this may take a few minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Use FP32 for CPU
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    print(f"✅ Model loaded successfully!")
    print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters\n")

    # Test prompts (Japanese and English since Shisa is bilingual)
    test_cases = [
        {
            "prompt": "こんにちは。あなたは誰ですか？",
            "description": "Japanese greeting"
        },
        {
            "prompt": "Hello, who are you?",
            "description": "English greeting"
        },
        {
            "prompt": "東京の有名な観光地を3つ教えてください。",
            "description": "Japanese question about Tokyo"
        },
        {
            "prompt": "Explain quantum computing in simple terms.",
            "description": "English technical question"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"Test {i}/{len(test_cases)}: {test_case['description']}")
        print(f"{'='*60}")

        prompt = test_case["prompt"]
        print(f"Prompt: {prompt}\n")

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

        # Generate
        print("Generating response...")
        start_time = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
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
        print(f"Response: {response}")
        print(f"\n⏱️  Generation time: {generation_time:.2f} seconds")
        print(f"📊 Tokens generated: {len(outputs[0]) - len(inputs['input_ids'][0])}")
        print(f"🚀 Speed: {(len(outputs[0]) - len(inputs['input_ids'][0])) / generation_time:.2f} tokens/sec\n")

    print("="*60)
    print("✅ All tests completed successfully!")
    print("="*60)

    return True

if __name__ == "__main__":
    try:
        test_shisa_pytorch()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

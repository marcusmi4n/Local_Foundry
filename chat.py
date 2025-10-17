#!/usr/bin/env python3
"""
Interactive chat with Shisa v2 Qwen2.5-7B ONNX model
"""

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import time

def chat_with_model():
    """Interactive chat with the ONNX model"""
    print("\n" + "="*70)
    print("Interactive Chat with Shisa v2 Qwen2.5-7B ONNX Model")
    print("="*70)
    print("Type 'exit' or 'quit' to end the session.")
    print()

    model_path = "marcusmi4n/shisa-v2-qwen2.5-7b-onnx"

    print(f"📁 Loading model from: {model_path}")
    print("⏳ This may take a few minutes for the first download...")

    # Load model and tokenizer
    print("Loading ONNX model...")
    load_start = time.time()
    model = ORTModelForCausalLM.from_pretrained(model_path)
    load_time = time.time() - load_start
    print(f"✅ Model loaded in {load_time:.2f} seconds")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("✅ Tokenizer loaded")
    print("\n" + "="*70)
    print("Chat session started.")
    print("="*70)


    while True:
        prompt = input("You: ")
        if prompt.lower() in ["exit", "quit"]:
            print("Ending chat session.")
            break

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
            max_new_tokens=150,
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

        # The response includes the prompt, so we need to remove it.
        # This is a simple way to do it, but might not be perfect for all cases.
        if response.startswith(formatted_prompt):
             response = response[len(formatted_prompt):]
        elif response.startswith(prompt):
            response = response[len(prompt):]


        print(f"Model: {response.strip()}")
        print(f"\n⏱️  Time: {generation_time:.2f}s, 🚀 Speed: {tokens_generated/generation_time:.2f} tokens/sec")
        print("-" * 70)


if __name__ == "__main__":
    try:
        chat_with_model()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

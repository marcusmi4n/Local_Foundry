#!/usr/bin/env python3
"""
Test DialoGPT-medium ONNX model
"""

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import time

def test_dialogpt_model():
    """Test the DialoGPT ONNX model"""
    print("=== Testing DialoGPT-medium ONNX Model ===")

    # Load the converted model
    model = ORTModelForCausalLM.from_pretrained("models/DialoGPT")
    tokenizer = AutoTokenizer.from_pretrained("models/DialoGPT")

    # DialoGPT specific test cases
    test_cases = [
        "Hello, how are you?",
        "What's your favorite color?",
        "Tell me a joke",
        "How's the weather today?",
        "What do you think about AI?"
    ]

    for i, prompt in enumerate(test_cases, 1):
        print(f"\n=== Test {i} ===")
        print(f"Human: {prompt}")

        # Encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt')

        start_time = time.time()

        # Generate a response while limiting the total chat history
        chat_history_ids = model.generate(
            new_user_input_ids,
            max_length=new_user_input_ids.shape[-1] + 50,
            num_beams=3,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id
        )

        generation_time = time.time() - start_time

        # Extract the bot response (remove the user input)
        bot_response = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

        print(f"Bot: {bot_response}")
        print(f"Time: {generation_time:.2f}s")

if __name__ == "__main__":
    test_dialogpt_model()
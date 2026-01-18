#!/usr/bin/env python3
"""
Test script to verify GPT-2 pretrained weight loading.

This script:
1. Loads our GPT-2 implementation with HuggingFace pretrained weights
2. Loads the original HuggingFace GPT-2 model
3. Compares outputs to verify correctness
4. Demonstrates text generation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import tiktoken
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.model import GPT, GPT2Config


def test_output_equivalence():
    """Test that our model produces identical outputs to HuggingFace."""
    print("=" * 70)
    print("Test 1: Output Equivalence")
    print("=" * 70)
    
    # Load both models
    print("\nLoading our GPT-2 implementation...")
    our_model = GPT.from_pretrained('gpt2')
    our_model.eval()
    
    print("\nLoading HuggingFace GPT-2...")
    hf_model = GPT2LMHeadModel.from_pretrained('gpt2')
    hf_model.eval()
    
    # Test input
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text = "Hello, I'm a language model"
    input_ids = tokenizer.encode(text, return_tensors='pt')
    
    print(f"\nTest input: '{text}'")
    print(f"Token IDs: {input_ids.tolist()}")
    
    # Forward pass
    with torch.no_grad():
        our_logits, _ = our_model(input_ids)
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits
    
    # Compare outputs
    max_diff = (our_logits - hf_logits).abs().max().item()
    mean_diff = (our_logits - hf_logits).abs().mean().item()
    
    print(f"\nOutput comparison:")
    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Mean absolute difference: {mean_diff:.2e}")
    
    # Numerical tolerance (should be very small, ~1e-5 for float32)
    if max_diff < 1e-4:
        print("\n✓ PASSED: Outputs match within tolerance!")
        return True
    else:
        print("\n✗ FAILED: Outputs differ significantly!")
        return False


def test_generation():
    """Test text generation with our model."""
    print("\n" + "=" * 70)
    print("Test 2: Text Generation")
    print("=" * 70)
    
    # Load model
    model = GPT.from_pretrained('gpt2')
    model.eval()
    
    # Use tiktoken for encoding (faster than HuggingFace tokenizer)
    enc = tiktoken.get_encoding('gpt2')
    
    # Test prompts
    prompts = [
        "Hello, I'm a language model",
        "The meaning of life is",
        "In a galaxy far, far away",
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        
        # Encode
        input_ids = torch.tensor([enc.encode(prompt)])
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=30,
                temperature=0.8,
                top_k=40,
            )
        
        # Decode
        generated_text = enc.decode(output_ids[0].tolist())
        print(f"Generated: '{generated_text}'")
    
    print("\n✓ Generation test completed!")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("GPT-2 Pretrained Weight Loading Test")
    print("=" * 70 + "\n")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU (CUDA not available)")
    
    # Run tests
    results = []
    
    try:
        results.append(("Output Equivalence", test_output_equivalence()))
    except Exception as e:
        print(f"\n✗ Output Equivalence test failed with error: {e}")
        results.append(("Output Equivalence", False))
    
    try:
        results.append(("Text Generation", test_generation()))
    except Exception as e:
        print(f"\n✗ Text Generation test failed with error: {e}")
        results.append(("Text Generation", False))
    
    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 All tests passed! Weight loading is correct.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


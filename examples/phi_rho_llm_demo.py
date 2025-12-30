"""
Demo script: Validate Phi-Rho quantization with a real LLM.

This script:
1. Loads a small LLM (TinyLlama)
2. Replaces all Linear layers with PhiRhoLinear
3. Runs a simple generation test
4. Compares output quality vs original model
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import Phi-Rho components
from torchao.quantization.prototype.phi_rho import PhiRhoCodebookConfig
from torchao.quantization.prototype.phi_rho.phi_rho_linear import PhiRhoLinear


def replace_linear_with_phi_rho(model, config=None):
    """
    Recursively replace all nn.Linear layers with PhiRhoLinear.
    
    Returns the number of layers replaced.
    """
    if config is None:
        config = PhiRhoCodebookConfig(num_codebook_entries=256)
    
    count = 0
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # Replace with PhiRhoLinear
            phi_linear = PhiRhoLinear.from_float(module, config)
            setattr(model, name, phi_linear)
            count += 1
        else:
            # Recurse into children
            count += replace_linear_with_phi_rho(module, config)
    
    return count


def get_memory_mb(model):
    """Estimate model memory usage in MB."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total / 1e6


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device.upper()}")
    
    # Use a small model for testing
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load original model
    model_orig = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model_orig.eval()
    
    # Measure original memory
    orig_memory = get_memory_mb(model_orig)
    print(f"Original model memory: {orig_memory:.1f} MB")
    
    # Create a quantized copy
    print("Creating Phi-Rho quantized copy...")
    model_quant = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model_quant.eval()
    
    # Replace Linear layers
    config = PhiRhoCodebookConfig(num_codebook_entries=256)
    num_replaced = replace_linear_with_phi_rho(model_quant, config)
    print(f"Replaced {num_replaced} Linear layers with PhiRhoLinear")
    
    # Measure quantized memory
    quant_memory = get_memory_mb(model_quant)
    print(f"Quantized model memory: {quant_memory:.1f} MB")
    print(f"Memory reduction: {(1 - quant_memory/orig_memory)*100:.1f}%")
    
    # Test generation
    prompt = "The quick brown fox"
    print(f"\nPrompt: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Original model generation
    print("\n--- Original Model ---")
    t0 = time.time()
    with torch.no_grad():
        outputs_orig = model_orig.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    t_orig = time.time() - t0
    text_orig = tokenizer.decode(outputs_orig[0], skip_special_tokens=True)
    print(f"Output: {text_orig}")
    print(f"Time: {t_orig:.2f}s")
    
    # Quantized model generation
    print("\n--- Phi-Rho Quantized Model ---")
    t0 = time.time()
    with torch.no_grad():
        outputs_quant = model_quant.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    t_quant = time.time() - t0
    text_quant = tokenizer.decode(outputs_quant[0], skip_special_tokens=True)
    print(f"Output: {text_quant}")
    print(f"Time: {t_quant:.2f}s")
    
    # Compare
    print("\n--- Comparison ---")
    if text_orig == text_quant:
        print("✅ Outputs are IDENTICAL!")
    else:
        print("⚠️  Outputs differ (expected with quantization)")
        # Show first difference
        for i, (c1, c2) in enumerate(zip(text_orig, text_quant)):
            if c1 != c2:
                print(f"   First diff at char {i}: '{c1}' vs '{c2}'")
                break
    
    print(f"\nSpeedup: {t_orig/t_quant:.2f}x")


if __name__ == "__main__":
    main()

"""
Example: Run Gemma 4 31B with RotorQuant KV Cache Compression

This example demonstrates how to use AirLLM with RotorQuant KV cache compression
to run the Gemma 4 31B DECKARD model on a single GPU with reduced VRAM usage.

Requirements:
    pip install airllm transformers torch

Usage:
    python run_gemma4_deckard.py

Model: https://huggingface.co/DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking
"""

import torch
from airllm import AutoModel

# ============================================================
# Configuration
# ============================================================
MODEL_ID = "DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking"
DEVICE = "cuda:0"
MAX_SEQ_LEN = 4096  # Adjust based on your GPU memory

# KV Cache Compression Options:
#   'planar3'      - 3-bit Givens rotation (default, best speed/quality)
#   'planar4'      - 4-bit Givens rotation (better quality)
#   'iso3'         - 3-bit quaternion (better quality than planar3)
#   'iso4'         - 4-bit quaternion (best quality)
#   'asym_planar3' - K=planar3, V=fp16 (zero PPL loss, K-only compression)
#   None           - No compression (default AirLLM behavior)
KV_COMPRESSION = "planar3"
KV_COMPRESSION_BITS = 3
BOUNDARY_LAYERS = 2  # Protect first 2 + last 2 layers from compression

# ============================================================
# Load Model
# ============================================================
print(f"Loading model: {MODEL_ID}")
print(f"KV Compression: {KV_COMPRESSION} ({KV_COMPRESSION_BITS}-bit)")
print(f"Boundary Layers: {BOUNDARY_LAYERS}")
print()

model = AutoModel.from_pretrained(
    MODEL_ID,
    device=DEVICE,
    max_seq_len=MAX_SEQ_LEN,
    kv_compression=KV_COMPRESSION,
    kv_compression_bits=KV_COMPRESSION_BITS,
    boundary_layers=BOUNDARY_LAYERS,
)

# ============================================================
# Inference
# ============================================================
prompts = [
    "Explain the concept of radiative cooling and its applications.",
    "Write a short story about a detective solving a time-travel mystery.",
]

for i, prompt in enumerate(prompts):
    print(f"\n{'=' * 60}")
    print(f"Prompt {i + 1}: {prompt}")
    print(f"{'=' * 60}")

    input_text = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )

    input_ids = model.tokenizer(input_text, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(DEVICE)

    # Generate with KV cache compression
    output = model.generate(
        input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        use_cache=True,
    )

    # Decode and print
    output_text = model.tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nResponse:\n{output_text}")

# ============================================================
# Memory Usage Report
# ============================================================
if torch.cuda.is_available():
    print(f"\n{'=' * 60}")
    print("GPU Memory Usage:")
    print(f"{'=' * 60}")
    print(f"Allocated: {torch.cuda.memory_allocated(DEVICE) / 1e9:.2f} GB")
    print(f"Reserved:  {torch.cuda.memory_reserved(DEVICE) / 1e9:.2f} GB")

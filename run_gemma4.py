#!/usr/bin/env python3
"""
Gemma 4 31B DECKARD - Interactive inference with RotorQuant KV Cache Compression.

Usage:
  Text only:   python run_gemma4.py
  With image:  python run_gemma4.py path/to/image.jpg

Environment variables:
  MODEL_ID            - HuggingFace repo ID or local path (default: DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking)
  DEVICE              - Device string (default: cuda:0)
  MAX_SEQ_LEN         - Maximum sequence length (default: 4096)
  KV_COMPRESSION      - Compression mode: planar3, planar4, iso3, iso4, asym_planar3, or empty for none
  KV_COMPRESSION_BITS - Bits per element (default: 3)
  BOUNDARY_LAYERS     - Layers protected from compression (default: 2)
  MAX_NEW_TOKENS      - Max tokens to generate (default: 256)
"""

import os
import sys

def main():
    # Configuration from environment variables
    model_id = os.environ.get(
        "MODEL_ID",
        r"E:\PROGETTI\PERSONALI\33_AIRLLM\models\gemma-4-31B-deckard",
    )
    device = os.environ.get("DEVICE", "cuda:0")
    max_seq_len = int(os.environ.get("MAX_SEQ_LEN", "4096"))
    kv_compression = os.environ.get("KV_COMPRESSION", "planar3") or None
    kv_compression_bits = int(os.environ.get("KV_COMPRESSION_BITS", "3"))
    boundary_layers = int(os.environ.get("BOUNDARY_LAYERS", "2"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "256"))

    # Optional image path from command line
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    import torch
    from airllm import AutoModel

    print("=" * 60)
    print(" Gemma 4 31B DECKARD - RotorQuant KV Cache Compression")
    print("=" * 60)
    print(f"  Model:           {model_id}")
    print(f"  Device:          {device}")
    print(f"  Max Seq Len:     {max_seq_len}")
    if kv_compression:
        print(f"  KV Compression:  {kv_compression} ({kv_compression_bits}-bit)")
    else:
        print("  KV Compression:  disabled")
    print(f"  Boundary Layers: {boundary_layers}")
    print(f"  Max New Tokens:  {max_new_tokens}")
    if image_path:
        print(f"  Image:           {image_path}")
    else:
        print("  Image:           none (text-only mode)")
    print("=" * 60)
    print()
    print("Loading model...")

    # Build kwargs
    kwargs = dict(
        device=device,
        max_seq_len=max_seq_len,
        boundary_layers=boundary_layers,
    )
    if kv_compression:
        kwargs["kv_compression"] = kv_compression
        kwargs["kv_compression_bits"] = kv_compression_bits

    model = AutoModel.from_pretrained(model_id, **kwargs)
    print("Model loaded!")

    # Load processor if available (for multimodal)
    processor = getattr(model, "processor", None)
    if processor is None and hasattr(model, "get_processor"):
        processor = model.get_processor()

    while True:
        prompt = input("\nPrompt (or 'quit'): ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        # Prepare inputs
        pixel_values = None
        pixel_position_ids = None
        padding_mask = None

        if image_path and processor is not None:
            from PIL import Image

            print(f"Processing image: {image_path}")
            image = Image.open(image_path).convert("RGB")
            inputs = processor(
                text=prompt,
                images=image,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(device)
            pixel_values = inputs.get("pixel_values")
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            pixel_position_ids = inputs.get("pixel_position_ids")
            if pixel_position_ids is not None:
                pixel_position_ids = pixel_position_ids.to(device)
            padding_mask = inputs.get("attention_mask")
            # For pixel attention, check for pixel_attention_mask
            if "pixel_attention_mask" in inputs:
                padding_mask = inputs["pixel_attention_mask"].to(device)
            print(f"  Input tokens: {input_ids.shape[1]}")
            if pixel_values is not None:
                print(f"  Image patches: {pixel_values.shape}")
        else:
            if image_path and processor is None:
                print("Warning: Image provided but no processor found, using text-only mode.")
            input_text = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = model.tokenizer(
                input_text, return_tensors="pt"
            )["input_ids"].to(device)
            print(f"  Input tokens: {input_ids.shape[1]}")

        # Generate
        generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
        )
        if pixel_values is not None:
            generate_kwargs["pixel_values"] = pixel_values
        if pixel_position_ids is not None:
            generate_kwargs["pixel_position_ids"] = pixel_position_ids
        if padding_mask is not None:
            generate_kwargs["padding_mask"] = padding_mask

        print("\nGenerating...")
        output = model.generate(input_ids, **generate_kwargs)

        print("\n" + "=" * 60)
        print("Response:")
        print("=" * 60)
        print(model.tokenizer.decode(output[0], skip_special_tokens=True))

        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated(device) / 1e9
            print(f"\nGPU Memory: {mem_gb:.2f} GB allocated")

        # Ask if user wants to continue with same/different image
        if image_path:
            new_image = input("\nChange image path (or Enter to keep, 'none' for text-only): ").strip()
            if new_image.lower() == "none":
                image_path = None
            elif new_image:
                image_path = new_image


if __name__ == "__main__":
    main()
@echo off
REM ============================================================
REM Run Gemma 4 31B DECKARD with Vision (Multimodal) Support
REM ============================================================
REM
REM KV Cache Compression Options (change KV_COMPRESSION):
REM   planar3      - 3-bit Givens rotation (default, best speed/quality)
REM   planar4      - 4-bit Givens rotation (better quality)
REM   iso3         - 3-bit quaternion (better quality than planar3)
REM   iso4         - 4-bit quaternion (best quality)
REM   asym_planar3 - K=planar3, V=fp16 (zero PPL loss, K-only compression)
REM   none         - No compression (set KV_COMPRESSION to empty string)
REM
REM Usage:
REM   Text only:     run_gemma4_deckard_vision.bat
REM   With image:    run_gemma4_deckard_vision.bat path\to\image.jpg
REM
REM Requirements:
REM   pip install -e . (install airllm in dev mode)
REM   pip install transformers torch accelerate safetensors Pillow
REM ============================================================

set MODEL_ID=DavidAU/gemma-4-31B-it-The-DECKARD-HERETIC-UNCENSORED-Thinking
set DEVICE=cuda:0
set MAX_SEQ_LEN=4096
set KV_COMPRESSION=planar3
set KV_COMPRESSION_BITS=3
set BOUNDARY_LAYERS=2
set MAX_NEW_TOKENS=256

echo ============================================================
echo  Gemma 4 31B DECKARD - Multimodal (Vision + Text)
echo ============================================================
echo  Model:          %MODEL_ID%
echo  Device:         %DEVICE%
echo  Max Seq Len:    %MAX_SEQ_LEN%
echo  KV Compression: %KV_COMPRESSION% (%KV_COMPRESSION_BITS%-bit)
echo  Boundary Layers: %BOUNDARY_LAYERS%
echo  Max New Tokens:  %MAX_NEW_TOKENS%
echo ============================================================
echo.

cd /d "%~dp0"

if "%~1"=="" (
    REM Text-only mode
    echo Running in TEXT-ONLY mode
    echo.
    python -c "import torch; from airllm import AutoModel; ^
model = AutoModel.from_pretrained('%MODEL_ID%', device='%DEVICE%', ^
max_seq_len=%MAX_SEQ_LEN%, kv_compression='%KV_COMPRESSION%', ^
kv_compression_bits=%KV_COMPRESSION_BITS%, ^
boundary_layers=%BOUNDARY_LAYERS%); ^
prompt = input('\nPrompt: '); ^
input_text = model.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True); ^
input_ids = model.tokenizer(input_text, return_tensors='pt')['input_ids'].to('%DEVICE%'); ^
output = model.generate(input_ids, max_new_tokens=%MAX_NEW_TOKENS%, do_sample=True, temperature=0.7, top_p=0.9, use_cache=True); ^
print('\nResponse:'); ^
print(model.tokenizer.decode(output[0], skip_special_tokens=True)); ^
if torch.cuda.is_available(): print(f'\nGPU Memory: {torch.cuda.memory_allocated(\"%DEVICE%\")/1e9:.2f} GB allocated')"
) else (
    REM Multimodal mode (text + image)
    echo Running in MULTIMODAL mode with image: %~1
    echo.
    python -c "import torch; from PIL import Image; from airllm import AutoModel; ^
model = AutoModel.from_pretrained('%MODEL_ID%', device='%DEVICE%', ^
max_seq_len=%MAX_SEQ_LEN%, kv_compression='%KV_COMPRESSION%', ^
kv_compression_bits=%KV_COMPRESSION_BITS%, ^
boundary_layers=%BOUNDARY_LAYERS%); ^
prompt = input('\nPrompt: '); ^
image = Image.open(r'%~1'); ^
processor = model.processor; ^
if processor is not None: ^
    inputs = processor(text=prompt, images=image, return_tensors='pt'); ^
    input_ids = inputs['input_ids'].to('%DEVICE%'); ^
    pixel_values = inputs.get('pixel_values', None); ^
    if pixel_values is not None: pixel_values = pixel_values.to('%DEVICE%'); ^
    pixel_position_ids = inputs.get('pixel_position_ids', None); ^
    if pixel_position_ids is not None: pixel_position_ids = pixel_position_ids.to('%DEVICE%'); ^
    print(f'Image tokens: {pixel_values.shape if pixel_values is not None else 0}'); ^
else: ^
    print('Warning: No processor found, falling back to text-only'); ^
    input_text = model.tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True); ^
    input_ids = model.tokenizer(input_text, return_tensors='pt')['input_ids'].to('%DEVICE%'); ^
    pixel_values = None; ^
output = model.generate(input_ids, max_new_tokens=%MAX_NEW_TOKENS%, do_sample=True, temperature=0.7, top_p=0.9, use_cache=True, ^
pixel_values=pixel_values, pixel_position_ids=pixel_position_ids if pixel_values is not None else None); ^
print('\nResponse:'); ^
print(model.tokenizer.decode(output[0], skip_special_tokens=True)); ^
if torch.cuda.is_available(): print(f'\nGPU Memory: {torch.cuda.memory_allocated(\"%DEVICE%\")/1e9:.2f} GB allocated')"
)

pause
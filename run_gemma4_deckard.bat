@echo off
REM ============================================================
REM Run Gemma 4 31B DECKARD with RotorQuant KV Cache Compression
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
REM   Text only:    run_gemma4_deckard.bat
REM   With image:   run_gemma4_deckard.bat path\to\image.jpg
REM
REM Requirements:
REM   pip install -e . (install airllm in dev mode)
REM   pip install transformers torch accelerate safetensors Pillow
REM ============================================================

set MODEL_ID=E:\PROGETTI\PERSONALI\33_AIRLLM\models\gemma-4-31B-deckard
set DEVICE=cuda:0
set MAX_SEQ_LEN=4096
set KV_COMPRESSION=planar3
set KV_COMPRESSION_BITS=3
set BOUNDARY_LAYERS=2
set MAX_NEW_TOKENS=256

cd /d "%~dp0"

python run_gemma4.py %1
pause
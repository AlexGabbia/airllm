@echo off
REM ============================================================
REM Run Gemma 4 31B DECKARD - Multimodal (Vision + Text)
REM ============================================================
REM
REM Usage:
REM   Text only:    run_gemma4_deckard_vision.bat
REM   With image:   run_gemma4_deckard_vision.bat path\to\image.jpg
REM
REM Same as run_gemma4_deckard.bat - both call run_gemma4.py
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
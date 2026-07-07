@echo off
REM Train small (k1 baseline) + small_k16head and compare aliasing (SNR-A).
REM Run from the repo root. Passes all args through to the Python tool.
REM
REM   run_head_kernel_ab.bat --data-dir "D:\captures\ADA MP-1 Captures FULL 1-25"
REM   run_head_kernel_ab.bat --data-dir "...captures..." --epochs 60 --od1 8 --od2 8
REM
python "%~dp0tools\head_kernel_ab.py" %*

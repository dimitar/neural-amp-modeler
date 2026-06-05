@echo off
setlocal enabledelayedexpansion

REM Non-interactive variant of resume_training.bat for use over SSH/tmux.
REM Runs directly in the current console (no new-window relaunch, no pause).
REM For local GUI use, prefer resume_training.bat which opens a fresh window.
REM
REM Defaults resume the latest checkpoint in parametric_output\checkpoints
REM using the 'small' preset. To resume the archived 101K run instead,
REM edit the OUTPUT_DIR and MODEL_SIZE vars below.

cd /d "%~dp0"

set "OUTPUT_DIR=parametric_output"
set "MODEL_SIZE=small"

set "CHECKPOINT="
for /f "delims=" %%i in ('dir /b /o-d "%OUTPUT_DIR%\checkpoints\best-*.ckpt" 2^>nul') do (
    if not defined CHECKPOINT set "CHECKPOINT=%OUTPUT_DIR%\checkpoints\%%i"
)

if not defined CHECKPOINT (
    echo ERROR: No checkpoint found in %OUTPUT_DIR%\checkpoints
    exit /b 1
)

echo ============================================================
echo  NAM Parametric Training - Resume
echo ============================================================
echo  Checkpoint : !CHECKPOINT!
echo  Model size : %MODEL_SIZE%
echo  Output dir : %OUTPUT_DIR%
echo  Data dir   : data\ADA MP-1-Tube-Dist-PS2-96khz
echo  Started    : %DATE% %TIME%
echo ============================================================

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" nam || (
    echo ERROR: Failed to activate conda env 'nam'
    exit /b 1
)

python train_parametric.py train ^
    --data-dir "data/ADA MP-1-Tube-Dist-PS2-96khz" ^
    --output-dir "%OUTPUT_DIR%" ^
    --model-size %MODEL_SIZE% ^
    --delay 10 ^
    --train-stop-seconds -9.0 ^
    --val-start-seconds -9.0 ^
    --resume "!CHECKPOINT!"

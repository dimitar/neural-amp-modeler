@echo off
setlocal enabledelayedexpansion

REM Non-interactive variant of start_training.bat for use over SSH/tmux.
REM Runs directly in the current console (no new-window relaunch, no pause).
REM For local GUI use, prefer start_training.bat which opens a fresh window.

cd /d "%~dp0"

if exist "parametric_output\checkpoints\best-*.ckpt" (
    echo WARNING: parametric_output\checkpoints already contains checkpoints.
    echo Starting a fresh run will append to the existing output directory.
    echo Aborting. Rename parametric_output if you want a clean slate, or use resume.bat.
    exit /b 1
)

echo ============================================================
echo  NAM Parametric Training - Fresh Run
echo ============================================================
echo  Model size : small (~26K params)
echo  Data dir   : data\ADA MP-1-Tube-Dist-PS2-96khz
echo  Started    : %DATE% %TIME%
echo ============================================================

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" nam || (
    echo ERROR: Failed to activate conda env 'nam'
    exit /b 1
)

python train_parametric.py train ^
    --data-dir "data/ADA MP-1-Tube-Dist-PS2-96khz" ^
    --delay 10 ^
    --train-stop-seconds -9.0 ^
    --val-start-seconds -9.0 ^
    --model-size small

@echo off
setlocal enabledelayedexpansion

REM Self-relaunch: outer mode opens a new terminal that runs this same .bat
REM with --inner, so the training command executes in a fresh window.
REM
REM Defaults resume the latest checkpoint in parametric_output\checkpoints
REM using the 'small' preset. To resume the archived 101K run instead:
REM   1. Edit OUTPUT_DIR below to parametric_output_101k
REM   2. Edit MODEL_SIZE below to large
if /I "%~1"=="--inner" goto :run

start "NAM Parametric Training (resume)" /D "%~dp0" cmd /k "%~f0" --inner
exit /b 0

:run
cd /d "%~dp0"

set "OUTPUT_DIR=parametric_output"
set "MODEL_SIZE=small"

REM Auto-detect newest checkpoint by modification time
set "CHECKPOINT="
for /f "delims=" %%i in ('dir /b /o-d "%OUTPUT_DIR%\checkpoints\best-*.ckpt" 2^>nul') do (
    if not defined CHECKPOINT set "CHECKPOINT=%OUTPUT_DIR%\checkpoints\%%i"
)

if not defined CHECKPOINT (
    echo.
    echo ERROR: No checkpoint found in %OUTPUT_DIR%\checkpoints
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  NAM Parametric Training - Resume
echo ============================================================
echo  Checkpoint : !CHECKPOINT!
echo  Model size : %MODEL_SIZE%
echo  Output dir : %OUTPUT_DIR%
echo  Data dir   : data\ADA MP-1-Tube-Dist-PS2-96khz
echo  Started    : %DATE% %TIME%
echo ============================================================
echo.

call "%USERPROFILE%\miniconda3\Scripts\activate.bat" nam
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate conda env 'nam'
    echo.
    pause
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

echo.
echo ============================================================
echo  Training process exited.
echo ============================================================
pause

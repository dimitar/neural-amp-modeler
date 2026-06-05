@echo off
setlocal enabledelayedexpansion

REM Resumes the Tube-Clean parametric training run from the latest checkpoint.
REM Self-relaunch: outer mode opens a new terminal that runs this same .bat
REM with --inner, so the training command executes in a fresh window.

if /I "%~1"=="--inner" goto :run

start "NAM Parametric Training - Tube-Clean (resume)" /D "%~dp0" cmd /k "%~f0" --inner
exit /b 0

:run
cd /d "%~dp0"

set "OUTPUT_DIR=parametric_output_tube_clean"
set "MODEL_SIZE=small"
set "DATA_DIR=data/ADA MP-1-Tube-Clean-PS2-96khz"

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
echo  NAM Parametric Training - Tube-Clean - Resume
echo ============================================================
echo  Checkpoint : !CHECKPOINT!
echo  Model size : %MODEL_SIZE%
echo  Data dir   : %DATA_DIR%
echo  Output dir : %OUTPUT_DIR%
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
    --data-dir "%DATA_DIR%" ^
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

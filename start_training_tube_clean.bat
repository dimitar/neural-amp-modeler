@echo off
setlocal enabledelayedexpansion

REM Launches a fresh parametric training run for the ADA MP-1 Tube-Clean
REM 96kHz dataset (small preset) in a new terminal.
REM For resuming from a checkpoint, use resume_training_tube_clean.bat.

if /I "%~1"=="--inner" goto :run

start "NAM Parametric Training - Tube-Clean (fresh)" /D "%~dp0" cmd /k "%~f0" --inner
exit /b 0

:run
cd /d "%~dp0"

set "OUTPUT_DIR=parametric_output_tube_clean"
set "DATA_DIR=data/ADA MP-1-Tube-Clean-PS2-96khz"

if exist "%OUTPUT_DIR%\checkpoints\best-*.ckpt" (
    echo.
    echo WARNING: %OUTPUT_DIR%\checkpoints already contains checkpoints.
    echo Starting a fresh run will append to the existing output directory.
    echo If you want a clean slate, rename %OUTPUT_DIR% first.
    echo.
    choice /C YN /M "Continue anyway"
    if errorlevel 2 exit /b 1
    echo.
)

echo.
echo ============================================================
echo  NAM Parametric Training - Tube-Clean - Fresh Run
echo ============================================================
echo  Model size : small (~26K params)
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
    --delay 10 ^
    --train-stop-seconds -9.0 ^
    --val-start-seconds -9.0 ^
    --model-size small

echo.
echo ============================================================
echo  Training process exited.
echo ============================================================
pause

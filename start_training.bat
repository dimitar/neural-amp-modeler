@echo off
setlocal enabledelayedexpansion

REM Launches a fresh parametric training run (small preset) in a new terminal.
REM For resuming from a checkpoint, use resume_training.bat instead.

if /I "%~1"=="--inner" goto :run

start "NAM Parametric Training (fresh)" /D "%~dp0" cmd /k "%~f0" --inner
exit /b 0

:run
cd /d "%~dp0"

if exist "parametric_output\checkpoints\best-*.ckpt" (
    echo.
    echo WARNING: parametric_output\checkpoints already contains checkpoints.
    echo Starting a fresh run will append to the existing output directory.
    echo If you want a clean slate, rename parametric_output first.
    echo.
    choice /C YN /M "Continue anyway"
    if errorlevel 2 exit /b 1
    echo.
)

echo.
echo ============================================================
echo  NAM Parametric Training - Fresh Run
echo ============================================================
echo  Model size : small (~26K params)
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
    --delay 10 ^
    --train-stop-seconds -9.0 ^
    --val-start-seconds -9.0 ^
    --model-size small

echo.
echo ============================================================
echo  Training process exited.
echo ============================================================
pause

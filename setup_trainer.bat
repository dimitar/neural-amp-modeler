@echo off
REM First-time setup for the NAM Parametric Trainer.
REM Creates the 'nam' conda environment from environments\environment_gpu.yml.
REM Safe to re-run (conda reports the env already exists). Run from the
REM neural-amp-modeler folder, or let launch_trainer.bat invoke it automatically.

setlocal
cd /d "%~dp0"

REM ── Find a conda installation (without requiring it on PATH) ─────────────────
set "CONDA_ROOT="
for %%P in (
    "%CONDA_PREFIX%"
    "%CONDA_PREFIX_1%"
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "%LOCALAPPDATA%\miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "%ProgramData%\miniconda3"
    "%ProgramData%\Anaconda3"
) do if not defined CONDA_ROOT if exist "%%~P\Scripts\activate.bat" set "CONDA_ROOT=%%~P"
if not defined CONDA_ROOT for /f "delims=" %%C in ('where conda 2^>nul') do (
    if not defined CONDA_ROOT if exist "%%~dpC..\Scripts\activate.bat" set "CONDA_ROOT=%%~dpC.."
)

if not defined CONDA_ROOT (
    echo.
    echo Could not find Miniconda/Anaconda. Install Miniconda first.
    echo.
    pause
    exit /b 1
)
if not exist "environments\environment_gpu.yml" (
    echo.
    echo environments\environment_gpu.yml was not found next to this script.
    echo Run this from the neural-amp-modeler folder.
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo  NAM Trainer - first-time environment setup
echo.
echo  Creating the 'nam' conda environment from
echo  environments\environment_gpu.yml
echo.
echo  This downloads PyTorch + CUDA and can take 10-30+ minutes.
echo  Leave this window open until it finishes.
echo ============================================================
echo.

call "%CONDA_ROOT%\Scripts\activate.bat"

REM Create the env if it isn't there yet (harmless "prefix already exists" if it is).
if not exist "%CONDA_ROOT%\envs\nam\python.exe" (
    call conda env create -f environments\environment_gpu.yml
)

if not exist "%CONDA_ROOT%\envs\nam\python.exe" (
    echo.
    echo Environment creation did NOT finish successfully. Review the messages above.
    echo.
    pause
    exit /b 1
)

REM Ensure the trainer UI components are installed (also repairs an env that was
REM created before these were added to environment_gpu.yml).
echo.
echo Ensuring trainer UI components (gradio, pywebview) are installed...
call "%CONDA_ROOT%\Scripts\activate.bat" "%CONDA_ROOT%\envs\nam"
call python -m pip install gradio pywebview

echo.
if exist "%CONDA_ROOT%\envs\nam\Lib\site-packages\gradio" if exist "%CONDA_ROOT%\envs\nam\Lib\site-packages\webview" (
    echo Setup complete. The trainer can now be launched.
    echo.
    pause
    exit /b 0
)
echo Setup finished, but the UI components may not have installed correctly.
echo Review the messages above.
echo.
pause
endlocal

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
call conda env create -f environments\environment_gpu.yml

echo.
if exist "%CONDA_ROOT%\envs\nam\pythonw.exe" (
    echo Setup complete. The trainer can now be launched.
) else (
    echo Setup did NOT finish successfully. Review the messages above.
    echo ^(If you saw "prefix already exists", the env may already be set up.^)
)
echo.
pause
endlocal

@echo off
REM Launches the NAM Parametric Trainer in a native window, with no console.
REM
REM Self-bootstraps the 'nam' conda environment for THIS process only so it works
REM whether double-clicked from the desktop, run from a plain cmd, or run from the
REM Anaconda Prompt. It never runs `conda init` and never edits the system PATH —
REM activation here is scoped to this process (and discarded by endlocal).

setlocal

REM ── 1. Locate the conda installation (without requiring it on PATH) ──────────
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

REM Fallback: conda already on PATH (e.g. launched from the Anaconda Prompt).
if not defined CONDA_ROOT for /f "delims=" %%C in ('where conda 2^>nul') do (
    if not defined CONDA_ROOT if exist "%%~dpC..\Scripts\activate.bat" set "CONDA_ROOT=%%~dpC.."
)

if not defined CONDA_ROOT (
    echo.
    echo Could not find a Miniconda/Anaconda installation.
    echo Install Miniconda, or launch this from the Anaconda Prompt.
    echo.
    pause
    exit /b 1
)

REM ── 2. Activate the 'nam' env for this process only ─────────────────────────
call "%CONDA_ROOT%\Scripts\activate.bat" nam
if errorlevel 1 (
    echo.
    echo Failed to activate the 'nam' conda environment using "%CONDA_ROOT%".
    echo Make sure the env exists:  conda env create -f environments\environment_gpu.yml
    echo.
    pause
    exit /b 1
)

REM ── 3. Launch from the env's own interpreter (avoids any PATH lookup) ────────
set "ENV_PYW=%CONDA_PREFIX%\pythonw.exe"
if not exist "%ENV_PYW%" set "ENV_PYW=pythonw"

cd /d "%~dp0"
start "" "%ENV_PYW%" -m trainer_app.launch

endlocal

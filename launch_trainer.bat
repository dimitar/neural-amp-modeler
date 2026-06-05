@echo off
REM Launches the NAM Parametric Trainer in a native window, with no console.
REM
REM Self-bootstraps the 'nam' conda environment for THIS process only so it works
REM whether double-clicked, run from a plain cmd, or run from the Anaconda Prompt.
REM It never runs `conda init` and never edits the system PATH.
REM
REM IMPORTANT: we do NOT trust `call activate.bat`'s exit code. On some conda
REM versions it returns nonzero even when activation actually succeeded (which made
REM an earlier version report "failed to activate" on working setups). Instead we
REM locate the env BY PATH and launch the env's own pythonw.exe directly.

setlocal

REM ── Find the conda install that contains the 'nam' env ──────────────────────
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
) do if not defined CONDA_ROOT if exist "%%~P\envs\nam\python.exe" set "CONDA_ROOT=%%~P"
if not defined CONDA_ROOT for /f "delims=" %%C in ('where conda 2^>nul') do (
    if not defined CONDA_ROOT if exist "%%~dpC..\envs\nam\python.exe" set "CONDA_ROOT=%%~dpC.."
)

if not defined CONDA_ROOT (
    echo.
    echo The 'nam' conda environment was not found.
    echo Run setup_trainer.bat once to create it, then try again.
    echo.
    pause
    exit /b 1
)

set "NAM_ENV=%CONDA_ROOT%\envs\nam"

if not exist "%NAM_ENV%\pythonw.exe" (
    echo.
    echo The 'nam' env at "%NAM_ENV%" looks incomplete ^(no pythonw.exe^).
    echo Re-run setup_trainer.bat to repair it.
    echo.
    pause
    exit /b 1
)

REM ── Ensure the trainer UI components are installed (gradio, pywebview). ──────
REM    Missing UI deps were the cause of the window "closing silently": pythonw
REM    crashed on `import gradio`/`import webview` with no console to show it.
if not exist "%NAM_ENV%\Lib\site-packages\gradio" goto :setup
if not exist "%NAM_ENV%\Lib\site-packages\webview" goto :setup
goto :launch

:setup
echo.
echo The trainer UI components aren't installed in the 'nam' env yet.
echo Running one-time setup to install them ^(a setup window will open^)...
echo.
start "NAM Trainer setup" /wait "%~dp0setup_trainer.bat"
if not exist "%NAM_ENV%\Lib\site-packages\gradio" goto :setupfail
if not exist "%NAM_ENV%\Lib\site-packages\webview" goto :setupfail
goto :launch

:setupfail
echo.
echo Setup did not install the trainer UI components. Check the setup window output,
echo or run:  conda activate nam  ^&^&  pip install gradio pywebview
echo.
pause
exit /b 1

:launch
REM Activate for this process only (sets up DLL/PATH). Exit code intentionally
REM ignored; success is verified by the checks above.
call "%CONDA_ROOT%\Scripts\activate.bat" "%NAM_ENV%"
cd /d "%~dp0"
start "" "%NAM_ENV%\pythonw.exe" -m trainer_app.launch

endlocal

@echo off
REM Launches the NAM Parametric Trainer with no console window.
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" nam
cd /d "%~dp0"
start "" pythonw -m trainer_app.launch

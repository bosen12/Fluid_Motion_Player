@echo off
setlocal
cd /d "%~dp0"
py -3 -m pip install -r requirements.txt -q
py -3 -m fluid_motion %*
if errorlevel 1 (
  python -m pip install -r requirements.txt -q
  python -m fluid_motion %*
)

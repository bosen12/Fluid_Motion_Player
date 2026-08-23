@echo off
setlocal
cd /d "%~dp0"
py -3 -m pip install -r requirements.txt pyinstaller -q
py -3 -m PyInstaller --noconfirm --clean FluidMotion.spec
if errorlevel 1 (
  python -m pip install -r requirements.txt pyinstaller -q
  python -m PyInstaller --noconfirm --clean FluidMotion.spec
)
if exist dist\FluidMotion.exe copy /Y dist\FluidMotion.exe FluidMotion.exe >nul
echo.
echo Built: dist\FluidMotion.exe
endlocal

@echo off
setlocal
cd /d "%~dp0"
py -3 -m pip install -r requirements.txt pyinstaller -q
py -3 -m PyInstaller --noconfirm --clean FluidMotion.spec
if errorlevel 1 (
  python -m pip install -r requirements.txt pyinstaller -q
  python -m PyInstaller --noconfirm --clean FluidMotion.spec
  if errorlevel 1 goto :failed
)
if not exist dist\FluidMotion.exe goto :failed
copy /Y dist\FluidMotion.exe FluidMotion.exe >nul
if errorlevel 1 (
  echo.
  echo Build succeeded but FluidMotion.exe is locked -- close the running app, then copy
  echo dist\FluidMotion.exe over it manually.
  endlocal
  exit /b 1
)
echo.
echo Built: dist\FluidMotion.exe
endlocal
exit /b 0

:failed
echo.
echo BUILD FAILED -- dist\FluidMotion.exe was not produced.
echo If FluidMotion is running, quit it from the tray first: PyInstaller cannot
echo overwrite a locked exe.
endlocal
exit /b 1

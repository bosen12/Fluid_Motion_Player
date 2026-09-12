@echo off
setlocal
rem Pinned to 3.14 on purpose. This used to say `py -3`, which resolves to
rem whatever the py launcher's default happens to be -- 3.14 today, and every
rem release so far was built with it. If that default ever moves, `py -3`
rem would ship a different interpreter with no sign that anything changed:
rem the same commit built on 3.10 produces a visibly smaller, different
rem bundle. Failing loudly when 3.14 is absent is the point.
cd /d "%~dp0"
py -3.14 -m pip install -r requirements.txt pyinstaller -q
if errorlevel 1 goto :failed
py -3.14 -m PyInstaller --noconfirm --clean FluidMotion.spec
if errorlevel 1 goto :failed
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

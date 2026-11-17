@echo off
setlocal
cd /d D:\longmemeval
call ".venv\Scripts\activate.bat"
for /f "delims=" %%D in ('powershell -NoProfile -Command "(Get-ChildItem 'results\secom_bge\secom_*' -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName"') do set "RUN_DIR=%%D"
if not defined RUN_DIR (echo no run dir & exit /b 1)
echo %RUN_DIR%
python "%~dp0check_run.py" "%RUN_DIR%"
pause

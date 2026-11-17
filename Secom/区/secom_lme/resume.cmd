@echo off
setlocal
cd /d D:\longmemeval
call ".venv\Scripts\activate.bat"
for /f "delims=" %%D in ('powershell -NoProfile -Command "(Get-ChildItem 'results\secom_bge\secom_*' -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName"') do set "RUN_DIR=%%D"
if not defined RUN_DIR (echo no run dir & exit /b 1)
echo resume %RUN_DIR%
python -u run_secom_lme.py --resume "%RUN_DIR%" --data longmemeval_m_cleaned.json --subset subset_m_500_ids.json --topk 10 --skip-abs
pause

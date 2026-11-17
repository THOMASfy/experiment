@echo off
REM Restart the LongMemEval-S (mem0-lme, full quality) run only.
REM M is intentionally NOT chained here: we review S metrics first, then start M manually.
REM Allowed: 12:00-14:00 and 18:00-09:00 (北京时间); blocked Mon-Fri 09:00-12:00 / 14:00-18:00.
cd /d C:\experiment
start "" "C:\Users\NewAdmin\AppData\Local\Programs\Python\Python312\pythonw.exe" ^
  "C:\experiment\schedule_runner.py" ^
  --run-dir "C:\experiment\results\mem0_lme_s\mem0lme_20261116-090147" ^
  --find-existing ^
  --poll 15 ^
  --kill-on-exit

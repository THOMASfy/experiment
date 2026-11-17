$ErrorActionPreference = "Continue"
$target = Get-Date "2026-11-16 18:00:05"
while ((Get-Date) -lt $target) { Start-Sleep -Seconds 30 }
Set-Location "C:\experiment"
$py = "C:\Users\NewAdmin\AppData\Local\Programs\Python\Python312\python.exe"
& $py -u raw_bge_eval.py --resume-dir "results\raw_bge_s_all\raw_20261116-091015" --data longmemeval_s_cleaned.json --granularity turn --roles all --batch-size 128 *>> results\raw_bge_s_all_console.log

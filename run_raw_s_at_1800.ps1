$ErrorActionPreference = "Continue"
$target = Get-Date "2026-11-12 18:00:10"
while ((Get-Date) -lt $target) { Start-Sleep -Seconds 20 }
Set-Location "C:\experiment"
$py = "C:\Users\NewAdmin\AppData\Local\Programs\Python\Python312\python.exe"
& $py -u raw_bge_eval.py --data longmemeval_s_cleaned.json --out-root results\raw_bge_s_user --granularity turn --roles user --batch-size 128 *>> results\raw_bge_s_user_console.log
& $py -u raw_bge_eval.py --data longmemeval_s_cleaned.json --out-root results\raw_bge_s_all --granularity turn --roles all --batch-size 128 *>> results\raw_bge_s_all_console.log

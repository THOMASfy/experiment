$ErrorActionPreference = "Continue"
$target = Get-Date "2026-11-12 18:00:25"
while ((Get-Date) -lt $target) { Start-Sleep -Seconds 15 }
Set-Location "C:\experiment"
$py = "C:\Users\NewAdmin\AppData\Local\Programs\Python\Python312\python.exe"
& $py -u run_mem0_lme.py --data tiny_lme.json --profile lme --extract-cache results\extract_cache\smoke.sqlite --topk 10 --limit 1 --workers 1 --out-root results\mem0_lme_smoke --delete-after-question *>> results\mem0_lme_smoke_console.log

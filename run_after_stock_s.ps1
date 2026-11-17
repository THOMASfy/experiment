$ErrorActionPreference = "Continue"
$root = "C:\experiment"
$stockS = Join-Path $root "results\mem0_bge_s\mem0_20261112-143726\aggregate.json"
$py = "C:\Users\NewAdmin\AppData\Local\Programs\Python\Python312\python.exe"
$pyw = "C:\Users\NewAdmin\AppData\Local\Programs\Python\Python312\pythonw.exe"

Write-Output "waiting for stock S aggregate: $stockS"
while (-not (Test-Path $stockS)) { Start-Sleep -Seconds 60 }
Write-Output "stock S done at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

Set-Location $root

# 两个验证跑并行 (各 8 workers -> 合计 16 并发, 和 stock S 一样)
$runA = "C:\experiment\results\mem0_lme_s_val\mem0lme_20261112-171058"
$runB = "C:\experiment\results\mem0_lme_s_val\mem0lme_20261112-171059"

Start-Process -FilePath $pyw -ArgumentList @(
    "-u", "$root\schedule_runner.py", "--run-dir", $runA, "--find-existing", "--poll", "15", "--kill-on-exit"
) -WorkingDirectory $root -WindowStyle Hidden
Start-Process -FilePath $pyw -ArgumentList @(
    "-u", "$root\schedule_runner.py", "--run-dir", $runB, "--find-existing", "--poll", "15", "--kill-on-exit"
) -WorkingDirectory $root -WindowStyle Hidden

Write-Output "started validation A ($runA) and B ($runB) at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# 等两个验证跑都出 aggregate
while (-not (Test-Path (Join-Path $runA "aggregate.json"))) { Start-Sleep -Seconds 60 }
Write-Output "validation A done at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
while (-not (Test-Path (Join-Path $runB "aggregate.json"))) { Start-Sleep -Seconds 60 }
Write-Output "validation B done at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "ALL_VALIDATIONS_DONE"

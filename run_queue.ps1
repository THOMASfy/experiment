$ErrorActionPreference = "Continue"
$root = "C:\experiment"
$pyw = "C:\Users\NewAdmin\AppData\Local\Programs\Python\Python312\pythonw.exe"
$log = Join-Path $root "results\queue.log"

function Say($m) {
    $line = "{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Add-Content -Path $log -Value $line
}

Set-Location $root

$stockS = Join-Path $root "results\mem0_bge_s\mem0_20261112-143726\aggregate.json"
$sLmeDir = Join-Path $root "results\mem0_lme_s\mem0lme_20261116-090147"
$mLmeDir = Join-Path $root "results\mem0_lme_m\mem0lme_20261112-180716"

function Start-Stage($dir, $name) {
    Say "start stage $name -> $dir"
    Start-Process -FilePath $pyw -ArgumentList @(
        "-u", "$root\schedule_runner.py", "--run-dir", $dir, "--find-existing", "--poll", "15", "--kill-on-exit"
    ) -WorkingDirectory $root -WindowStyle Hidden | Out-Null
}

function Wait-Agg($dir, $name) {
    $agg = Join-Path $dir "aggregate.json"
    while (-not (Test-Path $agg)) { Start-Sleep -Seconds 60 }
    Say "stage $name done (aggregate.json exists)"
}

Say "queue start | S-stock -> S-lme(auto) -> M-lme(auto)"

Wait-Agg (Split-Path $stockS) "S-stock"
Start-Stage $sLmeDir "S-lme"
Wait-Agg $sLmeDir "S-lme"

Start-Stage $mLmeDir "M-lme"
Wait-Agg $mLmeDir "M-lme"

Say "ALL DONE"

$ErrorActionPreference = "Continue"
$root = "C:\experiment"
$py = "C:\Users\NewAdmin\AppData\Local\Programs\Python\Python312\python.exe"
$pyw = "C:\Users\NewAdmin\AppData\Local\Programs\Python\Python312\pythonw.exe"
$log = Join-Path $root "results\m_stage.log"

function Say($m) {
    $line = "{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Add-Content -Path $log -Value $line
}

Set-Location $root

$sLmeDir = Join-Path $root "results\mem0_lme_s\mem0lme_20261116-090147"
$mLmeDir = Join-Path $root "results\mem0_lme_m\mem0lme_20261112-180716"
$decision = Join-Path $root "results\m_decision.json"

Say "waiting for S-lme aggregate..."
while (-not (Test-Path (Join-Path $sLmeDir "aggregate.json"))) { Start-Sleep -Seconds 60 }
Say "S-lme done at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

# 配置: 有决策文件就用决策文件, 否则用保守默认 (rich / 无上限 / 不粗筛 / workers 24)
$cfg = @{ workers = 24; fact_style = "rich"; max_facts = 0; batch_sessions = 1; prefilter_topk = 0 }
if (Test-Path $decision) {
    try {
        $d = Get-Content $decision -Raw | ConvertFrom-Json
        if ($null -ne $d.workers) { $cfg.workers = $d.workers }
        if ($null -ne $d.fact_style) { $cfg.fact_style = $d.fact_style }
        if ($null -ne $d.max_facts) { $cfg.max_facts = $d.max_facts }
        if ($null -ne $d.batch_sessions) { $cfg.batch_sessions = $d.batch_sessions }
        if ($null -ne $d.prefilter_topk) { $cfg.prefilter_topk = $d.prefilter_topk }
        Say "using decision file: $decision"
    } catch {
        Say "decision file unreadable, using default: $($_.Exception.Message)"
    }
} else {
    Say "no decision file -> conservative default (rich/no-cap/no-prefilter)"
}
Say ("M config: " + ($cfg | ConvertTo-Json -Compress))

& $py -u set_run_config.py --run-dir $mLmeDir --workers $cfg.workers --fact-style $cfg.fact_style `
    --max-facts $cfg.max_facts --batch-sessions $cfg.batch_sessions --prefilter-topk $cfg.prefilter_topk *>> $log

Say "starting M scheduler"
Start-Process -FilePath $pyw -ArgumentList @(
    "-u", "$root\schedule_runner.py", "--run-dir", $mLmeDir, "--find-existing", "--poll", "15", "--kill-on-exit"
) -WorkingDirectory $root -WindowStyle Hidden | Out-Null

while (-not (Test-Path (Join-Path $mLmeDir "aggregate.json"))) { Start-Sleep -Seconds 60 }
Say "M DONE"

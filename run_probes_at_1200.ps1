$ErrorActionPreference = "Continue"
$root = "C:\experiment"
$py = "C:\Users\NewAdmin\AppData\Local\Programs\Python\Python312\python.exe"
$log = Join-Path $root "results\probe_smoke.log"

function Say($m) {
    $line = "{0} {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $m
    Add-Content -Path $log -Value $line
    Write-Output $line
}

Set-Location $root
$target = Get-Date "2026-11-16 12:00:20"
while ((Get-Date) -lt $target) { Start-Sleep -Seconds 20 }
Say "probes start"

# 1) 并发探针: 决定 M 的 workers 能加到多少 (零质量风险)
Say "probe 1/3: concurrency"
& $py -u probe_concurrency.py --levels 8,16,24,32 --calls-per-level 8 --out results\concurrency_probe.json *>> results\probe_concurrency_console.log
Say "concurrency probe done"

# 2) prefilter 召回天花板 (零 LLM 成本, 只用 BGE)
Say "probe 2/3: prefilter ceiling (S)"
& $py -u probe_prefilter_ceiling.py --data longmemeval_s_cleaned.json --n 14 --out results\prefilter_ceiling_S.json *>> results\probe_ceiling_console.log
Say "probe 2b/3: prefilter ceiling (M, 6 题)"
& $py -u probe_prefilter_ceiling.py --data longmemeval_m_cleaned.json --n 6 --out results\prefilter_ceiling_M.json *>> results\probe_ceiling_console.log
Say "prefilter ceiling done"

# 3) 变体 smoke: 在同一批 14 题上跑 (各自独立 cache)
Say "probe 3/3: smoke variants (short5 + prefilter64)"
& $py -u smoke_lme_variants.py --subset subsets\s_smoke_12.json --workers 8 --variants short5,prefilter64 --out-root results\smoke_lme *>> results\smoke_console.log
Say "smoke variants done"
Say "PROBES_DONE"

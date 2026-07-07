# =============================================================================
# RUN_ALL_SAHI_TTA.ps1 -- unattended runner for the 3 figure/eval scripts.
# Runs them ONE AFTER ANOTHER with logging, keep-awake, venv auto-repair and
# auto-retry (the big eval resumes from its cache, so retries lose nothing).
#
# USE (VS Code PowerShell terminal, from the CBAM_P2Head folder):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\RUN_ALL_SAHI_TTA.ps1
# Then walk away. Everything is logged to .\logs\<timestamp>_*.log
# =============================================================================

$ErrorActionPreference = "Continue"
$ROOT  = Split-Path -Parent $MyInvocation.MyCommand.Path
$VENV  = "E:\Thesis_mofazzal_2007074\mofazzal1"
$LOGS  = Join-Path $ROOT "logs"
if (-not (Test-Path $LOGS)) { New-Item -ItemType Directory -Path $LOGS | Out-Null }
$STAMP = Get-Date -Format "yyyyMMdd_HHmmss"
Set-Location $ROOT

function Say($msg) { Write-Host ("[runner] " + $msg) -ForegroundColor Cyan }

# ---------- 0. keep the PC awake while this script runs ----------
try {
    Add-Type -Name PW -Namespace Sys -MemberDefinition `
        '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
    [Sys.PW]::SetThreadExecutionState([uint32]"0x80000001") | Out-Null   # ES_CONTINUOUS|ES_SYSTEM_REQUIRED
    Say "keep-awake ON (system will not sleep while this window is open)"
} catch { Say "keep-awake failed (non-fatal) -- disable sleep manually if needed" }

# ---------- 1. find a working python for the venv (auto-repair pyvenv.cfg) ----------
$PY = Join-Path $VENV "Scripts\python.exe"
function Test-Py($exe) {
    if (-not (Test-Path $exe)) { return $false }
    $v = & $exe --version 2>$null
    return ($LASTEXITCODE -eq 0 -and $v)
}

if (-not (Test-Py $PY)) {
    Say "venv python is BROKEN (base interpreter missing) -- attempting pyvenv.cfg repair"
    $candidates = @(
        "$env:LocalAppData\Programs\Python\Python311",
        "C:\Program Files\Python311",
        "C:\Python311",
        "$env:LocalAppData\Programs\Python\Python312",
        "C:\Program Files\Python312"
    )
    # also ask the py launcher, newest first
    try {
        $pyList = & py -0p 2>$null
        if ($pyList) {
            foreach ($line in $pyList) {
                $m = [regex]::Match($line, "([A-Za-z]:\\\S+python\.exe)")
                if ($m.Success) { $candidates += (Split-Path -Parent $m.Groups[1].Value) }
            }
        }
    } catch {}
    $base = $null
    foreach ($c in $candidates) {
        if (Test-Path (Join-Path $c "python.exe")) { $base = $c; break }
    }
    if ($null -eq $base) {
        Say "NO Python found on this PC. Install per-user Python 3.11.9 first:"
        Say '  cd $env:TEMP'
        Say '  Invoke-WebRequest https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe -OutFile py311.exe'
        Say '  .\py311.exe /quiet InstallAllUsers=0 InstallLauncherAllUsers=0 PrependPath=1 Include_launcher=1'
        Say "then open a NEW terminal and re-run this script."
        exit 1
    }
    Say "found base Python at: $base -- patching $VENV\pyvenv.cfg"
    $cfgPath = Join-Path $VENV "pyvenv.cfg"
    Copy-Item $cfgPath "$cfgPath.bak" -Force
    $cfg = Get-Content $cfgPath
    $cfg = $cfg | ForEach-Object {
        if     ($_ -match "^home\s*=")       { "home = $base" }
        elseif ($_ -match "^executable\s*=") { "executable = $(Join-Path $base 'python.exe')" }
        elseif ($_ -match "^command\s*=")    { "command = $(Join-Path $base 'python.exe') -m venv $VENV" }
        else   { $_ }
    }
    Set-Content -Path $cfgPath -Value $cfg -Encoding ascii
    if (-not (Test-Py $PY)) {
        Say "repair FAILED -- venv still broken. If the found Python is not 3.11, install 3.11.9 (command above)."
        exit 1
    }
    Say "venv REPAIRED (packages preserved, nothing reinstalled)"
}
$pyver = & $PY --version
Say "python OK: $pyver  ($PY)"
& $PY -c "import torch; print('[runner] torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---------- 2. pre-flight ----------
$weights = "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM_P2Head\runs\20260602_063759_yolo11m_cbam_p2head_s0_nogit\weights\best.pt"
$testimg = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3\test\images"
if (-not (Test-Path $weights)) { Say "FATAL: canonical best.pt missing: $weights"; exit 1 }
$n = (Get-ChildItem $testimg -File -ErrorAction SilentlyContinue).Count
if ($n -lt 2000) { Say "FATAL: C2A test images missing/incomplete ($n found)"; exit 1 }
Say "pre-flight OK (best.pt present, C2A test = $n imgs)"

# ---------- 3. run the three scripts sequentially ----------
# name, max attempts (the eval RESUMES from cache -> retries lose nothing)
$jobs = @(
    @{ script = "make_arch_figure_images.py";   tries = 2 },
    @{ script = "make_report_figure_images.py"; tries = 2 },
    @{ script = "sahi_tta_cbam_p2_thesis.py";   tries = 3 }
)
$results = @()
foreach ($j in $jobs) {
    $s = $j.script
    if (-not (Test-Path (Join-Path $ROOT $s))) {
        Say "SKIP $s -- file not found in $ROOT"
        $results += "$s : SKIPPED (missing)"
        continue
    }
    $ok = $false
    for ($a = 1; $a -le $j.tries; $a++) {
        $log = Join-Path $LOGS ("{0}_{1}_try{2}.log" -f $STAMP, ($s -replace "\.py$",""), $a)
        Say ">>> $s  (attempt $a/$($j.tries))  log: $log"
        & $PY -u $s *>&1 | Tee-Object -FilePath $log
        if ($LASTEXITCODE -eq 0) { $ok = $true; Say "<<< $s DONE"; break }
        Say "<<< $s FAILED (exit $LASTEXITCODE)"
        if ($a -lt $j.tries) { Say "retrying in 60 s (eval resumes from cache)"; Start-Sleep -Seconds 60 }
    }
    if ($ok) { $results += "$s : OK" } else { $results += "$s : FAILED after $($j.tries) attempts" }
}

# ---------- 4. summary ----------
try { [Sys.PW]::SetThreadExecutionState([uint32]"0x80000000") | Out-Null } catch {}
Say "=================== RUN SUMMARY ==================="
foreach ($r in $results) { Say $r }
Say "figure outputs : $ROOT\arch_fig_out  +  $ROOT\report_fig_out"
Say "eval outputs   : $ROOT\runs_sahi_tta\<newest>\metrics\grand_summary.md"
Say "logs           : $LOGS"
$results -join "`r`n" | Set-Content (Join-Path $LOGS "$STAMP`_SUMMARY.txt")

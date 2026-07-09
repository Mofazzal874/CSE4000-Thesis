<#
07_flush_gpu.ps1 -- Safe GPU flusher for the SHARED lab boxes (PC-2/PC-3).

Never kills blindly: default is a DRY RUN that lists every process holding the
GPU (with owner/start-time when visible). Killing requires the explicit -Kill
switch and only touches processes whose name matches -Name (default: python),
so the other user's session apps are never collateral damage.

Usage:
  .\07_flush_gpu.ps1 -Gpu 1              # dry run: who is holding GPU 1?
  .\07_flush_gpu.ps1 -Gpu 1 -Kill        # kill python processes on GPU 1
  .\07_flush_gpu.ps1 -Gpu 1 -Kill -Name ollama   # widen filter deliberately

If memory stays held with NO visible processes: it's a stale session or a
driver ghost -- see the printed guidance (sign out stale users / reboot,
AFTER coordinating on a shared box).
#>
param(
  [int]$Gpu = 1,
  [switch]$Kill,
  [string]$Name = "python"
)
$ErrorActionPreference = "Continue"

Write-Host "=== GPU $Gpu before ===" -ForegroundColor Cyan
& nvidia-smi -i $Gpu --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

# ---- collect processes on this GPU: compute-apps query first, WDDM table fallback
$procs = @()
$rows = & nvidia-smi -i $Gpu --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>$null
if ($rows) {
  foreach ($r in @($rows)) {
    $f = $r.Split(",") | ForEach-Object { $_.Trim() }
    if ($f.Count -ge 2 -and $f[0] -match '^\d+$') {
      $procs += [pscustomobject]@{ PID = [int]$f[0]; Proc = $f[1]; Mem = $f[-1]; Type = "C" }
    }
  }
}
if (-not $procs) {
  # WDDM fallback: parse the full nvidia-smi process table, keep rows for this GPU
  $inproc = $false
  foreach ($line in (& nvidia-smi)) {
    if ($line -match 'Processes:') { $inproc = $true; continue }
    if ($inproc -and $line -match '^\|\s+(\d+)\s+\S+\s+\S+\s+(\d+)\s+(\S+)\s+(.+?)\s+(\S+)\s*\|\s*$') {
      if ([int]$Matches[1] -eq $Gpu) {
        $procs += [pscustomobject]@{ PID = [int]$Matches[2]; Type = $Matches[3];
                                     Proc = $Matches[4].Trim(); Mem = $Matches[5] }
      }
    }
  }
}

if (-not $procs) {
  Write-Host "`nNo processes visible on GPU $Gpu, yet memory may still be held." -ForegroundColor Yellow
  Write-Host "That means a stale user session or a WDDM driver ghost. Options:"
  Write-Host "  1) 'query user'  -> look for a disconnected session, then 'logoff <ID>' (admin)."
  Write-Host "  2) Reboot the box -- ONLY after coordinating (shared machine!) and only if"
  Write-Host "     AnyDesk runs as a service (it does on the lab PCs), or you lose remote access."
  exit
}

Write-Host "`n=== processes on GPU $Gpu ===" -ForegroundColor Cyan
foreach ($p in $procs) {
  $gp = Get-Process -Id $p.PID -ErrorAction SilentlyContinue
  $who = ""
  try { $who = (Get-Process -Id $p.PID -IncludeUserName -ErrorAction Stop).UserName } catch {}
  $extra = if ($gp) { "exe=$($gp.ProcessName) started=$($gp.StartTime) user=$who" } else { "(already gone)" }
  "{0,-8} type={1,-4} mem={2,-10} {3}`n         path: {4}" -f $p.PID, $p.Type, $p.Mem, $extra, $p.Proc
}

if (-not $Kill) {
  Write-Host "`nDRY RUN -- nothing killed. Re-run with  -Kill  to terminate processes matching '*$Name*'." -ForegroundColor Yellow
  exit
}

$targets = $procs | Where-Object { $_.Proc -like "*$Name*" -or
                                   ((Get-Process -Id $_.PID -ErrorAction SilentlyContinue).ProcessName -like "*$Name*") }
if (-not $targets) {
  Write-Host "`nNothing on GPU $Gpu matches '*$Name*'. Refusing to kill unmatched processes" -ForegroundColor Yellow
  Write-Host "(pass -Name <substring> deliberately if you are SURE the process is disposable)."
  exit
}
foreach ($t in $targets) {
  Write-Host "KILLING PID $($t.PID)  ($($t.Proc))" -ForegroundColor Red
  Stop-Process -Id $t.PID -Force -ErrorAction Continue
}
Start-Sleep -Seconds 4
Write-Host "`n=== GPU $Gpu after ===" -ForegroundColor Cyan
& nvidia-smi -i $Gpu --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv

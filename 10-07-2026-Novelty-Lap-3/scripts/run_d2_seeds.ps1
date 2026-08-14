# run_d2_seeds.ps1 -- chain the 4 D2 significance seed runs + auto-eval, UNATTENDED + resume-safe.
#
# Launch (PC-1):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
#   cd "E:\Thesis_mofazzal_2007074\10-07-2026-Novelty-Lap-3\scripts"
#   .\run_d2_seeds.ps1
#
# It trains assign_s1 -> control_s1 -> assign_s2 -> control_s2 (each ~12h, one after another on the
# single GPU), evals each on the C2A scene-split test, and drops a `.d2seed_done` marker per finished run.
# POWER CUT / REBOOT: just launch it again. Done runs are SKIPPED; the interrupted one RESUMES (--resume).
# A run that errors is left unmarked and retried on the next launch; the chain moves on so the others still run.

$ErrorActionPreference = "Continue"
$scripts = "E:\Thesis_mofazzal_2007074\10-07-2026-Novelty-Lap-3\scripts"
$py      = "E:\Thesis_mofazzal_2007074\mofazzal1\Scripts\python.exe"    # venv python by full path (no activation needed)
$DATA    = "E:\Thesis_mofazzal_2007074\scenesplit_pc1.yaml"
$T       = "E:\Thesis_mofazzal_2007074\common\c2a\C2A_Dataset\new_dataset3_scenesplit_v1\test"
Set-Location $scripts

# assign first (matters most), then control; seed 1 before seed 2. After (1)+(2) you already have 2 seeds.
$runs = @(
  @{ name = "s3_assign_full_s1";  alpha = "0.5"; seed = 1 },
  @{ name = "s3_control_full_s1"; alpha = "0.0"; seed = 1 },
  @{ name = "s3_assign_full_s2";  alpha = "0.5"; seed = 2 },
  @{ name = "s3_control_full_s2"; alpha = "0.0"; seed = 2 }
)

foreach ($r in $runs) {
  $name   = $r.name
  $marker = Join-Path $scripts "runs_s1\$name\.d2seed_done"
  if (Test-Path $marker) { Write-Host "[chain] SKIP $name (already complete)"; continue }

  # SELF-HEAL: a power cut mid-write can corrupt last.pt. If it won't load, fall back to the newest
  # epoch*.pt snapshot (from --save-period 25) so --resume works; if none, drop it and restart the run.
  $wdir = Join-Path $scripts "runs_s1\$name\weights"
  $lastpt = Join-Path $wdir "last.pt"
  if (Test-Path $lastpt) {
    & $py -c "import torch,sys; torch.load(sys.argv[1],map_location='cpu',weights_only=False)" $lastpt 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
      $ep = Get-ChildItem (Join-Path $wdir "epoch*.pt") -ErrorAction SilentlyContinue |
            Sort-Object { [int]($_.BaseName -replace 'epoch','') } | Select-Object -Last 1
      if ($ep) {
        Write-Host "[chain] HEAL ${name}: last.pt corrupt -> resuming from $($ep.Name)"
        Copy-Item $lastpt "$lastpt.corrupt" -Force -ErrorAction SilentlyContinue
        Copy-Item $ep.FullName $lastpt -Force
      } else {
        Write-Host "[chain] HEAL ${name}: last.pt corrupt, no epoch snapshot -> restarting this run fresh"
        Remove-Item $lastpt -Force
      }
    }
  }

  Write-Host "[chain] ===== TRAIN $name (alpha=$($r.alpha) seed=$($r.seed))  $(Get-Date -Format u) ====="
  & $py 20_assign_patch.py --train --data $DATA --alpha $r.alpha --pretrained yolo11m.pt `
      --epochs 300 --patience 50 --name $name --seed $r.seed --batch 12 --device 0 `
      --reserve-gb 1 --save-period 25 --resume
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[chain] WARN: $name training exited $LASTEXITCODE -- will retry on next launch; moving on"
    continue
  }

  Write-Host "[chain] ===== EVAL $name  $(Get-Date -Format u) ====="
  & $py 18_s1_eval.py --variant control --tag $name --weights "runs_s1\$name\weights\best.pt" `
      --images-dir "$T\images" --gt-json "$T\test_annotations.json" --device 0
  if ($LASTEXITCODE -eq 0) {
    New-Item -ItemType File -Force $marker | Out-Null
    Write-Host "[chain] DONE $name (marked complete)  $(Get-Date -Format u)"
  } else {
    Write-Host "[chain] WARN: $name eval failed -- not marking; re-launch to retry the eval"
  }
}

Write-Host "[chain] ===== ALL 4 SEED RUNS PROCESSED. Next: bootstrap the seed pairs (see D2_SIGNIFICANCE.md). ====="

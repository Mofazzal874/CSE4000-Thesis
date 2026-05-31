<#
=====================================================================
 run_chain.ps1  --  sequential thesis training runner (AnyDesk PC)
=====================================================================
 Runs the thesis ablation scripts ONE AFTER ANOTHER on the single GPU.
 When step 1 finishes, step 2 picks up automatically. Each step is an
 independent full training (each fine-tunes from yolo11m.pt) -- so a
 failure in one step does NOT stop the next; every step's exit code is
 recorded and printed in a summary at the end.

 PRE-REQUISITES (do these once, manually, BEFORE running this):
   1. Smoke-test each script individually (set SMOKE_TEST=True, run, confirm
      PASS, set back to False). This runner does NOT smoke for you -- though
      each script will auto-smoke if its 24h marker is missing.
   2. Make sure the venv name + script paths below are correct for THIS PC.

 USAGE (from anywhere in PowerShell):
   cd "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs"
   .\run_chain.ps1

 If PowerShell blocks the script ("running scripts is disabled"):
   Set-ExecutionPolicy -Scope Process -Bypass
   then run .\run_chain.ps1 again.
=====================================================================
#>

# --- EDIT THESE 3 THINGS TO MATCH THIS PC --------------------------
$VenvActivate = "E:\Thesis_mofazzal_2007074\mofazzal1\Scripts\Activate.ps1"

# Ordered list of scripts to run, in sequence. Edit paths/order freely.
# (These are the AnyDesk-PC paths: parent "Benchmarking YOLOs", folders
#  "CBAM" and "CBAM_P2Head", filenames without the "v".)
$Steps = @(
    "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM\yolo11m_cbam_thesis.py",
    "E:\Thesis_mofazzal_2007074\Benchmarking YOLOs\CBAM_P2Head\yolo11m_cbam_p2head_thesis.py"
)
# -------------------------------------------------------------------

$ErrorActionPreference = "Continue"
$python = "python"

Write-Host ("=" * 70)
Write-Host "THESIS TRAINING CHAIN  --  $($Steps.Count) step(s), sequential, single GPU"
Write-Host ("=" * 70)

# 1) Activate the virtual environment (once)
if (Test-Path $VenvActivate) {
    Write-Host "[chain] activating venv: $VenvActivate"
    & $VenvActivate
} else {
    Write-Host "[chain] WARN: venv activate script not found at $VenvActivate"
    Write-Host "[chain] continuing with whatever 'python' is on PATH."
}

# Confirm which python we're using
$pyExe = (& $python -c "import sys; print(sys.executable)") 2>$null
Write-Host "[chain] python = $pyExe"
Write-Host ""

# 2) Run each step in order, recording results
$results = @()
$stepNum = 0
foreach ($script in $Steps) {
    $stepNum++
    $name = Split-Path $script -Leaf
    $folder = Split-Path $script -Parent
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    Write-Host ("-" * 70)
    Write-Host "[chain] STEP $stepNum/$($Steps.Count)  ::  $name"
    Write-Host "[chain] start: $stamp"
    Write-Host ("-" * 70)

    if (-not (Test-Path $script)) {
        Write-Host "[chain] ERROR: script not found -> $script  (skipping)"
        $results += [pscustomobject]@{ Step = $stepNum; Script = $name; ExitCode = "MISSING"; Start = $stamp; End = "-" }
        continue
    }

    # Per-step log next to the script. Tee shows live output AND saves a log.
    $logFile = Join-Path $folder ("chain_run_{0}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"))

    # Run by full path -- the script derives its own SCRIPT_DIR from __file__,
    # so the working directory does not matter for outputs/dataset probing.
    & $python $script *>&1 | Tee-Object -FilePath $logFile
    $code = $LASTEXITCODE

    $endStamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    if ($code -eq 0) {
        Write-Host "[chain] STEP $stepNum DONE (exit 0) at $endStamp  log: $logFile"
    } else {
        Write-Host "[chain] STEP $stepNum FAILED (exit $code) at $endStamp  log: $logFile"
        Write-Host "[chain] continuing to the next step anyway (each step is independent)."
    }
    $results += [pscustomobject]@{ Step = $stepNum; Script = $name; ExitCode = $code; Start = $stamp; End = $endStamp }
    Write-Host ""
}

# 3) Summary
Write-Host ("=" * 70)
Write-Host "[chain] ALL STEPS COMPLETE -- summary:"
Write-Host ("=" * 70)
$results | Format-Table -AutoSize
Write-Host "[chain] Done. Review the per-step logs and each run's runs/<run_id>/ output."

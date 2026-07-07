# =============================================================================
# REPAIR_VENV.ps1 -- fix the mofazzal1 venv whose base Python (C:\Program Files\
# Python311) was deleted. NO ADMIN NEEDED: uses the Python 3.11.9 EMBEDDABLE zip
# (a plain zip, no installer, never elevates). Re-points the venv at it.
# Preserves ALL installed packages (torch / ultralytics / etc.) -- nothing reinstalled.
#
# USE (VS Code PowerShell, from the CBAM_P2Head folder):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\REPAIR_VENV.ps1
# If it prints SUCCESS, run:  .\RUN_ALL_SAHI_TTA.ps1
# =============================================================================
$ErrorActionPreference = "Stop"
$VENV   = "E:\Thesis_mofazzal_2007074\mofazzal1"
$BASE   = "E:\Thesis_mofazzal_2007074\py311_base"      # embeddable lands here (user-writable, no admin)
$ZIPURL = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
$ZIP    = Join-Path $env:TEMP "py311_embed.zip"

function Say($m) { Write-Host ("[repair] " + $m) -ForegroundColor Cyan }

if (-not (Test-Path $VENV)) { Say "FATAL: venv not found at $VENV"; exit 1 }

# 1. fetch + extract the embeddable (no installer, no elevation)
if (-not (Test-Path (Join-Path $BASE "python.exe"))) {
    Say "downloading Python 3.11.9 embeddable (~10 MB, no admin) ..."
    [Net.ServicePointManager]::SecurityProtocol = 'Tls12'
    Invoke-WebRequest $ZIPURL -OutFile $ZIP
    if ((Get-Item $ZIP).Length -lt 5000000) { Say "download too small -- network issue, retry"; exit 1 }
    Say "extracting to $BASE ..."
    if (Test-Path $BASE) { Remove-Item $BASE -Recurse -Force }
    Expand-Archive $ZIP -DestinationPath $BASE -Force
} else { Say "embeddable already present at $BASE" }

# 2. disable the embeddable's ._pth so python uses normal (venv-aware) path + site resolution
Get-ChildItem $BASE -Filter "*._pth" -ErrorAction SilentlyContinue | ForEach-Object {
    Rename-Item $_.FullName ($_.FullName + ".disabled") -Force
    Say ("disabled " + $_.Name)
}

# 3. insurance: put the runtime DLLs next to the venv's python.exe too
foreach ($dll in @("python311.dll", "python3.dll", "vcruntime140.dll", "vcruntime140_1.dll")) {
    $src = Join-Path $BASE $dll
    if (Test-Path $src) { Copy-Item $src (Join-Path $VENV "Scripts") -Force }
}

# 4. re-point pyvenv.cfg at the embeddable base (backup first)
$cfg = Join-Path $VENV "pyvenv.cfg"
if (Test-Path $cfg) { Copy-Item $cfg ($cfg + ".bak") -Force }
@(
    "home = $BASE",
    "include-system-site-packages = false",
    "version = 3.11.9",
    "executable = $(Join-Path $BASE 'python.exe')",
    "command = $(Join-Path $BASE 'python.exe') -m venv $VENV"
) | Set-Content -Path $cfg -Encoding ascii
Say "pyvenv.cfg re-pointed to $BASE (backup: pyvenv.cfg.bak)"

# 5. verify the venv actually works now
$PY = Join-Path $VENV "Scripts\python.exe"
Say "verifying interpreter + packages ..."
& $PY -c "import sys; print('  python', sys.version.split()[0])"
& $PY -c "import torch, ultralytics; print('  torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| ultralytics', ultralytics.__version__)"
if ($LASTEXITCODE -eq 0) {
    Say "SUCCESS -- venv repaired, packages intact. Next:  .\RUN_ALL_SAHI_TTA.ps1"
} else {
    Say "EMBEDDABLE REPAIR did not fully work (torch import failed)."
    Say "Fall back to PC-4: it already has a working env + the CBAM+P2 model + C2A."
}

# PC-2 pre-run checklist — flush, launch, protect (use EVERY time before running on PC-2)

PC-2 = 2×A6000 SHARED, GPU 1 = ours, GPU 0 = peers (env `2007116` "waveletswt" + a `Python311`).
**Peers have deliberately `Stop-Process`-killed our runs 3+ times (2026-07-24).** This checklist
flushes safely, targets the right GPU, and applies every anti-starvation lever available without
admin. Nothing here stops a deliberate kill — see §5.

## 0. Golden rules (why things went wrong before)
- **Always `--device 1`** (NOT `--device 0`). Without `CUDA_VISIBLE_DEVICES=1` set in the window,
  `--device 0` = physical GPU 0 (the peers' busy GPU) → slow + collides. `--device 1` targets our
  free GPU 1 directly, no env var needed. (This was the root cause of the "slow + died" run.)
- **Ownership = venv path**, not nvidia-smi's process column. Ours = `...2007074\2007074\...python.exe`.
  On Windows/WDDM nvidia-smi mislabels which process is on which GPU — trust the PATH, and trust
  the training WINDOW advancing (not nvidia-smi's GPU-Util%, which is unreliable on WDDM).
- **Only ever kill our OWN venv's python.** Never touch `2007116`, `Python311`, or system procs
  (dwm/explorer/Excel/SearchUI). Never kill anything on GPU 0.

## 1. See what's running + who owns it
```powershell
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, StartTime, Path
nvidia-smi
```
Ours = the `2007074\2007074` path. Note its PID (evidence if killed again).

## 2. Flush GPU 1 — stop ONLY our own stale python (path-filtered = safe)
```powershell
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*2007074\2007074*" } | Stop-Process -Force
nvidia-smi -i 1
```
GPU 1 memory should drop to ~120 MB (just Windows). This kills only OUR leftovers (dead run's
main + lingering dataloader workers) — never peers', never system.

## 3. Launch on GPU 1 with RAM cache + reservation
```powershell
cd "D:\student_2k20\2007074\10-07-2026-Novelty-Lap-3\scripts"
python .\17_s1_pilot.py --variant fccg --data .\scenesplit_pc2.yaml --device 1 --reserve-gb 40 --resume
```
Confirm at startup: `[gpu-guard] CLAIMED dev1`, `CUDA:1`, `Caching images (X GB RAM)`, then the
epoch bar advancing. `--resume` continues from the last saved epoch (safe; falls back to fresh if
no checkpoint). Runner already defaults to `cache=ram` + `workers=4`.

## 4. Anti-starvation (two levers — the ceiling without admin)
- **RAM cache** (default) → no disk reads after startup → immune to peers hogging the disk.
- **CPU-priority boost** — after it starts, in a 2nd window:
```powershell
Get-Process python | Where-Object { $_.Path -like "*2007074\2007074*" } | ForEach-Object { $_.PriorityClass = 'High' }
```
Gives our training + dataloader higher CPU priority vs the peers' GPU-0 job. (nvidia-smi showing
GPU 1 at 0% util is usually WDDM noise OR data-starvation — judge by the WINDOW advancing, not util.)

## 5. The real fix (technical measures can't stop deliberate kills)
- **Escalate to supervisor** — peers force-killing our jobs is sabotage; screenshot their
  `Stop-Process` commands + our PID. This is the only true fix on a shared login.
- **Ask admin for exclusive mode:** `nvidia-smi -c EXCLUSIVE_PROCESS -i 1` (stops co-launching;
  does NOT stop a Task-Manager kill).
- **Better allocation:** run the CRITICAL experiment (FCCG) on the SAFE dedicated machine (PC-1),
  and the less-critical baseline on contested PC-2. A sabotaged baseline is cheap; a sabotaged
  headline result is not. Every run here saves `last.pt` every epoch → a kill costs ≤1 epoch,
  re-launch with §3 (`--resume`).

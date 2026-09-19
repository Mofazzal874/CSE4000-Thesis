"""Validate an explicit public boundary and produce the standalone pendrive copy."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / "public-showcase"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # No private source folder, model file or hidden credential file is allowed.
    allowed_dirs = {"analysis", "assets", "demos", "docs", "documents", "presentation", "results"}
    allowed_suffixes = {".md", ".svg", ".png", ".jpg", ".csv", ".json", ".txt", ".pdf", ".mp4", ".cff"}
    files = sorted(p for p in PUBLIC.rglob("*") if p.is_file())
    for p in files:
        rel = p.relative_to(PUBLIC)
        if p.is_symlink():
            raise ValueError(f"Symlink is not an approved artifact: {rel}")
        if len(rel.parts) > 1 and rel.parts[0] not in allowed_dirs:
            raise ValueError(f"Unexpected directory: {rel}")
        if p.suffix == ".py":
            if rel.as_posix() != "analysis/verify_results.py":
                raise ValueError(f"Unapproved code: {rel}")
        elif p.suffix not in allowed_suffixes and rel.as_posix() != ".gitignore":
            raise ValueError(f"Unapproved artifact: {rel}")
        if p.stat().st_size > 25 * 1024 * 1024:
            raise ValueError(f"Artifact exceeds the selected 25 MiB upload budget: {rel}")
        if p.suffix in {".md", ".csv", ".json", ".py", ".txt", ".cff"}:
            s = p.read_text(encoding="utf-8-sig")
            if re.search(r"(?<![A-Za-z])[A-Za-z]:[\\/]", s):
                raise ValueError(f"Absolute workstation path: {rel}")
            if re.search(r"(?:github_pat_|ghp_|sk-proj-)[A-Za-z0-9_]+", s):
                raise ValueError(f"Credential-like token: {rel}")

    numerical = sorted(p for p in (PUBLIC / "results").iterdir() if p.suffix in {".csv", ".json"})
    (PUBLIC / "results/SHA256SUMS.txt").write_text(
        "".join(f"{sha(p)}  {p.name}\n" for p in numerical), encoding="utf-8")
    # Keep the public companion canonical and adjust only paths for the research-root copy.
    readme = (PUBLIC / "README.md").read_text(encoding="utf-8")
    root_readme = re.sub(r"\]\((assets/|results/|docs/|demos/|documents/|presentation/|RELEASE_MANIFEST)",
                         r"](public-showcase/\1", readme)
    root_readme = root_readme.replace("python analysis/verify_results.py", "python public-showcase/analysis/verify_results.py")
    root_readme += """

## Research workspace navigation

The portable companion is the folder to share when detector source access is not intended. This research workspace also contains the implementation and working history; its repository visibility must be managed separately.

- [Professor guide](pendrive/README.md): complete local handover and portable showcase.
- [Project status](START_HERE.md) and [active lap](27-07-2026-Novelty-Lap-4/README.md).
- [Dataset map](DATA_MAP.md).
- [Showcase audit and release record](docs/2026-09-19_showcase_full_audit.md).

The pendrive links are local-workspace links; that folder is not tracked in the research repository.
"""
    (ROOT / "README.md").write_text(root_readme, encoding="utf-8")

    files = sorted(p for p in PUBLIC.rglob("*") if p.is_file() and p.name != "RELEASE_MANIFEST.json")
    manifest = {"version": "1.0.0", "prepared": "2026-09-19", "scope": "research companion, no detector code or model weights",
                "files": [{"path": p.relative_to(PUBLIC).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in files]}
    (PUBLIC / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, str(PUBLIC / "analysis/verify_results.py")], check=True)

    # Add a portable copy next to the full archive, preserving the hashed original archive.
    target = ROOT / "pendrive/PUBLIC_SHOWCASE"
    shutil.copytree(PUBLIC, target, dirs_exist_ok=True)
    subprocess.run([sys.executable, str(target / "analysis/verify_results.py")], check=True)
    zip_path = ROOT / "pendrive/PUBLIC_SHOWCASE_2007074.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for p in sorted(target.rglob("*")):
            if p.is_file():
                archive.write(p, "PUBLIC_SHOWCASE/" + p.relative_to(target).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP integrity failure: {bad}")
    print(f"Standalone release: {len(manifest['files']) + 1} files")
    print(f"ZIP: {zip_path.relative_to(ROOT)} ({zip_path.stat().st_size / 1048576:.2f} MiB)")


if __name__ == "__main__":
    main()

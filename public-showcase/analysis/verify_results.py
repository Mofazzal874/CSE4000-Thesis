"""Check the published result exports without the private detector implementation."""

import csv
import hashlib
import json
import re
from pathlib import Path
from statistics import mean, stdev
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def rows(name):
    with (RESULTS / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    listed = {r["path"] for r in manifest["files"]}
    actual_files = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
                    if p.is_file() and not any(part in {".git", ".venv", "__pycache__"}
                                               for part in p.relative_to(ROOT).parts)
                    and p.name != "RELEASE_MANIFEST.json"}
    require(actual_files == listed, "Release file inventory differs from manifest")
    for record in manifest["files"]:
        path = ROOT / record["path"]
        require(path.stat().st_size == record["bytes"], f"Size mismatch: {record['path']}")
        require(hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"],
                f"Release hash mismatch: {record['path']}")
    for document in ROOT.rglob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            linked = (document.parent / unquote(target.split("#")[0])).resolve()
            require(linked.is_relative_to(ROOT.resolve()), f"Link escapes companion: {target}")
            require(linked.exists(), f"Broken link in {document.name}: {target}")

    for line in (RESULTS / "SHA256SUMS.txt").read_text().splitlines():
        expected, name = line.split("  ", 1)
        actual = hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
        require(actual == expected, f"Hash mismatch: {name}")

    seeds = rows("tab_4_11b_three_seed_replication.csv")
    require([r["seed"] for r in seeds] == ["0", "1", "2"], "Unexpected seed set")
    deltas = [float(r["delta_very_tiny_pp"]) for r in seeds]
    require(round(mean(deltas), 2) == 1.62, "Unexpected seed mean")
    require(round(stdev(deltas), 2) == 0.49, "Unexpected seed sample SD")
    for row in seeds:
        computed = 100 * (float(row["assign_very_tiny"]) - float(row["control_very_tiny"]))
        require(abs(computed - float(row["delta_very_tiny_pp"])) < 0.011, "Seed delta mismatch")
    print(f"Very-tiny recall gain: {mean(deltas):.2f} +/- {stdev(deltas):.2f} percentage points (sample SD)")

    counts = rows("tab_4_5_per_size_recall.csv")
    for row in counts:
        for key in row:
            if key.endswith("_gt"):
                prefix = key[:-3]
                gt, matched = int(row[key]), int(row[prefix + "_matched"])
                require(0 <= matched <= gt, "Invalid matching count")
                require(abs(matched / gt - float(row[prefix + "_recall"])) <= 0.000051, "Recall mismatch")
    additional = int(counts[2]["very_tiny_lt8_matched"]) - int(counts[0]["very_tiny_lt8_matched"])
    require(additional == 369, "Unexpected additional instance count")
    print(f"Official-split additional very-tiny matches: {additional}")

    bootstrap = json.loads((RESULTS / "bootstrap_s3_control_full_vs_s3_assign_full.json").read_text())
    require(bootstrap["n_img"] == 2040 and bootstrap["n_boot"] == 1000, "Assignment bootstrap protocol mismatch")
    vt = bootstrap["results"]["recall_very_tiny"]
    require(vt["delta_pp"] == 2.08 and vt["ci95_pp"] == [1.77, 2.38], "Assignment interval mismatch")
    event = json.loads((RESULTS / "bootstrap_s3_xevent_vs_d3all_xevent.json").read_text())
    ap = event["results"]["AP50"]
    require(ap["delta_pp"] == 1.69 and ap["p"] == 0.468 and not ap["sig"], "Unseen-event summary mismatch")
    print("Archived bootstrap summaries: assignment +2.08 pp; unseen-event p=0.468 (not significant)")

    for case in rows("failure_and_comparison_cases.csv"):
        tp, fp, fn, gt = (int(case[k]) for k in ("rec_TP", "rec_FP", "rec_FN", "gt"))
        require(tp + fn == gt, "Gallery ground-truth count mismatch")
        require(abs(tp / gt - float(case["rec_recall"])) < 0.000001, "Gallery recall mismatch")
    for case in rows("gallery_overlay_counts.csv"):
        tp, fp, fn, gt = (int(case[k]) for k in ("TP", "FP", "FN", "gt"))
        require(tp + fn == gt, "Overlay ground-truth count mismatch")
        require(abs(tp / gt - float(case["recall"])) < 0.000001, "Overlay recall mismatch")
        require(abs(5 * tp / (5 * tp + 4 * fn + fp) - float(case["F2"])) < 0.000001,
                "Overlay F2 mismatch")

    stages = rows("tab_4_13b_adaptation_stages.csv")
    for domain in ("drone", "rset", "xevent", "c2a"):
        before = next(r for r in stages if r["domain"] == domain and r["stage"].startswith("un-adapted"))
        after = next(r for r in stages if r["domain"] == domain and r["stage"].startswith("all-real"))
        delta = 100 * (float(after["AP50"]) - float(before["AP50"]))
        print(f"{domain}: AP50 {float(before['AP50']):.4f} -> {float(after['AP50']):.4f} ({delta:+.2f} pp)")
    print("PASS: full release hashes, local links, gallery counts, result arithmetic, and archived summaries.")
    print("Training, inference, and bootstrap resampling were not rerun.")


if __name__ == "__main__":
    main()

"""Prepare an explicit selection of existing thesis artifacts for the public companion.

Does not run the detector or modify the original submission archive.
Requires the workstation's PyMuPDF, Pillow, matplotlib and imageio-ffmpeg.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess

import fitz
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "pendrive/THESIS_2007074_MSA-YOLO"
PUBLIC = ROOT / "public-showcase"
PROVENANCE = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, destination):
    src = ARCHIVE / source
    dst = PUBLIC / destination
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    PROVENANCE.append({"output": destination, "source": str(src.relative_to(ROOT)).replace("\\", "/"),
                       "source_sha256": digest(src), "operation": "byte-identical copy"})
    return dst


def main():
    copy("01_THESIS/2007074_MSA-YOLO_Thesis_Final.pdf", "documents/thesis.pdf")
    deck = copy("01_THESIS/presentation/2007074_Mofazzal_Disaster_human.pdf", "presentation/defense-presentation.pdf")
    # The PPTX contains embedded videos and is 94.7 MiB; the PDF provides all 43 pages.
    d = fitz.open(deck)
    for page_no in (1, 14, 18, 26, 28, 30):
        p = d[page_no - 1]
        p.get_pixmap(matrix=fitz.Matrix(1200 / p.rect.width, 1200 / p.rect.width), alpha=False).save(
            str(PUBLIC / "presentation" / f"slide-{page_no:02}.png"))

    cases = "04_OUTPUTS/04_failure_and_comparison_cases/"
    pairs = [
        ("04_improved_vs_baseline/01_collapsed_building_image0509_0", "improved"),
    ]
    for prefix, name in pairs:
        for model in ("baseline", "cbam_p2"):
            copy(cases + prefix + "__" + model + ".jpg", f"assets/examples/{name}-{model}.jpg")
    # Select the archived rank-1 regression pair without guessing the image stem.
    regression = sorted((ARCHIVE / cases / "05_regressed_vs_baseline").glob("01_*__cbam_p2.jpg"))
    if len(regression) != 1:
        raise ValueError("Expected one rank-1 regression case")
    for model in ("baseline", "cbam_p2"):
        src = regression[0].with_name(regression[0].name.replace("__cbam_p2", "__" + model))
        copy(str(src.relative_to(ARCHIVE)), f"assets/examples/regressed-{model}.jpg")
    copy(cases + "01_worst_recall/01_flood_image0087_1__cbam_p2.jpg", "assets/examples/missed-people.jpg")
    copy("05_NUMERICAL_RESULTS/failure_analysis/failure_and_comparison_cases.csv", "results/failure_and_comparison_cases.csv")
    for name in ("bootstrap_s3_control_full_vs_s3_assign_full.json", "bootstrap_s3_xevent_vs_d3all_xevent.json"):
        copy("05_NUMERICAL_RESULTS/02_extended_study_scenesplit/" + name, "results/" + name)

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    media = []
    for altitude, stem in ((10, "10m_cbam_p2_plain_conf0.40"), (50, "50m_cbam_p2_sahi_tta_conf0.40")):
        src = ARCHIVE / "04_OUTPUTS/03_demo_real_footage/annotated_videos" / (stem + ".mp4")
        dst = PUBLIC / "demos" / f"drone-{altitude}m.mp4"
        # Keep the complete clip and existing frame cadence. Re-encode only for sharing.
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                        "-map", "0:v:0", "-an", "-vf", "scale=1280:-2", "-c:v", "libx264",
                        "-preset", "medium", "-crf", "25", "-pix_fmt", "yuv420p",
                        "-map_metadata", "-1", "-movflags", "+faststart", str(dst)], check=True)
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-ss", "5", "-i", str(dst),
                        "-frames:v", "1", "-update", "1", str(PUBLIC / "demos" / f"drone-{altitude}m.jpg")], check=True)
        record = {"file": dst.name, "source_filename": src.name, "source_sha256": digest(src),
                  "altitude_m": altitude, "confidence": 0.4,
                  "inference": "plain" if altitude == 10 else "SAHI 256 / overlap 0.30 + TTA (archived label)",
                  "model": "YOLO11m + CBAM + P2 (original defense demo)",
                  "checkpoint_provenance": "filename token, annotate_video.py model map, and model archive; generating-run hash unavailable",
                  "transform": "complete clip; H.264 1280px width; source frame cadence retained; no audio; no inference rerun"}
        media.append(record)
        PROVENANCE.append({"output": str(dst.relative_to(PUBLIC)).replace("\\", "/"),
                           "source": str(src.relative_to(ROOT)).replace("\\", "/"),
                           "source_sha256": digest(src), "operation": record["transform"]})
    (PUBLIC / "demos/media.json").write_text(json.dumps(media, indent=2) + "\n", encoding="utf-8")
    private = ROOT / "tmp/showcase-audit/media-provenance.json"
    private.parent.mkdir(parents=True, exist_ok=True)
    private.write_text(json.dumps(PROVENANCE, indent=2) + "\n", encoding="utf-8")
    make_charts()
    print("Prepared documents, six slide previews, five qualitative images, two videos, and result charts.")


def make_charts():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    with (PUBLIC / "results/tab_4_11b_three_seed_replication.csv").open() as f:
        seeds = list(csv.DictReader(f))
    with (PUBLIC / "results/tab_4_13b_adaptation_stages.csv").open() as f:
        stages = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1, 1.35]})
    xs = list(range(3)); vals = [float(r["delta_very_tiny_pp"]) for r in seeds]
    axes[0].bar(xs, vals, color="#217e76", width=.55)
    axes[0].set(xticks=xs, xticklabels=["Seed 0", "Seed 1", "Seed 2"], ylim=(0, 2.7),
                ylabel="Change in recall (percentage points)", title="Sub-8-pixel recall: matched seed pairs")
    for x, v in zip(xs, vals): axes[0].text(x, v + .07, f"+{v:.2f}", ha="center", fontweight="bold")
    axes[0].text(.5, .94, "Mean +1.62; sample SD 0.49 pp", transform=axes[0].transAxes, ha="center", fontsize=10)
    domains = [("drone", "Drone"), ("rset", "Same-video\ndisaster"), ("xevent", "Unseen\nevent")]
    for offset, prefix, color, label in ((-.18, "un-adapted", "#a5b6c6", "Before adaptation"), (.18, "all-real", "#217e76", "All-real adaptation")):
        values = [next(float(r["AP50"]) for r in stages if r["domain"] == domain and r["stage"].startswith(prefix)) for domain, _ in domains]
        axes[1].bar([x + offset for x in xs], values, width=.34, color=color, label=label)
        for x, v in zip(xs, values): axes[1].text(x + offset, v + .018, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set(xticks=xs, xticklabels=[name for _, name in domains], ylim=(0, 1.08), ylabel="AP50", title="Adaptation gains depend on the domain")
    axes[1].legend(frameon=False, loc="upper left", fontsize=9)
    axes[1].text(2, .62, "Change not significant\np = 0.47", ha="center", fontsize=9, color="#6c5360")
    for ax in axes: ax.grid(axis="y", alpha=.15); ax.set_axisbelow(True)
    fig.text(.5, .018, "Scene-disjoint assignment study and held-out real-domain evaluations. Sources: Tables 4.11b and 4.13b.", ha="center", fontsize=9, color="#53606a")
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(PUBLIC / "assets/results-overview.png", dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""
demo_app.py -- Thesis-defense demo UI (runs OFFLINE on the laptop, no GPU).

Reads the precomputed predictions from <demo>/results/ and serves a local
Gradio app: side-by-side model comparison with a live confidence slider,
drone-shoot frames + annotated videos, and the report's result tables.

Launch (or just double-click launch_demo.bat in the demo folder):
    .venv\\Scripts\\python.exe app\\demo_app.py
Then the browser opens at http://127.0.0.1:7860 automatically.

The app never loads a neural network -- it draws cached boxes. If results/
is missing it still launches and says what to copy where.
"""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # fully offline

import cv2
import numpy as np
import gradio as gr

APP_DIR = Path(__file__).resolve().parent
DEMO_ROOT = APP_DIR.parent

# ----------------------------------------------------------- report tables
# Verbatim from Defense/draft1_30_6_26/chapters/04_implementation_results.tex
MAIN_ABLATION_MD = """
### Additive ablation — C2A test set, COCO protocol (report Table `tab:main`)
Latency measured end-to-end on RTX 4070 Ti SUPER.

| Model | Params (M) | GFLOPs | AP | AP50 | AP75 | AR100 | F1 | F2 | Lat. (ms) |
|---|---|---|---|---|---|---|---|---|---|
| YOLO11m baseline | 20.03 | 67.7 | **0.615** | 0.843 | 0.655 | 0.691 | **0.850** | 0.840 | 13.7 |
| + CBAM | 19.08 | 66.9 | **0.616** | 0.847 | 0.656 | 0.692 | **0.850** | 0.841 | **13.5** |
| **+ CBAM + P2 (recommended)** | 19.57 | 86.7 | 0.615 | **0.853** | 0.660 | 0.703 | 0.848 | **0.844** | 14.6 |
| + Mamba + CBAM + P2 | 22.01 | 98.4 | 0.614 | 0.852 | **0.662** | **0.704** | 0.846 | **0.844** | 41.1 |
"""

SAHI_TABLE_MD = """
### SAHI / TTA inference-time settings — CBAM+P2 model (report Table `tab:sahi`)
Per-image box matching at IoU 0.5; latency on the 4070 Ti SUPER.

| Setting | Size | P | R | F1 | F2 | very-tiny R | Lat. (ms) |
|---|---|---|---|---|---|---|---|
| no SAHI (baseline) | 640 | 0.857 | 0.835 | 0.845 | 0.839 | 0.758 | 15 |
| **+ SAHI (demo config)** | 256 | 0.853 | 0.846 | 0.850 | 0.848 | 0.788 | 162 |
| + SAHI | 320 | 0.861 | 0.844 | **0.852** | 0.847 | 0.782 | 113 |
| + SAHI | 512 | 0.864 | 0.837 | 0.850 | 0.842 | 0.763 | 66 |
| + SAHI | 640 | 0.864 | 0.833 | 0.848 | 0.839 | 0.756 | 54 |
| + TTA | 1280 | 0.774 | 0.877 | 0.822 | **0.854** | **0.850** | 60 |
"""


# ----------------------------------------------------------- data loading
def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


class Store:
    def __init__(self, results_dir):
        self.results = Path(results_dir)
        self.c2a = load_json(self.results / "predictions_c2a.json")
        self.drone = load_json(self.results / "predictions_drone.json")
        self.latency = load_json(self.results / "latency_summary.json")
        self.c2a_img_dir = DEMO_ROOT / "data" / "c2a_test" / "images"
        self.drone_img_dir = self.results / "drone_frames"
        self.videos = sorted((self.results / "annotated_videos").glob("*.mp4")) \
            if (self.results / "annotated_videos").is_dir() else []
        self.other_videos = sorted((self.results / "other_sources").glob("*.mp4")) \
            if (self.results / "other_sources").is_dir() else []
        # curation order: best 'rescue' first
        self.c2a_order = []
        hints = self.results / "curation_hints.csv"
        if hints.is_file() and self.c2a:
            with open(hints, newline="") as f:
                for row in csv.DictReader(f):
                    if row["file"] in self.c2a["images"]:
                        self.c2a_order.append(
                            (row["file"],
                             f"{row['file']}  (GT {row['gt_n']}, "
                             f"SAHI rescues +{row['rescue']})"))
        elif self.c2a:
            self.c2a_order = [(k, k) for k in sorted(self.c2a["images"])]

    def gpu(self):
        if self.latency:
            return self.latency.get("gpu", "GPU")
        return "GPU"


# ----------------------------------------------------------- drawing
def draw_panel(img_path, rec, cfg_tag, conf_thr, show_gt):
    """Return (annotated RGB image, caption)."""
    data = np.fromfile(str(img_path), np.uint8)          # unicode-safe read
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return None, f"image not found: {img_path.name}"
    h, w = img.shape[:2]
    th = max(2, round(min(w, h) / 350))
    if show_gt:
        for x1, y1, x2, y2 in rec.get("gt", []):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 140, 255), max(1, th - 1))          # orange (BGR)
    n = 0
    entry = rec.get(cfg_tag)
    if entry:
        for x1, y1, x2, y2, c in entry["boxes"]:
            if c < conf_thr:
                continue
            n += 1
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                          (40, 220, 0), th)                        # green (BGR)
    ms = entry["ms"] if entry else "?"
    gt_n = len(rec.get("gt", []))
    cap = f"{n} detections at conf >= {conf_thr:.2f}   |   {ms} ms/image (GPU)"
    if show_gt and "gt" in rec:
        cap += f"   |   GT humans: {gt_n} (orange)"
    if max(h, w) > 1600:                       # downscale AFTER drawing: the
        s = 1600 / max(h, w)                   # browser shows ~800px anyway and
        img = cv2.resize(img, (int(w * s), int(h * s)))  # PNG-encoding 4K is slow
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cap


def make_compare_fn(store, which):
    """which: 'c2a' or 'drone'."""
    def fn(fname, left_cfg, right_cfg, conf_thr, show_gt):
        data = store.c2a if which == "c2a" else store.drone
        img_dir = store.c2a_img_dir if which == "c2a" else store.drone_img_dir
        if not data or not fname or fname not in data["images"]:
            return None, "no data", None, "no data"
        rec = data["images"][fname]
        cfgs = data["meta"]["configs"]
        li, lc = draw_panel(img_dir / fname, rec, left_cfg, conf_thr,
                            show_gt and which == "c2a")
        ri, rc = draw_panel(img_dir / fname, rec, right_cfg, conf_thr,
                            show_gt and which == "c2a")
        return (li, f"**{cfgs.get(left_cfg, left_cfg)}** — {lc}",
                ri, f"**{cfgs.get(right_cfg, right_cfg)}** — {rc}")
    return fn


# ----------------------------------------------------------- live inference
def _nms(xyxy, scores, iou_thr=0.45, max_det=300):
    order = scores.argsort()[::-1][:3000]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if len(keep) >= max_det:
            break
        rest = order[1:]
        xx1 = np.maximum(xyxy[i, 0], xyxy[rest, 0])
        yy1 = np.maximum(xyxy[i, 1], xyxy[rest, 1])
        xx2 = np.minimum(xyxy[i, 2], xyxy[rest, 2])
        yy2 = np.minimum(xyxy[i, 3], xyxy[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a1 = (xyxy[i, 2] - xyxy[i, 0]) * (xyxy[i, 3] - xyxy[i, 1])
        a2 = (xyxy[rest, 2] - xyxy[rest, 0]) * (xyxy[rest, 3] - xyxy[rest, 1])
        iou = inter / np.clip(a1 + a2 - inter, 1e-9, None)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, int)


class LiveEngine:
    """CPU inference of the CBAM+P2 model via ONNX Runtime (no ultralytics,
    no torch, no CBAM registration needed -- the graph is self-contained)."""

    def __init__(self, onnx_path, imgsz=640):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.imgsz = imgsz
        self.infer_bgr(np.full((480, 640, 3), 114, np.uint8), 0.5)  # warmup

    def infer_bgr(self, bgr, conf_thr):
        t0 = time.perf_counter()
        h0, w0 = bgr.shape[:2]
        g = min(self.imgsz / h0, self.imgsz / w0)
        nw, nh = round(w0 * g), round(h0 * g)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh))
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.iname: np.ascontiguousarray(x)})[0][0]
        if out.shape[0] != 5 and out.shape[-1] == 5:   # (N,5) -> (5,N)
            out = out.T
        cxywh, scores = out[:4].T, out[4]
        keep = scores >= conf_thr
        cxywh, scores = cxywh[keep], scores[keep]
        xyxy = np.empty_like(cxywh)
        xyxy[:, 0] = cxywh[:, 0] - cxywh[:, 2] / 2
        xyxy[:, 1] = cxywh[:, 1] - cxywh[:, 3] / 2
        xyxy[:, 2] = cxywh[:, 0] + cxywh[:, 2] / 2
        xyxy[:, 3] = cxywh[:, 1] + cxywh[:, 3] / 2
        if len(xyxy):
            idx = _nms(xyxy, scores)
            xyxy, scores = xyxy[idx], scores[idx]
            xyxy[:, [0, 2]] = np.clip((xyxy[:, [0, 2]] - left) / g, 0, w0)
            xyxy[:, [1, 3]] = np.clip((xyxy[:, [1, 3]] - top) / g, 0, h0)
        ms = (time.perf_counter() - t0) * 1000
        return xyxy, scores, ms


def make_live_fn(engine):
    def run(img_rgb, conf_thr):
        if img_rgb is None:
            return None, "*upload an image (or use the random button) first*"
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        xyxy, scores, ms = engine.infer_bgr(bgr, conf_thr)
        h, w = bgr.shape[:2]
        th = max(2, round(min(w, h) / 350))
        for (x1, y1, x2, y2), s in zip(xyxy, scores):
            cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)),
                          (40, 220, 0), th)
        if max(h, w) > 1600:
            sc = 1600 / max(h, w)
            bgr = cv2.resize(bgr, (int(w * sc), int(h * sc)))
        cap = (f"**{len(xyxy)} detections** at conf >= {conf_thr:.2f}   |   "
               f"**{ms:.0f} ms — computed LIVE on this laptop's CPU** "
               f"(ONNX Runtime, YOLO11m+CBAM+P2 @640)")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cap
    return run


def step_fn(options):
    """prev/next over a dropdown's value list."""
    def step(cur, delta):
        vals = [v for v, _ in options] if options and isinstance(options[0], tuple) \
            else list(options)
        if not vals:
            return gr.update()
        try:
            i = vals.index(cur)
        except ValueError:
            i = 0
        return vals[(i + delta) % len(vals)]
    return step


# ----------------------------------------------------------- app
def build(store, engine=None):
    missing_msg = (
        "## ⚠ results not found\n"
        f"Expected `{store.results}` to contain `predictions_c2a.json` etc.\n\n"
        "Copy the **results** folder from the GPU PC into `Defense\\demo\\results\\` "
        "and restart this app (double-click `launch_demo.bat`)."
    )

    with gr.Blocks(title="Thesis Defense Demo — 2007074") as demo:
        gr.Markdown(
            "# Aerial Human Detection in Disaster Scenes\n"
            "**YOLO11m + CBAM + P2 detection head, with SAHI+TTA inference** — "
            "C2A test set & own drone footage. All boxes precomputed on "
            f"{store.gpu()}; this UI re-thresholds them live.")

        # ------------------------------------------------ TAB 1: C2A
        with gr.Tab("C2A test set"):
            if not store.c2a:
                gr.Markdown(missing_msg)
            else:
                cfgs = store.c2a["meta"]["configs"]          # tag -> label
                cfg_choices = [(lbl, tag) for tag, lbl in cfgs.items()]
                file_choices = [(lbl, val) for val, lbl in store.c2a_order]
                with gr.Row():
                    fsel = gr.Dropdown(choices=file_choices,
                                       value=file_choices[0][1],
                                       label="Test image (sorted: biggest SAHI rescue first)",
                                       scale=6)
                    prev_b = gr.Button("◀ Prev", scale=1)
                    next_b = gr.Button("Next ▶", scale=1)
                with gr.Row():
                    conf = gr.Slider(0.10, 0.90, value=0.25, step=0.01,
                                     label="Confidence threshold (live)", scale=4)
                    show_gt = gr.Checkbox(value=False,
                                          label="Show ground truth (orange)", scale=1)
                with gr.Row():
                    lcfg = gr.Dropdown(choices=cfg_choices, value="baseline",
                                       label="Left panel")
                    rcfg = gr.Dropdown(choices=cfg_choices,
                                       value="cbam_p2_sahi_tta", label="Right panel")
                with gr.Row():
                    with gr.Column():
                        lcap = gr.Markdown()
                        limg = gr.Image(label=None, show_label=False, height=560,
                                        format="jpeg")
                    with gr.Column():
                        rcap = gr.Markdown()
                        rimg = gr.Image(label=None, show_label=False, height=560,
                                        format="jpeg")

                cmp_fn = make_compare_fn(store, "c2a")
                ins = [fsel, lcfg, rcfg, conf, show_gt]
                outs = [limg, lcap, rimg, rcap]
                for comp in ins:
                    # sliders re-render on mouse RELEASE, not every drag tick
                    ev = comp.release if isinstance(comp, gr.Slider) and \
                        hasattr(comp, "release") else comp.change
                    ev(cmp_fn, ins, outs)
                stepper = step_fn([(v, l) for v, l in store.c2a_order])
                prev_b.click(lambda c: stepper(c, -1), fsel, fsel)
                next_b.click(lambda c: stepper(c, +1), fsel, fsel)
                demo.load(cmp_fn, ins, outs)

        # ------------------------------------------------ TAB 2: drone
        with gr.Tab("Drone shoot (own footage)"):
            if store.videos:
                gr.Markdown("### Annotated flight video (rendered on GPU, "
                            "boxes = model output)")
                vsel = gr.Dropdown(choices=[str(v) for v in store.videos],
                                   value=str(store.videos[0]),
                                   label="Rendered video")
                vid = gr.Video(value=str(store.videos[0]), height=520)
                vsel.change(lambda p: p, vsel, vid)
            else:
                gr.Markdown("*(no annotated videos found in "
                            "results/annotated_videos — frames below still work)*")
            if not store.drone:
                gr.Markdown(missing_msg)
            else:
                dcfgs = store.drone["meta"]["configs"]
                dcfg_choices = [(lbl, tag) for tag, lbl in dcfgs.items()]
                dfiles = sorted(store.drone["images"])
                gr.Markdown("### Frame-by-frame comparison (10m / 30m / 50m altitude)")
                with gr.Row():
                    dfsel = gr.Dropdown(choices=dfiles, value=dfiles[0],
                                        label="Drone frame", scale=6)
                    dprev = gr.Button("◀ Prev", scale=1)
                    dnext = gr.Button("Next ▶", scale=1)
                dconf = gr.Slider(0.10, 0.90, value=0.30, step=0.01,
                                  label="Confidence threshold (live)")
                with gr.Row():
                    dl = gr.Dropdown(choices=dcfg_choices, value="cbam_p2",
                                     label="Left panel")
                    dr = gr.Dropdown(choices=dcfg_choices,
                                     value="cbam_p2_sahi_tta", label="Right panel")
                with gr.Row():
                    with gr.Column():
                        dlcap = gr.Markdown()
                        dlimg = gr.Image(show_label=False, height=560, format="jpeg")
                    with gr.Column():
                        drcap = gr.Markdown()
                        drimg = gr.Image(show_label=False, height=560, format="jpeg")
                dcmp = make_compare_fn(store, "drone")
                dins = [dfsel, dl, dr, dconf, gr.State(False)]
                douts = [dlimg, dlcap, drimg, drcap]
                for comp in (dfsel, dl, dr, dconf):
                    ev = comp.release if isinstance(comp, gr.Slider) and \
                        hasattr(comp, "release") else comp.change
                    ev(dcmp, dins, douts)
                dstep = step_fn(dfiles)
                dprev.click(lambda c: dstep(c, -1), dfsel, dfsel)
                dnext.click(lambda c: dstep(c, +1), dfsel, dfsel)
                demo.load(dcmp, dins, douts)

        # ------------------------------------------------ TAB: other sources
        with gr.Tab("Other sources (news footage)"):
            gr.Markdown(
                "### Out-of-domain check: public disaster footage\n"
                "Same model + SAHI pipeline applied to news/YouTube disaster "
                "videos — a qualitative look at generalization beyond the "
                "training distribution.")
            if not store.other_videos:
                gr.Markdown("*(no videos yet — place annotated MP4s in "
                            "`results\\other_sources\\` and restart)*")
            else:
                osel = gr.Dropdown(
                    choices=[(v.stem, str(v)) for v in store.other_videos],
                    value=str(store.other_videos[0]), label="Video")
                ovid = gr.Video(value=str(store.other_videos[0]), height=520)
                osel.change(lambda p: p, osel, ovid)

        # ------------------------------------------------ TAB 3: live CPU
        with gr.Tab("Live inference (CPU)"):
            if engine is None:
                gr.Markdown("*(live tab disabled: `models/cbam_p2head.onnx` "
                            "not found or onnxruntime missing)*")
            else:
                gr.Markdown(
                    "**Nothing here is cached.** Every image is inferenced live "
                    "on this laptop's CPU (ONNX Runtime, no GPU). The C2A test "
                    "split has **2043** images; only 250 were precomputed for "
                    "the other tabs — pick any image at random and watch the "
                    "model run.")
                with gr.Row():
                    lin = gr.Image(type="numpy", sources=["upload", "webcam",
                                                          "clipboard"],
                                   label="Input image", height=380)
                with gr.Row():
                    lconf = gr.Slider(0.10, 0.90, value=0.30, step=0.01,
                                      label="Confidence threshold", scale=3)
                    rand_b = gr.Button("🎲 Random C2A test image", scale=1)
                    run_b = gr.Button("▶ Run live", variant="primary", scale=1)
                lcapt = gr.Markdown()
                lout = gr.Image(show_label=False, height=560, format="jpeg")

                live_fn = make_live_fn(engine)

                def rand_and_run(conf_thr):
                    files = list(store.c2a_img_dir.glob("*.*"))
                    if not files:
                        return None, None, "*no test images found*"
                    p = random.choice(files)
                    data = np.fromfile(str(p), np.uint8)
                    rgb = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR),
                                       cv2.COLOR_BGR2RGB)
                    out, cap = live_fn(rgb, conf_thr)
                    return rgb, out, cap + f"   |   image: `{p.name}`"

                run_b.click(live_fn, [lin, lconf], [lout, lcapt])
                lin.upload(live_fn, [lin, lconf], [lout, lcapt])
                rand_b.click(rand_and_run, [lconf], [lin, lout, lcapt])

        # ------------------------------------------------ TAB 4: results
        with gr.Tab("Results (from the report)"):
            gr.Markdown(MAIN_ABLATION_MD)
            gr.Markdown(SAHI_TABLE_MD)
            if store.latency:
                rows = "\n".join(
                    f"| {k} | {v.get('mean','-')} | {v.get('p50','-')} | "
                    f"{v.get('p95','-')} | {v.get('n','-')} |"
                    for k, v in store.latency["per_config_ms"].items())
                gr.Markdown(
                    f"### Demo precompute latency — measured on {store.gpu()}\n"
                    "| config | mean ms | p50 ms | p95 ms | n |\n|---|---|---|---|---|\n"
                    + rows)
            gr.Markdown(
                "**Scope note:** demo shows in-domain performance (C2A held-out "
                "test split) plus our own drone footage. Cross-domain transfer "
                "(SARD) is evaluated separately in the thesis and discussed as "
                "future work.")
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(DEMO_ROOT / "results"))
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    store = Store(args.results)
    engine = None
    onnx_path = DEMO_ROOT / "models" / "cbam_p2head.onnx"
    if onnx_path.is_file():
        try:
            engine = LiveEngine(onnx_path)
            print("live engine ready (ONNX Runtime CPU)")
        except Exception as e:
            print(f"live engine unavailable: {e}")
    demo = build(store, engine)
    demo.launch(server_name="127.0.0.1", server_port=args.port,
                inbrowser=not args.no_browser, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()

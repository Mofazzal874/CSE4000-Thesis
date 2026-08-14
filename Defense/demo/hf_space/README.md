---
title: MSA-YOLO Thesis Defense Demo
emoji: 🚁
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.20.0
app_file: app/demo_app.py
pinned: false
---

# MSA-YOLO — Tiny-Human Detection in Aerial Search-and-Rescue Imagery

Thesis-defense demo for **MSA-YOLO** (YOLO11m + CBAM + P2), CSE-4000, Dept. of CSE,
Khulna University of Engineering & Technology — Md Mofazzal Hosen (2007074),
supervised by Prof. Dr. Sk. Md. Masudul Ahsan.

**Tabs:**
- **Model comparison** — side-by-side cached predictions (baseline vs MSA-YOLO, plain vs
  SAHI+TTA) on C2A test scenes with a live confidence slider. No network runs here —
  boxes were precomputed on an RTX 4070 Ti SUPER.
- **Drone shoot** — annotated frames and videos from the author's own drone footage.
- **Live inference (CPU)** — upload an image/short video and the ONNX-exported MSA-YOLO
  annotates it live on the Space's CPU (plain 640-px inference; expect ~1 s per frame).
- **Results** — the report's headline tables.

**Headline result:** AP50 0.853 · F2 0.844 · 19.6 M params · 14.6 ms — second among all
published detectors on the C2A benchmark at one-third the size of the leader.

C2A dataset: R. A. Nihal et al., "UAV-Enhanced Combination to Application," ICPR 2024.
Sample images included here are a 250-image demo subset of the public C2A test split.

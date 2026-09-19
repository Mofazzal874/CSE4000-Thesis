# Drone demonstrations

Both complete clips are included in this package. Click a preview to play or download the MP4. These are recorded outputs, so viewing them requires no training code, model weights, Python environment, or GPU.

## 10 m — plain CBAM + P2

[![10 m annotated drone preview](drone-10m.jpg)](drone-10m.mp4)

**[Play the complete 10 m clip](drone-10m.mp4)** · about 33.83 seconds · confidence 0.40

## 50 m — tiled inference and TTA

[![50 m annotated drone preview](drone-50m.jpg)](drone-50m.mp4)

**[Play the complete 50 m clip](drone-50m.mp4)** · about 37.34 seconds · confidence 0.40 · SAHI tile 256, overlap 0.30, with archived TTA setting

## Provenance and interpretation

The archived filenames are `10m_cbam_p2_plain_conf0.40.mp4` and `50m_cbam_p2_sahi_tta_conf0.40.mp4`. The original renderer constructs filenames from its selected model token and inference flags. Its `cbam_p2` mapping points to the original CBAM+P2 architecture checkpoint, and the defense README records that model's source as the June 2 official-split training run.

This establishes the intended model lineage. A per-video generating-run checkpoint hash is unavailable, so the clips are not claimed as cryptographically verified checkpoint reproductions. They are **original defense demonstrations**, not outputs proving the later `d3_all` adaptation scores.

The sharing copies retain the entire archived duration and frame cadence. They are encoded as H.264 at 1280 × 720 with no audio and browser-friendly playback metadata. No detector was rerun, no detection was added or removed, and no playback-speed change was introduced during this conversion. The archived renderer may have processed a subset of original camera frames, so encoded frame rate is not model inference throughput.

Green boxes in these videos are predicted detections; they are not matched ground-truth labels. The two clips use different inference settings and do not form a controlled altitude comparison. Resizing for sharing can make small boxes less legible than in the 1920-pixel archived outputs.

For measured false positives and missed people, use the [failure gallery](../docs/GALLERY.md). For quantitative domain adaptation, use the [held-out results](../results/tab_4_13_final_model_all_heldout_sets.csv).

[Machine-readable media provenance](media.json) records source filenames, source SHA-256 hashes, settings, and conversion. Third-party disaster-news videos have not been copied into this companion.

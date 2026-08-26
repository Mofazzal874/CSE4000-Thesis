"""Recompose the PC-1 qualitative panel into a paper-ready figure.

The raw panel from 29_qualitative_panel.py is 2064x2244 with baked-in label strips whose
text clipped, and its drone cells are mostly empty grass. This script splits the four cells,
crops each ROW to the same informative window (so the before/after pair stays comparable),
tags the cells (a) to (d), and writes a compact JPEG for the two-column figure.

Crop windows are derived from where the drawn boxes actually are, measured per row:
  drone   box pixels span y 0-720   -> window y 0-780
  disaster box pixels span y 306-1071 -> window y 295-1075
Both windows are the full cell width (1032 px) and 780 px tall, so all four cells share one
aspect ratio and the grid is regular.
"""
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "fig_qualitative.png"

# (y0, y1, x0, x1) of each cell in the raw panel; rows split at the label strips
CELLS = [
    ("a", (0, 1080, 0, 1032)),        # drone, zero-shot
    ("b", (0, 1080, 1032, 2064)),     # drone, adapted
    ("c", (1122, 2202, 0, 1032)),     # disaster, zero-shot
    ("d", (1122, 2202, 1032, 2064)),  # disaster, adapted
]
WINDOW = {"a": (0, 780), "b": (0, 780), "c": (295, 1075), "d": (295, 1075)}
TAG_H = 34


def tag(cell, letter):
    """White chip with the sub-figure letter, top-left, matching IEEE (a)/(b) labeling."""
    out = cell.copy()
    cv2.rectangle(out, (0, 0), (74, TAG_H), (255, 255, 255), -1)
    cv2.rectangle(out, (0, 0), (74, TAG_H), (40, 40, 40), 1)
    cv2.putText(out, f"({letter})", (8, TAG_H - 9), cv2.FONT_HERSHEY_DUPLEX,
                0.85, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def main() -> int:
    panel = cv2.imread(str(SRC))
    if panel is None:
        print(f"FATAL: {SRC} not found"); return 1
    cells = []
    for letter, (y0, y1, x0, x1) in CELLS:
        c = panel[y0:y1, x0:x1]
        wy0, wy1 = WINDOW[letter]
        c = c[wy0:wy1, :]
        cells.append(tag(c, letter))
        print(f"cell ({letter}): {c.shape[1]}x{c.shape[0]}")

    gap = 10                                   # thin white gutter between cells
    def row(a, b):
        sep = np.full((a.shape[0], gap, 3), 255, dtype="uint8")
        return cv2.hconcat([a, sep, b])
    top, bot = row(cells[0], cells[1]), row(cells[2], cells[3])
    sep = np.full((gap, top.shape[1], 3), 255, dtype="uint8")
    grid = cv2.vconcat([top, sep, bot])

    target_w = 2000                            # ~294 dpi at 6.8 in print width
    scale = target_w / grid.shape[1]
    grid = cv2.resize(grid, (target_w, round(grid.shape[0] * scale)),
                      interpolation=cv2.INTER_AREA)
    out = HERE / "fig_qualitative_panel.jpg"
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 93])
    mb = out.stat().st_size / 1048576
    print(f"wrote {out.name}  {grid.shape[1]}x{grid.shape[0]}  "
          f"aspect {grid.shape[1]/grid.shape[0]:.2f}  {mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

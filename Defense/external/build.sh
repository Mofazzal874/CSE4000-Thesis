#!/usr/bin/env bash
# Local (WSL) build — exact replica of the Overleaf XeLaTeX pipeline.
# Usage:  bash build.sh
set -e
xelatex -interaction=nonstopmode -halt-on-error main.tex
biber main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
echo "Done -> main.pdf"

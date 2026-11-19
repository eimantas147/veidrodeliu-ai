#!/bin/sh
# Visas eksperimentu ciklas: duomenys -> modeliai -> paveikslai
set -e
python3 src/generate_data.py
python3 src/train_eval.py
python3 src/make_figures.py

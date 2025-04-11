#!/bin/bash

echo "Running collinear.py..."
python3 pos_processing/collinear.py

echo "Fixing .npy format..."
python3 tools/fix_npy_format.py

echo "Running superpointREF.py with config..."
python3 prediction/superpointREF.py \
  --config prediction/superpoint_config.json

echo "Pipeline complete."

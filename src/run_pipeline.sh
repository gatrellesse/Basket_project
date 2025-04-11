#!/bin/bash

# Activate virtual environment if needed
# source venv/bin/activate

echo "Running collinear.py..."
python3 pos_processing/collinear.py

echo "Running superpointREF.py with config..."
python3 prediction/superpointREF_Inter.py \
  --config prediction/superpoint_config.json

echo "Pipeline complete."

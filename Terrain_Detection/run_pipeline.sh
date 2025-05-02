#!/bin/bash

echo "Running collinear.py..."
python3 src/pos_processing/collinear.py \
  --config src/prediction/superpoint_config.json

echo "Running superpointREF.py with config..."
python3 src/prediction/superpointREF.py \
  --config src/prediction/superpoint_config.json

echo "Terrain Dection complete."

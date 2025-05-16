#!/bin/bash
# Script to run the enhanced meta-model inference with Gaussian kernel smoothing and GeoTIFF export

# Meta-model path
META_MODEL_PATH="/home/users/li1995/global_flood/FloodRisk-DL/terratorch/meta_learning_results_20250512_101047/meta_model_random_forest.joblib"

# Directories
OUTPUT_DIR="./meta_inference_enhanced_results"

# Domains to process (comma-separated)
DOMAINS="HOU001,HOU007,SF001,SF002"

# Rainfall levels to process (comma-separated, leave empty for all available levels)
RAINFALL_LEVELS=""

# Gaussian kernel parameters
WINDOW_SIZE=512
OVERLAP=128
SIGMA=0.5

# Run the inference script
python run_inference.py \
  --meta_model $META_MODEL_PATH \
  --output_dir $OUTPUT_DIR \
  --domains $DOMAINS \
  --rainfall_levels $RAINFALL_LEVELS \
  --window_size $WINDOW_SIZE \
  --overlap $OVERLAP \
  --sigma $SIGMA \
  --save_geotiffs

echo "Enhanced meta-model inference completed."

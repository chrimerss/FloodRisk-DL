#!/bin/bash

# Script to run the complete meta-learning pipeline
# Usage: ./run_pipeline.sh [train_domains] [test_domains] [model_type] [max_depth]

# Default parameters
TRAIN_DOMAINS=${1:-"HOU007,SF002,OKC001,LA001,AUS001,DAL002,ATL001"}
# 'HOU002', 'HOU003', 'HOU004', 'HOU005', 'HOU006',
    #              'AUS001','DAL001','OKC001','OKC002','LA001','SF001',
    #              'NYC001','ATL001','ATL002','ORL001','ORL002','MIA001'

    # ['HOU007', 'AUS002','SF002']
TEST_DOMAINS=${2:-"HOU002"}
MODEL_TYPE=${3:-"random_forest"}
MAX_DEPTH=${4:-30}

# Print parameters
echo "Running meta-learning pipeline with the following parameters:"
echo "Training domains: $TRAIN_DOMAINS"
echo "Testing domains: $TEST_DOMAINS"
echo "Model type: $MODEL_TYPE"
echo "Max depth: $MAX_DEPTH"
echo ""

# Run meta-learning
echo "Starting meta-learning process..."
python meta_model.py --train_domains="$TRAIN_DOMAINS" --test_domains="$TEST_DOMAINS" --model_type="$MODEL_TYPE" --max_depth="$MAX_DEPTH"

# Get the most recent results directory
RESULTS_DIR=$(ls -td ../meta_learning_results_* | head -1)
echo ""
echo "Meta-learning completed. Results saved to $RESULTS_DIR"

# Run analysis
echo ""
echo "Analyzing results..."
python analyze_results.py --results_dir="$RESULTS_DIR"

echo ""
echo "Pipeline completed successfully!"
echo "Review the analysis in $RESULTS_DIR/analysis/" 
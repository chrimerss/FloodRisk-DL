# Meta-Learning for Flood Prediction

This module implements a meta-learning approach for flood prediction by combining the outputs of multiple flood prediction models. The meta-learning model uses a decision tree or random forest to learn which model is most accurate for different terrain and rainfall conditions.

## Features

- Combines predictions from 7 different flood prediction models:
  - Bathtub model (physics-based)
  - ResNet models (RES50, RES101, RES152)
  - TerraTorch models (TINY, 100M, 600M)
- Uses additional features like DEM, slope, and rainfall
- Trains a decision tree or random forest to optimize prediction accuracy
- Evaluates using Jaccard score for different flood categories
- Generates visualizations comparing model performance
- Enhanced inference with Gaussian kernel smoothing for window boundary blending
- GeoTIFF export support for GIS integration

## Usage

### Training and Evaluation

```bash
# Train on default domains and test on others
python meta_model.py

# Train on specific domains
python meta_model.py --train_domains=HOU001,SF001,NYC002 --test_domains=HOU007,LA002

# Use random forest instead of decision tree
python meta_model.py --model_type=random_forest --max_depth=15
```

### Inference with Trained Meta-Model

```bash
# Basic inference
python run_inference.py

# Enhanced inference with Gaussian kernel smoothing and GeoTIFF export
python run_inference.py --window_size=512 --overlap=128 --sigma=0.5 --save_geotiffs

# Or use the provided shell script
bash run_enhanced_inference.sh
```

### API Usage

```python
from terratorch.meta_learning.meta_model import MetaLearningModel

# Initialize the meta-learning model
meta_model = MetaLearningModel(train_domains=["HOU001", "SF001", "NYC002"])

# Load base models
meta_model.load_base_models()

# Prepare training data
meta_model.prepare_training_data()

# Train meta-model
meta_model.train_meta_model(model_type='decision_tree', max_depth=10)

# Evaluate on test domains
results = meta_model.evaluate(test_domains=["HOU007", "LA002", "DAL002"])

# Make prediction for a specific domain and rainfall level
predictions = meta_model.predict(domain="HOU007", rainfall_level="121mm")
```

## How It Works

1. **Data Collection**: For each domain and rainfall level, predictions are collected from all base models.

2. **Feature Extraction**: For each pixel, features are extracted including:
   - Terrain features (DEM elevation, slope)
   - Rainfall value
   - Predictions from each model (both class predictions and probabilities)

3. **Meta-Model Training**: A decision tree or random forest is trained to predict the correct flood category based on these features.

4. **Evaluation**: The meta-model is evaluated on test domains and compared with base models using the Jaccard score.

## Results

The meta-learning model often outperforms individual models by learning which model to trust in different conditions. Performance improvements are especially notable in:

- Accurately detecting the extent of flooding (binary classification)
- Improved accuracy in specific flood categories 
- Better generalization to unseen domains with different terrain characteristics

Results are saved in the output directory, including:
- Evaluation metrics in JSON format
- Visualizations comparing model predictions
- The trained meta-model (.joblib file)
- Training data (features and labels)
- GeoTIFF files for compatibility with GIS software (when using enhanced inference)

## Enhanced Inference

The enhanced inference script (`run_inference.py`) includes several improvements:

1. **Gaussian Kernel Smoothing**: Applies a Gaussian weight mask when processing prediction windows to create smooth transitions at window boundaries, eliminating visible seams in the output.

2. **Overlapping Windows**: Processes the input data with configurable overlap between adjacent windows for better boundary handling.

3. **GeoTIFF Export**: Saves prediction results as GeoTIFF files with proper geospatial metadata for integration with GIS software like QGIS or ArcGIS.

Key parameters for enhanced inference:

- `--window_size`: Size of the processing window (default: 512)
- `--overlap`: Overlap between adjacent windows in pixels (default: 128)
- `--sigma`: Standard deviation for the Gaussian kernel (default: 0.5)
- `--save_geotiffs`: Save predictions as GeoTIFF files
- `--no_save_geotiffs`: Disable GeoTIFF export
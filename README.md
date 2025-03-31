# FloodRisk-DL

A deep learning project for flood risk prediction using Swin Transformer, PyTorch Lightning, and Weights & Biases.

## Project Overview

This project uses a Swin Transformer model to predict maximum flood depth based on Digital Elevation Model (DEM) and rainfall data. It leverages:
- PyTorch Lightning for training
- Weights & Biases for experiment tracking
- Hydra for configuration management
- H5 datasets for efficient data storage and access

## Project Structure

```
FloodRisk-DL/
├── configs/                    # Configuration files
│   └── model_config.yaml       # Main configuration
├── notebooks/                  # Jupyter notebooks for experiments
├── src/                        # Source code
│   ├── data/                   # Data handling
│   │   └── flood_data_module.py # PyTorch Lightning DataModule for H5 data
│   ├── data_preparation/       # Scripts for data preparation
│   │   ├── prepare_dataset.py  # Script to create H5 dataset
│   │   └── explore_dataset.py  # Script to explore and visualize the dataset
│   ├── models/                 # Model definitions
│   │   └── flood_model.py      # Swin Transformer model for flood prediction
│   ├── utils/                  # Utility functions
│   │   └── visualization.py    # Visualization utilities
│   └── train_flood_model.py    # Main training script
└── requirements.txt            # Python dependencies
```

## Data Preparation

Before training, you need to prepare the data:

1. Your input data should be organized as follows:
   ```
   dataset/
   ├── [City ID]/
   │   ├── [City ID]_100-yr_WaterDepth_[timestep].tif
   │   ├── [City ID]_50-yr_WaterDepth_[timestep].tif
   │   ├── [City ID]_25-yr_WaterDepth_[timestep].tif
   │   ├── [City ID]_10-yr_WaterDepth_[timestep].tif
   │   └── [City ID]_DEM.tif
   └── ...
   ```

2. Run the data preparation script:
   ```bash
   cd src/data_preparation
   python prepare_dataset.py
   ```

3. Explore the created dataset:
   ```bash
   python explore_dataset.py
   ```

See `src/data_preparation/README.md` for detailed instructions.

## Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/FloodRisk-DL.git
cd FloodRisk-DL
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up Weights & Biases
```bash
wandb login
```

## Usage

### Training

After preparing the H5 dataset, update the dataset path in the configuration:

```bash
python src/train_flood_model.py data.h5_file=/path/to/flood_data.h5
```

To modify other configurations:

```bash
python src/train_flood_model.py data.batch_size=16 training.max_epochs=50
```

## Model Architecture

The model uses a Swin Transformer as the backbone feature extractor, followed by a custom regression head:

1. **Input**: A 2-channel image where:
   - Channel 1: DEM (elevation data)
   - Channel 2: Uniform rainfall value (the same value across the entire channel)

2. **Backbone**: Swin Transformer extracts features from the input

3. **Regression Head**: Custom layers that predict the maximum flood depth at each pixel

4. **Output**: A 1-channel image representing the predicted maximum flood depth

## Evaluation Metrics

The model is evaluated using:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score

## Future Work

- Implement advanced data augmentation techniques
- Experiment with different model architectures (UNet, DeepLabV3, etc.)
- Add post-processing for predictions
- Integrate uncertainty quantification
- Extend to regression with temporal predictions
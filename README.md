# FloodRisk-DL

A deep learning project for flood risk prediction using Swin Transformer, PyTorch Lightning, and Weights & Biases.

## Project Overview

This project uses a Swin Transformer model to predict flood risk based on rainfall and Digital Elevation Model (DEM) data. It leverages:
- PyTorch Lightning for training
- Weights & Biases for experiment tracking
- Hydra for configuration management

## Project Structure

```
FloodRisk-DL/
├── configs/               # Configuration files
│   └── model_config.yaml  # Main configuration
├── notebooks/             # Jupyter notebooks for experiments
├── src/                   # Source code
│   ├── data/              # Data handling
│   │   └── data_module.py # PyTorch Lightning DataModule
│   ├── models/            # Model definitions
│   │   └── swin_transformer.py # Swin Transformer model
│   ├── utils/             # Utility functions
│   │   └── visualization.py # Visualization utilities
│   └── train.py           # Main training script
└── requirements.txt       # Python dependencies
```

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

To train the model with default settings:

```bash
cd src
python train.py
```

To modify configurations:

```bash
python train.py data.batch_size=32 training.max_epochs=50
```

### Customizing Data

Currently, the project uses placeholder data. To use your actual data:
1. Place your rainfall and DEM data in a suitable directory
2. Modify `src/data/data_module.py` to load your specific data format
3. Update `configs/model_config.yaml` to point to your data directory

## Data Format

The expected data format is:
- Rainfall data: Raster images representing rainfall intensity
- DEM data: Raster images representing elevation
- Labels: Binary flood/no-flood indicators

## Future Work

- Implement data preprocessing pipeline for rainfall and DEM data
- Add more advanced augmentation techniques
- Experiment with different model architectures
- Add spatial validation strategies
- Implement post-processing for predictions
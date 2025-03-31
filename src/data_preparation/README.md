# Flood Prediction Data Preparation

This directory contains scripts for preparing flood prediction data from raw TIF files into an H5 dataset for model training.

## Data Structure

The expected input data structure is:

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

The output H5 dataset will have the following structure:

```
flood_data.h5
├── train/
│   ├── [City ID]/
│   │   ├── dem (dataset)
│   │   ├── 100-yr/
│   │   │   ├── max_flood_depth (dataset)
│   │   │   └── rainfall_value (attribute)
│   │   ├── 50-yr/...
│   │   ├── 25-yr/...
│   │   └── 10-yr/...
│   └── ...
├── val/...
└── test/...
```

## Scripts

### 1. `prepare_dataset.py`

This script processes the raw TIF files and creates an H5 dataset with train/val/test splits.

Usage:
```bash
python prepare_dataset.py
```

You can modify the script to customize:
- The path to the input dataset
- The path to save the H5 file
- The city IDs to use for testing
- The validation split ratio

### 2. `explore_dataset.py`

This script allows you to explore and visualize the created H5 dataset.

Usage:
```bash
# Explore the structure of the H5 file
python explore_dataset.py --h5_file flood_data.h5 --explore

# Count the number of samples in each split
python explore_dataset.py --h5_file flood_data.h5 --count

# Visualize a specific sample
python explore_dataset.py --h5_file flood_data.h5 --visualize --split test --city HOU001 --rainfall 100-yr

# Do all of the above with default parameters
python explore_dataset.py --h5_file flood_data.h5
```

## Creating the Dataset

1. Ensure you have the required dependencies installed:
   ```bash
   pip install h5py rasterio numpy tqdm matplotlib
   ```

2. Update the paths in `prepare_dataset.py` to match your data location.

3. By default, the script will:
   - Use `HOU001`, `LA001`, and `NYC001` as test cities
   - Use 10% of the remaining cities for validation
   - Use the rest for training
   - Extract the maximum flood depth across all time steps

4. Run the script:
   ```bash
   python prepare_dataset.py
   ```

5. Explore the created dataset:
   ```bash
   python explore_dataset.py
   ```

## Using the Dataset for Training

After creating the H5 dataset, update the `h5_file` path in the `configs/model_config.yaml` file and run the training script:

```bash
python src/train_flood_model.py
``` 
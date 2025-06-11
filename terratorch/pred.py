#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import terratorch
import rasterio
from rasterio.windows import Window
from pathlib import Path
from enum import Enum
from datetime import datetime
import random
from sklearn.metrics import jaccard_score, f1_score
import seaborn as sns
from collections import defaultdict
import time
import json
import argparse

from terratorch.tasks import SemanticSegmentationTask
from bathtub.bathtub import simple_bathtub_with_rainfall_robust

# Define buffer size for input data
BUFFER = 10

# Import model arguments from task_class.py
from task_class import model_args_res50, model_args_res101, model_args_res152, model_args_tiny, model_args_100, model_args_300, model_args_600

# Import the model paths
from model_pth import FloodCategory as ModelPaths

# Define flood categories
class FloodCategory(Enum):
    """Enumeration of flood categories based on max flood depth."""
    NO_FLOOD = 0      # max_flood_depth < 0.1
    NUISANCE = 1      # 0.1 <= max_flood_depth < 0.2
    MINOR = 2         # 0.2 <= max_flood_depth < 0.3
    MEDIUM = 3        # 0.3 <= max_flood_depth < 0.5
    MAJOR = 4         # 0.5 <= max_flood_depth < 1.0

# Colors for visualization
FLOOD_COLORS = {
    FloodCategory.NO_FLOOD: '#FFFFFF',   # White
    FloodCategory.NUISANCE: '#F8DCD9',   # Light pink
    FloodCategory.MINOR: '#E198B5',      # Medium light pink
    FloodCategory.MEDIUM: '#AA5FA5',     # Medium purple
    FloodCategory.MAJOR: '#5B3794',      # Dark purple
}

# Available rainfall levels
# RAINFALL_LEVELS = ["181mm", "162mm", "138mm", "123mm", "110mm", "98mm", "82mm", "70mm", "57mm", "48mm"]

with open("/home/users/li1995/global_flood/FloodBench/data/cities_rainfall.json", "r") as f:
        RAINFALL_DICT = json.load(f)

# 
    # {
    #     "City ID": "AUS001",
    #     "Desc": "Austin d01",
    #     "100-yr": "121 mm",
    #     "50-yr": "106 mm",
    #     "25-yr": "93 mm",
    #     "10-yr": "76 mm"
    # },

def extract_rainfall_levels(city_id, data, freqs=None):
    # Define the preferred order of return periods
    return_periods_ordered = [
        "1000-yr", "500-yr", "200-yr", "100-yr", "50-yr",
        "25-yr", "10-yr", "5-yr", "2-yr", "1-yr"
    ]
    if freqs:
        return_periods_ordered = freqs

    
    for entry in data:
        if entry["City ID"] == city_id:
            # Filter and order the rainfall values based on return periods
            rainfall_levels = [
                entry[rp].replace(" ", "") for rp in return_periods_ordered if rp in entry
            ]
            return rainfall_levels
    return []

# Function to classify flood depths into categories
def classify_depths(src):
    """Classify flood depths into categories."""
    categories = np.zeros_like(src, dtype=np.int64)
    categories = np.where(src >= 0.1, 1, categories)
    categories = np.where(src >= 0.2, 2, categories)
    categories = np.where(src >= 0.5, 3, categories)
    categories = np.where(src >= 1.0, 4, categories)    
    return categories

# Function to calculate slope from DEM
def calc_slope(src):
    """Calculate slope from DEM."""
    dx, dy = np.gradient(src)
    slope_rad = np.arctan(np.sqrt((dx**2 + dy**2)))
    slope_deg = np.degrees(slope_rad)
    return slope_deg

# Function to count pixels in each category
def count_pixels_by_category(prediction):
    """Count the number of pixels in each flood category."""
    pixel_counts = {}
    total_pixels = prediction.size
    
    for category in range(5):  # 0 to 4 for the flood categories
        count = np.sum(prediction == category)
        pixel_counts[category] = count
        
    # Calculate flood vs no-flood counts
    pixel_counts['no_flood'] = pixel_counts[0]  # Category 0
    pixel_counts['flood'] = total_pixels - pixel_counts[0]  # All except category 0
    
    return pixel_counts, total_pixels

# Function to load a model from a checkpoint
def load_model(checkpoint_path, model_type):
    """Load a model from checkpoint using the appropriate model arguments.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        model_type: Type of model ('TINY', '100M', '300M', or '600M')
    """
    # Select the appropriate model arguments based on model type
    if model_type == 'RES50':
        model_args= model_args_res50
    elif model_type == 'RES101':
        model_args= model_args_res101
    elif model_type == 'RES152':
        model_args= model_args_res152
    elif model_type == 'TINY':
        model_args = model_args_tiny
    elif model_type == '100M':
        model_args = model_args_100
    elif model_type == '300M':
        model_args = model_args_300
    elif model_type == '600M':
        model_args = model_args_600
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(f"Loading {model_type} model with backbone: {model_args['backbone']}")
    
    # Set up class information
    class_names = ["No Flood", "Nuisance Flood (0.1-0.2m]", "Minor Flood (0.2-0.4m]", "Moderate Flood (0.4-1.0m]", "Major Flood (>1.0m]"]
    class_weights = [0.5, 1.0, 1.0, 1.0, 1.0]
    
    # Create a temporary model to load from checkpoint
    if model_type.startswith('RES'):
        temp_model= SemanticSegmentationTask(
            model_args=model_args,
            model_factory="SMPModelFactory",
            loss="ce",
            lr=1e-4,
            ignore_index=-1,
            optimizer="AdamW",
            optimizer_hparams={"weight_decay": 0.05},
            freeze_backbone=False,
            class_names=class_names,
            class_weights=class_weights
        )
    else:
        temp_model = SemanticSegmentationTask(
            model_args=model_args,
            model_factory="EncoderDecoderFactory",
            loss="ce",
            optimizer="AdamW",
            optimizer_hparams={"weight_decay": 0.05},
            class_names=class_names,
            class_weights=class_weights,
            lr=1e-4,
            ignore_index=-1,
            freeze_backbone=False,
            plot_on_val=True
        )
    
    # Load the actual model from checkpoint
    model = SemanticSegmentationTask.load_from_checkpoint(
        checkpoint_path,
        model_factory=temp_model.hparams.model_factory,
        model_args=temp_model.hparams.model_args,
    )
    
    model.eval()  # Set to evaluation mode
    return model

# Function to prepare input data for a specific window
def prepare_data_window(domain, rainfall_level, window):
    """Prepare input data for model inference for a specific window.
    
    Args:
        domain: The domain to use (e.g., 'HOU001')
        rainfall_level: The rainfall level to use (e.g., '98mm')
        window: The rasterio.windows.Window object specifying the region to read
        
    Returns:
        dem, slope, rainfall, flood_cat, target
    """
    input_dir = f'/home/users/li1995/global_flood/UrbanFloods2D/dataset/{domain}'
    dem_input = os.path.join(input_dir, f"{domain}_DEM.tif")
    input_file = f'/home/users/li1995/global_flood/UrbanFloods2D/sample/{domain}_{rainfall_level}_max.tif'
    
    # Load DEM and calculate slope
    with rasterio.open(dem_input) as src:
        dem = src.read(window=window).squeeze().astype(np.float32)
        slope = calc_slope(dem)
        
        # Apply buffer to both DEM and slope - slicing with [BUFFER:-BUFFER, BUFFER:-BUFFER]
        # Only apply buffer if the arrays are large enough
        if dem.shape[0] > 2*BUFFER and dem.shape[1] > 2*BUFFER:
            dem = dem[BUFFER:-BUFFER, BUFFER:-BUFFER]
            slope = slope[BUFFER:-BUFFER, BUFFER:-BUFFER]
            
        dem = (dem - dem.mean()) / dem.std()  # Normalize
    
    # Load target flood depths
    with rasterio.open(input_file) as src:    
        target = src.read(window=window).squeeze().astype(np.float32)
        flood_cat = classify_depths(target)
        
        # Apply the same buffer to the target and flood_cat
        if target.shape[0] > 2*BUFFER and target.shape[1] > 2*BUFFER:
            target = target[BUFFER:-BUFFER, BUFFER:-BUFFER]
            flood_cat = flood_cat[BUFFER:-BUFFER, BUFFER:-BUFFER]
    
    # Prepare rainfall input (constant value across the image)
    rainfall = np.ones_like(dem) * int(rainfall_level.split('mm')[0])/1000.
    
    return dem, slope, rainfall, flood_cat, target

# Function to load the entire dataset
def load_full_dataset(domain, rainfall_level, crop_size=512):
    """Load the entire DEM and flood data for a domain and rainfall level.
    
    Args:
        domain: The domain to use (e.g., 'HOU001')
        rainfall_level: The rainfall level to use (e.g., '98mm')
        crop_size: Size of the window to use for inference (default: 512)
        
    Returns:
        dem, slope, rainfall, flood_cat, target (all sliced to multiples of crop_size), dem_mean, dem_std
    """
    input_dir = f'/home/users/li1995/global_flood/UrbanFloods2D/dataset/{domain}'
    dem_input = os.path.join(input_dir, f"{domain}_DEM.tif")
    input_file = f'/home/users/li1995/global_flood/UrbanFloods2D/sample/{domain}_{rainfall_level}_max.tif'
    
    # Load DEM and calculate slope for the entire domain
    with rasterio.open(dem_input) as src:
        dem = src.read(1).astype(np.float32)
        
    # Calculate slope from DEM
    slope = calc_slope(dem)
    
    # Load target flood depths
    with rasterio.open(input_file) as src:
        target = src.read(1).astype(np.float32)
    
    # Classify target depths into categories
    flood_cat = classify_depths(target)
    
    # Prepare rainfall input (constant value across the image)
    rainfall = np.ones_like(dem) * int(rainfall_level.split('mm')[0])/1000.
    
    # Apply buffer to all arrays if they are large enough
    if dem.shape[0] > 2*BUFFER and dem.shape[1] > 2*BUFFER:
        dem = dem[BUFFER:-BUFFER, BUFFER:-BUFFER]
        slope = slope[BUFFER:-BUFFER, BUFFER:-BUFFER]
        target = target[BUFFER:-BUFFER, BUFFER:-BUFFER]
        flood_cat = flood_cat[BUFFER:-BUFFER, BUFFER:-BUFFER]
        rainfall = rainfall[BUFFER:-BUFFER, BUFFER:-BUFFER]
    
    # Store the original DEM mean and std before normalizing
    dem_mean = dem.mean()
    dem_std = dem.std()
    
    # Normalize DEM
    dem = (dem - dem_mean) / dem_std
    
    # Calculate how many complete crop_size blocks fit in the dimensions
    height, width = dem.shape
    complete_height = (height // crop_size) * crop_size
    complete_width = (width // crop_size) * crop_size
    
    # Slice arrays to ensure dimensions are multiples of crop_size
    if complete_height < height or complete_width < width:
        print(f"  Slicing arrays to multiple of {crop_size}: from {height}x{width} to {complete_height}x{complete_width}")
        dem = dem[:complete_height, :complete_width]
        slope = slope[:complete_height, :complete_width]
        target = target[:complete_height, :complete_width]
        flood_cat = flood_cat[:complete_height, :complete_width]
        rainfall = rainfall[:complete_height, :complete_width]
    
    return dem, slope, rainfall, flood_cat, target, dem_mean, dem_std

# Function to get the full extent of an image
def get_image_dimensions(domain):
    """Get the dimensions of the DEM image.
    
    Args:
        domain: The domain to use (e.g., 'HOU001')
        
    Returns:
        height, width of the image
    """
    input_dir = f'/home/users/li1995/global_flood/UrbanFloods2D/dataset/{domain}'
    dem_input = os.path.join(input_dir, f"{domain}_DEM.tif")
    
    with rasterio.open(dem_input) as src:
        height = src.height
        width = src.width
    
    return height, width

# Function to calculate Jaccard and F1 scores
def calculate_jaccard_scores(true, pred):
    """Calculate Jaccard and F1 scores for different flood categories."""
    
    # Convert to binary (flood vs no flood)
    true_binary = (true > 0).astype(int)
    pred_binary = (pred > 0).astype(int)
    
    # Calculate Jaccard score for binary classification
    binary_jaccard = jaccard_score(true_binary.flatten(), pred_binary.flatten(), average='binary')
    # Calculate F1 score for binary classification
    binary_f1 = f1_score(true_binary.flatten(), pred_binary.flatten(), average='binary')
    
    # Calculate Jaccard and F1 scores for each class
    class_jaccard = {}
    class_f1 = {}
    for cls in range(5):  # 0, 1, 2, 3, 4
        # For each class, create binary masks
        true_cls = (true == cls).astype(int)
        pred_cls = (pred == cls).astype(int)
        
        # Calculate Jaccard and F1 score for this class
        class_jaccard[cls] = jaccard_score(true_cls.flatten(), pred_cls.flatten(), average='binary')
        class_f1[cls] = f1_score(true_cls.flatten(), pred_cls.flatten(), average='binary')
    
    return {
        'binary': binary_jaccard,
        'binary_f1': binary_f1,
        'no_flood': class_jaccard[0],  # Class 0: NO_FLOOD
        'no_flood_f1': class_f1[0],    # F1 score for NO_FLOOD
        'nuisance': class_jaccard[1],  # Class 1: NUISANCE
        'nuisance_f1': class_f1[1],    # F1 score for NUISANCE
        'minor': class_jaccard[2],     # Class 2: MINOR
        'minor_f1': class_f1[2],       # F1 score for MINOR
        'medium': class_jaccard[3],    # Class 3: MEDIUM
        'medium_f1': class_f1[3],      # F1 score for MEDIUM
        'major': class_jaccard[4],     # Class 4: MAJOR
        'major_f1': class_f1[4]        # F1 score for MAJOR
    }

# Function to plot histograms
def plot_histograms(all_results, model_names, output_dir, rainfall_level=None):
    """Plot histograms comparing model results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Plot histogram of binary classification (no flood vs flood) using Jaccard Score
    plt.figure(figsize=(10, 6))
    binary_data = {
        'No Flood': [results['no_flood'] for results in all_results],
        'Flood': [results['binary'] for results in all_results]
    }
    
    x = np.arange(len(model_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width/2, binary_data['No Flood'], width, label='No Flood')
    ax.bar(x + width/2, binary_data['Flood'], width, label='Flood')
    
    ax.set_ylabel('Jaccard Score')
    title = 'Binary Classification Performance by Model (Jaccard)'
    if rainfall_level:
        title += f' (Rainfall: {rainfall_level})'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    
    plt.tight_layout()
    filename = f'binary_histogram_jaccard_{timestamp}'
    if rainfall_level:
        filename += f'_{rainfall_level}'
    filename += '.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1b. Plot histogram of binary classification (no flood vs flood) using F1 Score
    plt.figure(figsize=(10, 6))
    binary_f1_data = {
        'No Flood': [results['no_flood_f1'] for results in all_results],
        'Flood': [results['binary_f1'] for results in all_results]
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - width/2, binary_f1_data['No Flood'], width, label='No Flood')
    ax.bar(x + width/2, binary_f1_data['Flood'], width, label='Flood')
    
    ax.set_ylabel('F1 Score')
    title = 'Binary Classification Performance by Model (F1)'
    if rainfall_level:
        title += f' (Rainfall: {rainfall_level})'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    
    plt.tight_layout()
    filename = f'binary_histogram_f1_{timestamp}'
    if rainfall_level:
        filename += f'_{rainfall_level}'
    filename += '.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot stacked histogram of flood categories using Jaccard Score
    flood_data = {
        'Nuisance': [results['nuisance'] for results in all_results],
        'Minor': [results['minor'] for results in all_results],
        'Medium': [results['medium'] for results in all_results],
        'Major': [results['major'] for results in all_results]
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(model_names))
    
    for category, scores in flood_data.items():
        ax.bar(model_names, scores, bottom=bottom, label=category)
        bottom += np.array(scores)
    
    ax.set_ylabel('Jaccard Score')
    title = 'Flood Category Classification Performance by Model (Jaccard)'
    if rainfall_level:
        title += f' (Rainfall: {rainfall_level})'
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    filename = f'flood_categories_histogram_jaccard_{timestamp}'
    if rainfall_level:
        filename += f'_{rainfall_level}'
    filename += '.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2b. Plot stacked histogram of flood categories using F1 Score
    flood_f1_data = {
        'Nuisance': [results['nuisance_f1'] for results in all_results],
        'Minor': [results['minor_f1'] for results in all_results],
        'Medium': [results['medium_f1'] for results in all_results],
        'Major': [results['major_f1'] for results in all_results]
    }
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bottom = np.zeros(len(model_names))
    
    for category, scores in flood_f1_data.items():
        ax.bar(model_names, scores, bottom=bottom, label=category)
        bottom += np.array(scores)
    
    ax.set_ylabel('F1 Score')
    title = 'Flood Category Classification Performance by Model (F1)'
    if rainfall_level:
        title += f' (Rainfall: {rainfall_level})'
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    filename = f'flood_categories_histogram_f1_{timestamp}'
    if rainfall_level:
        filename += f'_{rainfall_level}'
    filename += '.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Function to run bathtub model
def run_bathtub_model(dem, rainfall, dem_mean=None, dem_std=None):
    """Run the bathtub model on the input data.
    
    Args:
        dem: Digital Elevation Model as a 2D array (normalized)
        rainfall: 2D rainfall field with the same shape as the DEM
        dem_mean: Original DEM mean before normalization
        dem_std: Original DEM standard deviation before normalization
        
    Returns:
        water_depth: Water depth predictions from the bathtub model
        depth_categories: Categorized depth predictions
    """
    # Run the bathtub model
    print("  Running bathtub model...")
    
    # Denormalize the DEM using the provided mean and std, or use sensible defaults if not provided
    if dem_mean is not None and dem_std is not None:
        # Denormalize using the actual mean and std from the original data
        dem_denorm = dem * dem_std + dem_mean
        print(f"  Denormalizing DEM with actual mean={dem_mean:.2f}, std={dem_std:.2f}")
    else:
        # Fall back to reasonable defaults if mean and std are not provided
        dem_denorm = dem * 10 + 100  # Approximate denormalization
        print(f"  Using default denormalization values (mean=100, std=10)")
    
    # Run the bathtub model with rainfall
    water_depth = simple_bathtub_with_rainfall_robust(dem_denorm, rainfall, buffer=0)
    
    # Classify water depths into flood categories
    depth_categories = classify_depths(water_depth)
    
    return water_depth, depth_categories

# Create a Gaussian weight map for blending window predictions
def create_gaussian_weight_map(crop_size=512):
    """Create a Gaussian weight map for blending window predictions."""
    y = np.linspace(-1, 1, crop_size)
    x = np.linspace(-1, 1, crop_size)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2) / np.sqrt(2)
    
    # Create Gaussian weight with sigma=0.5 (adjust as needed)
    sigma = 0.5
    weights = np.exp(-(distance**2) / (2 * sigma**2))
    
    # Normalize weights
    weights = weights / np.max(weights)
    return weights

def main(args):
    # Set up device for inference
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    domain = args.domain
    crop_size = 512    # Size of each window (should match model input size)
    overlap = 128      # Overlap between windows for smoother predictions

    RAINFALL_LEVELS= extract_rainfall_levels(domain, RAINFALL_DICT)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), f'test_results_{domain}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the Gaussian weight map for blending
    weight_map = create_gaussian_weight_map(crop_size)
    
    # Dictionary to store model paths and names
    model_paths = {
        'RES50': ModelPaths.MODEL_RES50.value,
        'RES101': ModelPaths.MODEL_RES101.value,
        'RES152': ModelPaths.MODEL_RES152.value,
        'TINY': ModelPaths.MODEL_TINY.value,
        '100M': ModelPaths.MODEL_100M.value,
        '300M': ModelPaths.MODEL_300M.value,
        '600M': ModelPaths.MODEL_600M.value
    }
    
    # Extract the epoch numbers from checkpoint paths
    model_epochs = {}
    for name, path in model_paths.items():
        # Extract the epoch number from the path string
        try:
            epoch = int(path.split('epoch=')[1].split('.')[0])
            model_epochs[name] = epoch
        except:
            model_epochs[name] = 'unknown'
    
    # Add bathtub model to the list of models (putting it first)
    model_epochs['BATHTUB'] = 'N/A'  # No training epochs for bathtub model
    
    # Load models (do this once outside the loops)
    models = {}
    for model_name, model_path in model_paths.items():
        try:
            print(f"Loading model: {model_name}")
            models[model_name] = load_model(model_path, model_name)
            models[model_name] = models[model_name].to(device)
        except Exception as e:
            print(f"  Error loading model {model_name}: {e}")
    
    # Create CSV files for storing results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    region_jaccard_csv_path = os.path.join(output_dir, f'region_jaccard_scores_{timestamp}.csv')
    pixel_counts_csv_path = os.path.join(output_dir, f'overall_pixel_counts_{timestamp}.csv')
    
    # Headers for the CSV files
    region_jaccard_df = pd.DataFrame(columns=[
        'Rainfall', 'Model', 'Epochs',
        'Jaccard_No_Flood', 'Jaccard_Flood', 
        'Jaccard_Nuisance', 'Jaccard_Minor', 'Jaccard_Medium', 'Jaccard_Major',
        'F1_No_Flood', 'F1_Flood',
        'F1_Nuisance', 'F1_Minor', 'F1_Medium', 'F1_Major'
    ])
    
    pixel_counts_df = pd.DataFrame(columns=[
        'Rainfall', 'Category', 'Ground_Truth', 'Ground_Truth_Pct'
    ])
    
    # Add columns for each model to pixel counts dataframe, with bathtub first
    all_models = ['BATHTUB'] + list(model_paths.keys())
    for model_name in all_models:
        pixel_counts_df[f'Pred_{model_name}'] = None
        pixel_counts_df[f'Pred_{model_name}_Pct'] = None
    
    # Dictionary to store results for all rainfall levels
    all_rainfall_results = {}
    
    # Process each rainfall level
    for rainfall_level in RAINFALL_LEVELS:
        time_start= time.time()
        print(f"\nProcessing rainfall level: {rainfall_level}")
        
        try:
            # Load the entire dataset at once (with buffer applied)
            print(f"  Loading full dataset for {domain} with rainfall {rainfall_level}...")
            dem, slope, rainfall, flood_cat, target, dem_mean, dem_std = load_full_dataset(domain, rainfall_level)
            
            height, width = dem.shape
            print(f"  Dataset dimensions after applying buffer: {height} x {width}")
            
            # Create weighted prediction arrays for each model (bathtub first)
            print(f"  Creating weighted prediction arrays...")
            
            # For bathtub model: accumulate weighted flood depths before classification
            bathtub_depth_accumulator = np.zeros((height, width), dtype=np.float32)
            bathtub_weights_sum = np.zeros((height, width), dtype=np.float32)
            
            # For ML models: accumulate weighted class probabilities
            ml_predictions = {}
            ml_weights_sum = np.zeros((height, width), dtype=np.float32)
            for model_name in models.keys():
                ml_predictions[model_name] = np.zeros((5, height, width), dtype=np.float32)  # 5 classes
            
            # Calculate step size (stride) based on overlap
            stride = crop_size - overlap
            
            # Calculate how many windows we need with overlapping
            n_windows_y = max(1, (height - overlap) // stride)
            n_windows_x = max(1, (width - overlap) // stride)
            
            print(f"  Processing {n_windows_y * n_windows_x} windows ({n_windows_y} x {n_windows_x}) with overlap={overlap}, stride={stride}")
            
            # Process each overlapping window for inference
            for y in range(n_windows_y):
                for x in range(n_windows_x):
                    window_idx = y * n_windows_x + x
                    print(f"    Processing window {window_idx} ({x}, {y})")
                    
                    # Calculate window coordinates with overlapping
                    y_start = min(y * stride, height - crop_size)
                    x_start = min(x * stride, width - crop_size)
                    y_end = min(y_start + crop_size, height)
                    x_end = min(x_start + crop_size, width)
                    
                    # Extract window data
                    dem_window = dem[y_start:y_end, x_start:x_end]
                    slope_window = slope[y_start:y_end, x_start:x_end]
                    rainfall_window = rainfall[y_start:y_end, x_start:x_end]
                    
                    # Handle window size if smaller than crop_size
                    window_h = y_end - y_start
                    window_w = x_end - x_start
                    if window_h < crop_size or window_w < crop_size:
                        pad_h = max(0, crop_size - window_h)
                        pad_w = max(0, crop_size - window_w)
                        dem_window = np.pad(dem_window, ((0, pad_h), (0, pad_w)), mode='reflect')
                        slope_window = np.pad(slope_window, ((0, pad_h), (0, pad_w)), mode='reflect')
                        rainfall_window = np.pad(rainfall_window, ((0, pad_h), (0, pad_w)), mode='reflect')
                    
                    # Process bathtub model FIRST with weighting
                    try:
                        # Run bathtub model for this window
                        print(f"      Running bathtub model for window {window_idx}...")
                        water_depth, _ = run_bathtub_model(dem_window, rainfall_window, dem_mean, dem_std)
                        
                        # Apply weights to flood depths and accumulate
                        window_weights = weight_map[:window_h, :window_w]
                        weighted_depth = water_depth[:window_h, :window_w] * window_weights
                        bathtub_depth_accumulator[y_start:y_end, x_start:x_end] += weighted_depth
                        bathtub_weights_sum[y_start:y_end, x_start:x_end] += window_weights
                        
                    except Exception as e:
                        print(f"      Error processing bathtub model for window {window_idx}: {e}")
                    
                    # Process each ML model second with weighting
                    for model_name, model in models.items():
                        try:
                            # Prepare input tensor
                            model_input = np.stack([dem_window, slope_window, rainfall_window])
                            input_tensor = torch.from_numpy(model_input).unsqueeze(0).to(device)
                            
                            # Get prediction (class probabilities)
                            with torch.no_grad():
                                pred = model(input_tensor).output.squeeze().detach().cpu().numpy()
                            
                            # Apply weights to predictions and accumulate
                            window_weights = weight_map[:window_h, :window_w]
                            for c in range(pred.shape[0]):  # For each class channel
                                weighted_pred = pred[c, :window_h, :window_w] * window_weights
                                ml_predictions[model_name][c, y_start:y_end, x_start:x_end] += weighted_pred
                            
                            # Accumulate weights (same for all models)
                            if model_name == list(models.keys())[0]:  # Only do this once
                                ml_weights_sum[y_start:y_end, x_start:x_end] += window_weights
                        
                        except Exception as e:
                            print(f"      Error processing model {model_name} for window {window_idx}: {e}")
            
            # Finalize predictions with normalization and classification
            print(f"  Finalizing weighted predictions...")
            predictions = {}
            
            # For bathtub model: normalize depths and then classify
            normalized_depths = np.divide(bathtub_depth_accumulator, bathtub_weights_sum, 
                                        out=np.zeros_like(bathtub_depth_accumulator), 
                                        where=bathtub_weights_sum > 0)
            predictions['BATHTUB'] = classify_depths(normalized_depths)
            
            # For ML models: normalize probabilities and get argmax
            for model_name in models.keys():
                normalized_probs = np.zeros_like(ml_predictions[model_name])
                for c in range(5):
                    normalized_probs[c] = np.divide(ml_predictions[model_name][c], ml_weights_sum,
                                                  out=np.zeros_like(ml_predictions[model_name][c]),
                                                  where=ml_weights_sum > 0)
                predictions[model_name] = np.argmax(normalized_probs, axis=0).astype(np.int64)
            
            time_end= time.time()
            proc_time= (time_end - time_start)
            print(f'Processing time: {proc_time:.1f} seconds')
            
            # Calculate jaccard scores for the entire domain for each model (bathtub first)
            jaccard_scores = {}
            # Process bathtub first
            jaccard_scores['BATHTUB'] = calculate_jaccard_scores(flood_cat, predictions['BATHTUB'])
            
            # Then process the ML models
            for model_name, pred in predictions.items():
                if model_name == 'BATHTUB':
                    continue  # Already processed
                jaccard = calculate_jaccard_scores(flood_cat, pred)
                jaccard_scores[model_name] = jaccard
            
            # Add the scores to the region jaccard dataframe (bathtub first)
            # First add bathtub
            region_jaccard_df = pd.concat([region_jaccard_df, pd.DataFrame({
                'Rainfall': [rainfall_level],
                'Model': ['BATHTUB'],
                'Epochs': [model_epochs.get('BATHTUB', 'N/A')],
                'Jaccard_No_Flood': [jaccard_scores['BATHTUB']['no_flood']],
                'Jaccard_Flood': [jaccard_scores['BATHTUB']['binary']],
                'Jaccard_Nuisance': [jaccard_scores['BATHTUB']['nuisance']],
                'Jaccard_Minor': [jaccard_scores['BATHTUB']['minor']],
                'Jaccard_Medium': [jaccard_scores['BATHTUB']['medium']],
                'Jaccard_Major': [jaccard_scores['BATHTUB']['major']],
                'F1_No_Flood': [jaccard_scores['BATHTUB']['no_flood_f1']],
                'F1_Flood': [jaccard_scores['BATHTUB']['binary_f1']],
                'F1_Nuisance': [jaccard_scores['BATHTUB']['nuisance_f1']],
                'F1_Minor': [jaccard_scores['BATHTUB']['minor_f1']],
                'F1_Medium': [jaccard_scores['BATHTUB']['medium_f1']],
                'F1_Major': [jaccard_scores['BATHTUB']['major_f1']]
            })], ignore_index=True)
            
            # Then add the ML models
            for model_name in models.keys():
                region_jaccard_df = pd.concat([region_jaccard_df, pd.DataFrame({
                    'Rainfall': [rainfall_level],
                    'Model': [model_name],
                    'Epochs': [model_epochs.get(model_name, 'N/A')],
                    'Jaccard_No_Flood': [jaccard_scores[model_name]['no_flood']],
                    'Jaccard_Flood': [jaccard_scores[model_name]['binary']],
                    'Jaccard_Nuisance': [jaccard_scores[model_name]['nuisance']],
                    'Jaccard_Minor': [jaccard_scores[model_name]['minor']],
                    'Jaccard_Medium': [jaccard_scores[model_name]['medium']],
                    'Jaccard_Major': [jaccard_scores[model_name]['major']],
                    'F1_No_Flood': [jaccard_scores[model_name]['no_flood_f1']],
                    'F1_Flood': [jaccard_scores[model_name]['binary_f1']],
                    'F1_Nuisance': [jaccard_scores[model_name]['nuisance_f1']],
                    'F1_Minor': [jaccard_scores[model_name]['minor_f1']],
                    'F1_Medium': [jaccard_scores[model_name]['medium_f1']],
                    'F1_Major': [jaccard_scores[model_name]['major_f1']]
                })], ignore_index=True)
            
            # Count pixels for ground truth and each model prediction
            gt_counts, total_pixels = count_pixels_by_category(flood_cat)
            model_counts = {model_name: count_pixels_by_category(pred)[0] 
                           for model_name, pred in predictions.items()}
            
            # Add pixel counts to dataframe
            categories = ['No Flood', 'Nuisance', 'Minor', 'Medium', 'Major', 'Total Flood']
            for i, category in enumerate(categories):
                if i < 5:  # Regular flood categories
                    gt_count = gt_counts[i]
                else:  # Total flood (all non-zero categories)
                    gt_count = gt_counts['flood']
                
                row_data = {
                    'Rainfall': rainfall_level,
                    'Category': category,
                    'Ground_Truth': gt_count,
                    'Ground_Truth_Pct': (gt_count / total_pixels) * 100
                }
                
                # Add model predictions (bathtub first)
                model_order = ['BATHTUB'] + [m for m in model_counts.keys() if m != 'BATHTUB']
                for model_name in model_order:
                    counts = model_counts[model_name]
                    if i < 5:
                        pred_count = counts[i]
                    else:
                        pred_count = counts['flood']
                    
                    row_data[f'Pred_{model_name}'] = pred_count
                    row_data[f'Pred_{model_name}_Pct'] = (pred_count / total_pixels) * 100
                
                # Add row to dataframe
                pixel_counts_df = pd.concat([pixel_counts_df, pd.DataFrame([row_data])], ignore_index=True)
            
            # Store results for this rainfall level
            all_rainfall_results[rainfall_level] = {
                'dem': dem,
                'slope': slope,
                'flood_cat': flood_cat,
                'predictions': predictions,
                'jaccard_scores': jaccard_scores,
                'pixel_counts': {
                    'ground_truth': gt_counts,
                    'models': model_counts
                }
            }
            
            # Create visualization of full domain predictions
            print(f"  Creating visualization for {rainfall_level}...")
            
            # Create colormap for flood categories
            flood_colors = [FLOOD_COLORS[FloodCategory(i)] for i in range(len(FloodCategory))]
            cmap = ListedColormap(flood_colors)
            norm = mpl.colors.Normalize(vmin=0, vmax=4)
            
            # Create figure with subplots for ground truth and each model prediction
            n_models = len(models) + 1  # +1 for bathtub model
            total_plots = n_models + 1  # +1 for ground truth
            cols = 3  # Fixed number of columns
            rows = (total_plots + cols - 1) // cols  # Ceiling division to calculate needed rows
            
            # Create figure with appropriate size
            fig_width = 15  # 5 inches per column, 3 columns
            fig_height = 3.5 * rows  # 5 inches per row
            
            # Use constrained_layout=False so we can adjust spacing manually
            fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), constrained_layout=False)
            
            # Flatten axes for easier indexing
            axes = axes.flatten() if rows > 1 else axes
            
            # Hide any unused subplots
            for i in range(total_plots, len(axes)):
                axes[i].axis('off')
            
            # Plot ground truth
            axes[0].set_title("Ground Truth")
            im_gt = axes[0].imshow(flood_cat, cmap=cmap, norm=norm)
            cbar = fig.colorbar(im_gt, ax=axes[0], ticks=range(len(FloodCategory)), fraction=0.046)
            cbar.ax.set_yticklabels([cat.name for cat in FloodCategory])
            
            # Plot predictions for each model (bathtub first)
            model_names = ['BATHTUB'] + list(models.keys())
            for i, model_name in enumerate(model_names):
                binary_f1 = jaccard_scores[model_name]['binary_f1']
                axes[i+1].set_title(f"{model_name} - F1: {binary_f1:.3f}")
                im = axes[i+1].imshow(predictions[model_name], cmap=cmap, norm=norm)
                cbar = fig.colorbar(im, ax=axes[i+1], ticks=range(len(FloodCategory)), fraction=0.046)
                cbar.ax.set_yticklabels([cat.name for cat in FloodCategory])
            
            # Add title with rainfall information
            fig.suptitle(f"Full Domain Prediction - Rainfall: {rainfall_level}", fontsize=16, y=0.98)

            plt.subplots_adjust(wspace=0.3, hspace=0.3)
            
            # Use tight layout to minimize white space
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the suptitle
            
            # Save the figure
            region_viz_path = os.path.join(output_dir, f'domain_prediction_{rainfall_level}_{timestamp}.png')
            plt.savefig(region_viz_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            # Plot histograms comparing model performances
            if len(predictions) > 0:
                print(f"  Creating performance histograms for {rainfall_level}...")
                # Make sure bathtub is first
                ordered_model_names = ['BATHTUB'] + [m for m in list(models.keys())]
                model_results = [jaccard_scores[model_name] for model_name in ordered_model_names]
                plot_histograms(model_results, ordered_model_names, output_dir, rainfall_level=rainfall_level)
        
        except Exception as e:
            print(f"  Error processing rainfall level {rainfall_level}: {e}")
    
    # Save the overall results
    region_jaccard_df.to_csv(region_jaccard_csv_path, index=False)
    pixel_counts_df.to_csv(pixel_counts_csv_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"- {region_jaccard_csv_path}")
    print(f"- {pixel_counts_csv_path}")
    print(f"All visualizations saved to {output_dir}")
    
    # Plot a comparison of model performances across all rainfall levels
    if len(all_rainfall_results) > 0:
        print("\nGenerating comparison plots across rainfall levels...")
        
        # Sort rainfall levels by numeric value for proper ordering
        rainfall_numeric = [int(r.split('mm')[0]) for r in RAINFALL_LEVELS]
        sorted_indices = np.argsort(rainfall_numeric)
        sorted_rainfall = [RAINFALL_LEVELS[i] for i in sorted_indices]
        
        # Line plot of Jaccard scores by rainfall level
        plt.figure(figsize=(12, 8))
        
        # Make sure bathtub is plotted first (will appear first in legend)
        ordered_models = ['BATHTUB'] + list(models.keys())
        for model_name in ordered_models:
            model_data = region_jaccard_df[region_jaccard_df['Model'] == model_name]
            binary_scores = []
            
            for rainfall in sorted_rainfall:
                rainfall_row = model_data[model_data['Rainfall'] == rainfall]
                if not rainfall_row.empty:
                    binary_scores.append(rainfall_row['Jaccard_Flood'].values[0])
                else:
                    binary_scores.append(np.nan)
            
            plt.plot(sorted_rainfall, binary_scores, marker='o', linewidth=2, label=model_name)
        
        plt.xlabel('Rainfall Level')
        plt.ylabel('Jaccard Score (Binary Flood)')
        plt.title('Flood Detection Performance by Rainfall Level')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'rainfall_comparison_jaccard_{timestamp}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        # Line plot of F1 scores by rainfall level
        plt.figure(figsize=(12, 8))
        
        # Make sure bathtub is plotted first (will appear first in legend)
        for model_name in ordered_models:
            model_data = region_jaccard_df[region_jaccard_df['Model'] == model_name]
            binary_scores = []
            
            for rainfall in sorted_rainfall:
                rainfall_row = model_data[model_data['Rainfall'] == rainfall]
                if not rainfall_row.empty:
                    binary_scores.append(rainfall_row['F1_Flood'].values[0])
                else:
                    binary_scores.append(np.nan)
            
            plt.plot(sorted_rainfall, binary_scores, marker='o', linewidth=2, label=model_name)
        
        plt.xlabel('Rainfall Level')
        plt.ylabel('F1 Score (Binary Flood)')
        plt.title('Flood Detection F1 Performance by Rainfall Level')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'rainfall_comparison_f1_{timestamp}.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict flooding for different rainfall levels')
    parser.add_argument('--domain', type=str, default='HOU007', help='Domain to test')
    
    args = parser.parse_args()
    main(args)
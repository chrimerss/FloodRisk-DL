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
import argparse
import time
import json

# Import modules from terratorch
from terratorch.tasks import SemanticSegmentationTask
from task_class import model_args_res50, model_args_res101, model_args_res152, model_args_tiny, model_args_100, model_args_300, model_args_600
from model_pth import FloodCategory as ModelPaths

# Define buffer size for input data
BUFFER = 10

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

def extract_rainfall_levels(city_id, data):
    # Define the preferred order of return periods
    return_periods_ordered = [
        "1000-yr", "500-yr", "200-yr", "100-yr", "50-yr",
        "25-yr", "10-yr", "5-yr", "2-yr", "1-yr"
    ]
    
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

# Function to load the entire dataset
def load_full_dataset(domain, rainfall_level):
    """Load the entire DEM and flood data for a domain and rainfall level.
    
    Args:
        domain: The domain to use (e.g., 'HOU001')
        rainfall_level: The rainfall level to use (e.g., '98mm')
        
    Returns:
        dem, slope, rainfall, flood_cat, target
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

    if dem.shape[0] > 2*BUFFER and dem.shape[1] > 2*BUFFER:
        dem = dem[BUFFER:-BUFFER, BUFFER:-BUFFER]
        slope = slope[BUFFER:-BUFFER, BUFFER:-BUFFER]
        target = target[BUFFER:-BUFFER, BUFFER:-BUFFER]  
    # Classify target depths into categories
    flood_cat = classify_depths(target)
    
    # Prepare rainfall input (constant value across the image)
    rainfall = np.ones_like(dem) * int(rainfall_level.split('mm')[0])/1000.
    
    # Apply buffer to all arrays if they are large enough

    
    # Normalize DEM
    dem = (dem - dem.mean()) / dem.std()
    
    return dem, slope, rainfall, flood_cat, target

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

# Run inference with overlapping windows for smoother predictions
def run_inference_with_overlap(model, dem, slope, rainfall, device, crop_size=512, overlap=128):
    """Run inference with overlapping windows and blend the results."""
    height, width = dem.shape
    predictions = np.zeros((5, height, width), dtype=np.float32)  # Store class probabilities
    weights_sum = np.zeros((height, width), dtype=np.float32)
    
    # Get the Gaussian weight map
    weight_map = create_gaussian_weight_map(crop_size)
    
    # Calculate step size (stride) based on overlap
    stride = crop_size - overlap
    
    # Calculate how many windows we need
    n_windows_y = max(1, (height - overlap) // stride)
    n_windows_x = max(1, (width - overlap) // stride)
    
    print(f"Running inference with {n_windows_y}x{n_windows_x} windows, stride={stride}, overlap={overlap}")
    
    for y in range(n_windows_y):
        for x in range(n_windows_x):
            # Calculate window coordinates
            y_start = min(y * stride, height - crop_size)
            x_start = min(x * stride, width - crop_size)
            y_end = min(y_start + crop_size, height)
            x_end = min(x_start + crop_size, width)
            
            # Extract window data
            dem_window = dem[y_start:y_end, x_start:x_end]
            slope_window = slope[y_start:y_end, x_start:x_end]
            rainfall_window = rainfall[y_start:y_end, x_start:x_end]
            
            # Handle window size if smaller than crop_size
            if dem_window.shape[0] < crop_size or dem_window.shape[1] < crop_size:
                pad_h = max(0, crop_size - dem_window.shape[0])
                pad_w = max(0, crop_size - dem_window.shape[1])
                dem_window = np.pad(dem_window, ((0, pad_h), (0, pad_w)), mode='reflect')
                slope_window = np.pad(slope_window, ((0, pad_h), (0, pad_w)), mode='reflect')
                rainfall_window = np.pad(rainfall_window, ((0, pad_h), (0, pad_w)), mode='reflect')
            
            # Prepare input tensor
            model_input = np.stack([dem_window, slope_window, rainfall_window])
            input_tensor = torch.from_numpy(model_input).unsqueeze(0).to(device)
            
            # Get prediction
            with torch.no_grad():
                pred = model(input_tensor).output.squeeze().detach().cpu().numpy()
            
            # Extract actual window dimensions
            window_h = y_end - y_start
            window_w = x_end - x_start
            
            # Apply weights to prediction and add to accumulated predictions
            for c in range(pred.shape[0]):  # For each class channel
                weighted_pred = pred[c, :window_h, :window_w] * weight_map[:window_h, :window_w]
                predictions[c, y_start:y_end, x_start:x_end] += weighted_pred
            
            # Add weights to sum for later normalization
            weights_sum[y_start:y_end, x_start:x_end] += weight_map[:window_h, :window_w]
    
    # Normalize the predictions by the weight sum
    for c in range(predictions.shape[0]):
        predictions[c] = np.divide(predictions[c], weights_sum, out=np.zeros_like(predictions[c]), where=weights_sum > 0)
    
    # Get class with highest probability for each pixel
    final_pred = np.argmax(predictions, axis=0).astype(np.int64)
    
    # Calculate confidence (max probability) for each pixel
    confidence = np.max(predictions, axis=0)
    
    return final_pred, confidence

def main(args):
    # Set up device for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Domain to test
    domain = args.domain
    crop_size = 512  # Size of each window
    overlap = args.overlap  # Overlap between windows
    
    RAINFALL_LEVELS= extract_rainfall_levels(domain, RAINFALL_DICT)

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), f'pred_results')
    os.makedirs(output_dir, exist_ok=True)

    # Get model path
    model_type = args.model
    if model_type == 'RES50':
        model_path = ModelPaths.MODEL_RES50.value
    elif model_type == 'RES101':
        model_path = ModelPaths.MODEL_RES101.value
    elif model_type == 'RES152':
        model_path = ModelPaths.MODEL_RES152.value
    elif model_type == 'TINY':
        model_path = ModelPaths.MODEL_TINY.value
    elif model_type == '100M':
        model_path = ModelPaths.MODEL_100M.value
    elif model_type == '300M':
        model_path = ModelPaths.MODEL_300M.value
    elif model_type == '600M':
        model_path = ModelPaths.MODEL_600M.value
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load the model
    try:
        print(f"Loading {model_type} model from {model_path}")
        model = load_model(model_path, model_type)
        model = model.to(device)
    except Exception as e:
        print(f"Error loading model {model_type}: {e}")
        return
    
    # Get timestamp for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Sort rainfall levels by numeric value for proper ordering
    rainfall_numeric = [int(r.split('mm')[0]) for r in RAINFALL_LEVELS]
    sorted_indices = np.argsort(rainfall_numeric)[::-1]  # Sort in descending order
    sorted_rainfall = [RAINFALL_LEVELS[i] for i in sorted_indices]
    
    # Select only the specified number of rainfall levels
    if args.num_rainfall_levels > 0:
        sorted_rainfall = sorted_rainfall[:args.num_rainfall_levels]
    
    print(f"Processing rainfall levels: {sorted_rainfall}")
    
    # Create figure for visualization with rainfall scenarios as rows and 3 columns (GT, prediction, confidence)
    n_rainfall = len(sorted_rainfall)
    fig_height = 4 * n_rainfall  # 4 inches per row
    fig_width = 15  # 5 inches per column for 3 columns
    
    fig, axes = plt.subplots(n_rainfall, 3, figsize=(fig_width, fig_height))
    
    # Create colormap for flood categories
    flood_colors = [FLOOD_COLORS[FloodCategory(i)] for i in range(len(FloodCategory))]
    cmap = ListedColormap(flood_colors)
    norm = mpl.colors.Normalize(vmin=0, vmax=4)
    
    # Create colormap for confidence visualization
    confidence_cmap = plt.cm.YlOrBr
    
    # Process each rainfall level
    for i, rainfall_level in enumerate(sorted_rainfall):
        time_start = time.time()
        print(f"\nProcessing rainfall level: {rainfall_level}")
        
        try:
            # Load the dataset for this rainfall level
            dem, slope, rainfall, ground_truth, target = load_full_dataset(domain, rainfall_level)
            
            height, width = dem.shape
            print(f"Dataset dimensions: {height} x {width}")
            
            # Run inference with overlapping windows for smoother predictions
            prediction, confidence = run_inference_with_overlap(
                model, dem, slope, rainfall, device, 
                crop_size=crop_size, overlap=overlap
            )
            
            # Access the appropriate row of subplots (handle both single and multiple rows)
            if n_rainfall == 1:
                ax_row = axes
            else:
                ax_row = axes[i]
            
            # Plot ground truth
            ax_gt = ax_row[0]
            ax_gt.set_title(f"Ground Truth - {rainfall_level}")
            im_gt = ax_gt.imshow(ground_truth[:-crop_size,:-crop_size], cmap=cmap, norm=norm)
            
            # Plot prediction
            ax_pred = ax_row[1]
            ax_pred.set_title(f"{model_type} Prediction - {rainfall_level}")
            im_pred = ax_pred.imshow(prediction[:-crop_size,:-crop_size], cmap=cmap, norm=norm)
            
            # Plot confidence (logits value)
            ax_conf = ax_row[2]
            ax_conf.set_title(f"Prediction Confidence - {rainfall_level}")
            confidence_display = confidence[:-crop_size,:-crop_size]
            im_conf = ax_conf.imshow(confidence_display, cmap=confidence_cmap, vmin=0.0, vmax=1.0)
            
            time_end = time.time()
            proc_time = (time_end - time_start)
            print(f'Processing time for {rainfall_level}: {proc_time:.1f} seconds')
            
        except Exception as e:
            print(f"Error processing rainfall level {rainfall_level}: {e}")
            # If there's an error, leave the subplot empty
            if n_rainfall == 1:
                ax_row = axes
            else:
                ax_row = axes[i]
            ax_row[0].axis('off')
            ax_row[1].axis('off')
            ax_row[2].axis('off')
    
    # Add colorbars with adjusted positions to prevent overlap
    # First colorbar (flood category)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im_gt, cax=cbar_ax, ticks=range(len(FloodCategory)))
    cbar.ax.set_yticklabels([cat.name for cat in FloodCategory])
    # Move label to top
    cbar.ax.set_title('Flood Category', pad=10)
    
    # Second colorbar (confidence) - moved further to the right
    cbar_conf_ax = fig.add_axes([0.94, 0.15, 0.01, 0.7])  # [left, bottom, width, height]
    cbar_conf = fig.colorbar(im_conf, cax=cbar_conf_ax)
    # Move label to top
    cbar_conf.ax.set_title('Confidence', pad=10)
    
    # Add a title with domain and model information
    plt.suptitle(f"Flood Predictions for {domain} using {model_type}", fontsize=16, y=0.98)
    
    # Adjust layout with more room for colorbars
    plt.tight_layout(rect=[0, 0, 0.89, 0.95])  # Make room for colorbar and title
    
    # Save the figure
    output_filename = f"{domain}_{model_type}_predictions_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict flooding for different rainfall levels')
    parser.add_argument('--domain', type=str, default='HOU007', help='Domain to test')
    parser.add_argument('--model', type=str, default='300M', 
                        choices=['RES50', 'RES101', 'RES152', 'TINY', '100M', '300M', '600M'],
                        help='Model type to use')
    parser.add_argument('--overlap', type=int, default=128, 
                        help='Overlap size between windows (pixels)')
    parser.add_argument('--num_rainfall_levels', type=int, default=0,
                        help='Number of rainfall levels to visualize (0 for all)')
    
    args = parser.parse_args()
    main(args)
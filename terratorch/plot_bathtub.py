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
from bathtub.bathtub import simple_bathtub_with_rainfall_robust



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
RAINFALL_LEVELS = ["181mm", "162mm", "138mm", "123mm", "110mm", "98mm", "82mm", "70mm", "57mm", "48mm"]

# Function to classify flood depths into categories
def classify_depths(src):
    """Classify flood depths into categories."""
    categories = np.zeros_like(src, dtype=np.int64)
    categories = np.where(src >= 0.1, 1, categories)
    categories = np.where(src >= 0.2, 2, categories)
    categories = np.where(src >= 0.5, 3, categories)
    categories = np.where(src >= 1.0, 4, categories)    
    return categories

# Function to load the entire dataset
def load_full_dataset(domain, rainfall_level):
    """Load the entire DEM and flood data for a domain and rainfall level.
    
    Args:
        domain: The domain to use (e.g., 'HOU001')
        rainfall_level: The rainfall level to use (e.g., '98mm')
        
    Returns:
        dem, rainfall, flood_cat
    """
    input_dir = f'/home/users/li1995/global_flood/UrbanFloods2D/dataset/{domain}'
    dem_input = os.path.join(input_dir, f"{domain}_DEM.tif")
    input_file = f'/home/users/li1995/global_flood/UrbanFloods2D/sample/{domain}_{rainfall_level}_max.tif'
    
    # Load DEM and calculate slope for the entire domain
    with rasterio.open(dem_input) as src:
        dem = src.read(1).astype(np.float32)
    
    # Load target flood depths
    with rasterio.open(input_file) as src:
        target = src.read(1).astype(np.float32)

    if dem.shape[0] > 2*BUFFER and dem.shape[1] > 2*BUFFER:
        dem = dem[BUFFER:-BUFFER, BUFFER:-BUFFER]
        target = target[BUFFER:-BUFFER, BUFFER:-BUFFER]  
    # Classify target depths into categories
    flood_cat = classify_depths(target)
    
    # Prepare rainfall input (constant value across the image)
    rainfall = np.ones_like(dem) * int(rainfall_level.split('mm')[0])/1000.

    
    return dem, rainfall, flood_cat



# Run inference with overlapping windows for smoother predictions
def run(dem, rainfall):
    """Run simple bathtub model"""

    water_depth = simple_bathtub_with_rainfall_robust(dem, rainfall)
    
    
    return water_depth

def main(args):
    # Set up device for inference

    model_type='bathtub'
    
    # Domain to test
    domain = args.domain
    crop_size = 512  # Size of each window
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), f'pred_results')
    os.makedirs(output_dir, exist_ok=True)

    
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
    
    # Create figure for visualization with rainfall scenarios as rows and 2 columns (GT, prediction)
    n_rainfall = len(sorted_rainfall)
    fig_height = 4 * n_rainfall  # 4 inches per row
    fig_width = 10  # 5 inches per column for 2 columns
    
    fig, axes = plt.subplots(n_rainfall, 2, figsize=(fig_width, fig_height))
    
    # Create colormap for flood categories
    flood_colors = [FLOOD_COLORS[FloodCategory(i)] for i in range(len(FloodCategory))]
    cmap = ListedColormap(flood_colors)
    norm = mpl.colors.Normalize(vmin=0, vmax=4)
    
    # Process each rainfall level
    for i, rainfall_level in enumerate(sorted_rainfall):
        time_start = time.time()
        print(f"\nProcessing rainfall level: {rainfall_level}")
        
        # Load the dataset for this rainfall level
        dem, rainfall, ground_truth = load_full_dataset(domain, rainfall_level)
        
        height, width = dem.shape
        print(f"Dataset dimensions: {height} x {width}")
        
        # Run inference with overlapping windows for smoother predictions
        prediction = run(dem, rainfall)
        
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
        
        time_end = time.time()
        proc_time = (time_end - time_start)
        print(f'Processing time for {rainfall_level}: {proc_time:.1f} seconds')
            

    
    # Add a common colorbar for all subplots
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im_gt, cax=cbar_ax, ticks=range(len(FloodCategory)))
    cbar.ax.set_yticklabels([cat.name for cat in FloodCategory])
    
    # Add a title with domain and model information
    plt.suptitle(f"Flood Predictions for {domain} using {model_type}", fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])  # Make room for colorbar and title
    
    # Save the figure
    output_filename = f"{domain}_{model_type}_predictions_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict flooding for different rainfall levels')
    parser.add_argument('--domain', type=str, default='HOU007', help='Domain to test')
    parser.add_argument('--num_rainfall_levels', type=int, default=0,
                        help='Number of rainfall levels to visualize (0 for all)')
    
    args = parser.parse_args()
    main(args)
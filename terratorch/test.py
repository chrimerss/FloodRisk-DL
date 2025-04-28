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
from sklearn.metrics import jaccard_score
import seaborn as sns
from collections import defaultdict

from terratorch.tasks import SemanticSegmentationTask

# Import model arguments from task_class.py
from task_class import model_args_tiny, model_args_100, model_args_300, model_args_600

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
    if model_type == 'TINY':
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
        dem = (dem - dem.mean()) / dem.std()  # Normalize
    
    # Load target flood depths
    with rasterio.open(input_file) as src:    
        target = src.read(window=window).squeeze().astype(np.float32)
        flood_cat = classify_depths(target)
    
    # Prepare rainfall input (constant value across the image)
    rainfall = np.ones_like(dem) * int(rainfall_level.split('mm')[0])/1000.
    
    return dem, slope, rainfall, flood_cat, target

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

# Function to calculate Jaccard scores
def calculate_jaccard_scores(true, pred):
    """Calculate Jaccard scores for different flood categories."""
    
    # Convert to binary (flood vs no flood)
    true_binary = (true > 0).astype(int)
    pred_binary = (pred > 0).astype(int)
    
    # Calculate Jaccard score for binary classification
    binary_jaccard = jaccard_score(true_binary.flatten(), pred_binary.flatten(), average='binary')
    
    # Calculate Jaccard scores for each class
    class_jaccard = {}
    for cls in range(5):  # 0, 1, 2, 3, 4
        # For each class, create binary masks
        true_cls = (true == cls).astype(int)
        pred_cls = (pred == cls).astype(int)
        
        # Calculate Jaccard score for this class
        class_jaccard[cls] = jaccard_score(true_cls.flatten(), pred_cls.flatten(), average='binary')
    
    return {
        'binary': binary_jaccard,
        'no_flood': class_jaccard[0],  # Class 0: NO_FLOOD
        'nuisance': class_jaccard[1],  # Class 1: NUISANCE
        'minor': class_jaccard[2],     # Class 2: MINOR
        'medium': class_jaccard[3],    # Class 3: MEDIUM
        'major': class_jaccard[4]      # Class 4: MAJOR
    }

# Function to plot sample predictions
def plot_sample_predictions(dem, slope, flood_cat, predictions, model_names, output_dir, window_idx=None, rainfall_level=None):
    """Plot sample predictions from different models."""
    
    n_models = len(predictions)
    fig_width = 4 + 3 * n_models
    
    fig, axes = plt.subplots(1, 2 + n_models, figsize=(fig_width, 8), constrained_layout=True)
    
    # Create colormap for flood categories
    flood_colors = [FLOOD_COLORS[FloodCategory(i)] for i in range(len(FloodCategory))]
    cmap = ListedColormap(flood_colors)
    norm = mpl.colors.Normalize(vmin=0, vmax=4)
    
    # Plot DEM
    axes[0].set_title("DEM")
    im1 = axes[0].imshow(dem, cmap='terrain')
    fig.colorbar(im1, ax=axes[0])
    
    # Plot ground truth
    axes[1].set_title("Ground Truth")
    im2 = axes[1].imshow(flood_cat, cmap=cmap, norm=norm)
    cbar = fig.colorbar(im2, ax=axes[1], ticks=range(len(FloodCategory)))
    cbar.ax.set_yticklabels([cat.name for cat in FloodCategory])
    
    # Plot predictions for each model
    for i, (pred, model_name) in enumerate(zip(predictions, model_names)):
        axes[i+2].set_title(f"Predicted: {model_name}")
        im = axes[i+2].imshow(pred, cmap=cmap, norm=norm)
        cbar = fig.colorbar(im, ax=axes[i+2], ticks=range(len(FloodCategory)))
        cbar.ax.set_yticklabels([cat.name for cat in FloodCategory])
    
    # Add title with window and rainfall info if provided
    if window_idx is not None and rainfall_level is not None:
        fig.suptitle(f"Rainfall: {rainfall_level}, Window: {window_idx}", fontsize=16)
    
    # Save the figure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'sample_predictions_{timestamp}'
    if window_idx is not None:
        filename += f'_window{window_idx}'
    if rainfall_level is not None:
        filename += f'_{rainfall_level}'
    filename += '.png'
    
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot histograms
def plot_histograms(all_results, model_names, output_dir, rainfall_level=None):
    """Plot histograms comparing model results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Plot histogram of binary classification (no flood vs flood)
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
    title = 'Binary Classification Performance by Model'
    if rainfall_level:
        title += f' (Rainfall: {rainfall_level})'
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    
    plt.tight_layout()
    filename = f'binary_histogram_{timestamp}'
    if rainfall_level:
        filename += f'_{rainfall_level}'
    filename += '.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot stacked histogram of flood categories
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
    title = 'Flood Category Classification Performance by Model'
    if rainfall_level:
        title += f' (Rainfall: {rainfall_level})'
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    filename = f'flood_categories_histogram_{timestamp}'
    if rainfall_level:
        filename += f'_{rainfall_level}'
    filename += '.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Plot pixel count distribution for ground truth and predictions
    # This will be added in the main function

def main():
    # Set up device for inference
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), 'test_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Test parameters
    domain = 'HOU001'  # Domain to test
    crop_size = 512    # Size of each window (should match model input size)
    
    # Dictionary to store model paths and names
    model_paths = {
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
    
    # Load models (do this once outside the loops)
    models = {}
    for model_name, model_path in model_paths.items():
        try:
            print(f"Loading model: {model_name}")
            models[model_name] = load_model(model_path, model_name)
            models[model_name] = models[model_name].to(device)
        except Exception as e:
            print(f"  Error loading model {model_name}: {e}")
    
    # Get the full dimensions of the image
    height, width = get_image_dimensions(domain)
    print(f"Image dimensions: {height} x {width}")
    
    # Calculate how many windows we need to cover the entire image
    n_windows_y = height // crop_size
    n_windows_x = width // crop_size
    print(f"Processing {n_windows_y * n_windows_x} windows ({n_windows_y} x {n_windows_x})")
    
    # Container for all results across rainfall levels
    all_rainfall_results = {}
    
    # Create CSV files for storing all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_csv_path = os.path.join(output_dir, f'overall_model_evaluation_{timestamp}.csv')
    overall_pixel_csv_path = os.path.join(output_dir, f'overall_pixel_counts_{timestamp}.csv')
    
    # Headers for the overall CSV files
    overall_results_df = pd.DataFrame(columns=[
        'Rainfall', 'Window', 'Model', 'Epochs',
        'Jaccard_No_Flood', 'Jaccard_Flood',
        'Jaccard_Nuisance', 'Jaccard_Minor', 'Jaccard_Moderate', 'Jaccard_Major'
    ])
    
    overall_pixel_counts_df = pd.DataFrame(columns=[
        'Rainfall', 'Window', 'Category', 'Ground_Truth'
    ])
    # Add columns for each model
    for model_name in model_paths.keys():
        overall_pixel_counts_df[f'Pred_{model_name}'] = None
        overall_pixel_counts_df[f'Ground_Truth_Pct'] = None
        overall_pixel_counts_df[f'Pred_{model_name}_Pct'] = None
    
    # Process each rainfall level
    for rainfall_level in RAINFALL_LEVELS:
        print(f"\nProcessing rainfall level: {rainfall_level}")
        
        # Containers for aggregated results for this rainfall level
        rainfall_results = []
        rainfall_pixel_counts = []
        rainfall_predictions = []
        
        # Process each window
        for y in range(n_windows_y):
            for x in range(n_windows_x):
                window_idx = y * n_windows_x + x
                print(f"  Processing window {window_idx} ({x}, {y})")
                
                # Define the window
                window = Window(x * crop_size, y * crop_size, crop_size, crop_size)
                
                # Prepare data for this window
                try:
                    dem, slope, rainfall, flood_cat, target = prepare_data_window(domain, rainfall_level, window)
                except Exception as e:
                    print(f"    Error preparing data for window {window_idx}: {e}")
                    continue
                
                # Count pixels in ground truth for this window
                gt_pixel_counts, total_pixels = count_pixels_by_category(flood_cat)
                
                # Process each model
                window_results = []
                window_predictions = []
                window_pixel_counts = []
                
                for model_name, model in models.items():
                    try:
                        # Prepare input tensor
                        model_input = np.stack([dem, slope, rainfall])
                        input_tensor = torch.from_numpy(model_input).unsqueeze(0).to(device)
                        
                        # Get prediction
                        with torch.no_grad():
                            pred = model(input_tensor).output.squeeze().detach().cpu().numpy()
                            pred = pred.argmax(axis=0)
                        
                        # Calculate metrics
                        jaccard_scores = calculate_jaccard_scores(flood_cat, pred)
                        
                        # Count pixels in prediction
                        pred_pixel_counts, _ = count_pixels_by_category(pred)
                        
                        # Store results for this window and model
                        window_results.append(jaccard_scores)
                        window_predictions.append(pred)
                        window_pixel_counts.append(pred_pixel_counts)
                        
                        # Add to overall results DataFrame
                        overall_results_df = pd.concat([overall_results_df, pd.DataFrame({
                            'Rainfall': [rainfall_level],
                            'Window': [window_idx],
                            'Model': [model_name],
                            'Epochs': [model_epochs[model_name]],
                            'Jaccard_No_Flood': [jaccard_scores['no_flood']],
                            'Jaccard_Flood': [jaccard_scores['binary']],
                            'Jaccard_Nuisance': [jaccard_scores['nuisance']],
                            'Jaccard_Minor': [jaccard_scores['minor']],
                            'Jaccard_Moderate': [jaccard_scores['medium']],
                            'Jaccard_Major': [jaccard_scores['major']]
                        })], ignore_index=True)
                        
                    except Exception as e:
                        print(f"    Error processing model {model_name} for window {window_idx}: {e}")
                
                # If we have results for this window, save them
                if window_results:
                    # Store results from this window
                    rainfall_results.append(window_results)
                    rainfall_predictions.append(window_predictions)
                    rainfall_pixel_counts.append(window_pixel_counts)
                    
                    # Update pixel counts in overall DataFrame
                    categories = ['No Flood', 'Nuisance', 'Minor', 'Medium', 'Major', 'Total Flood']
                    for i, category in enumerate(categories):
                        if i < 5:  # Regular categories
                            gt_count = gt_pixel_counts[i]
                        else:  # Total Flood
                            gt_count = gt_pixel_counts['flood']
                        
                        row_data = {
                            'Rainfall': rainfall_level,
                            'Window': window_idx,
                            'Category': category,
                            'Ground_Truth': gt_count,
                            'Ground_Truth_Pct': (gt_count / total_pixels) * 100
                        }
                        
                        # Add model predictions
                        for j, (model_name, pixel_counts) in enumerate(zip(model_paths.keys(), window_pixel_counts)):
                            if i < 5:  # Regular categories
                                pred_count = pixel_counts[i]
                            else:  # Total Flood
                                pred_count = pixel_counts['flood']
                            
                            row_data[f'Pred_{model_name}'] = pred_count
                            row_data[f'Pred_{model_name}_Pct'] = (pred_count / total_pixels) * 100
                        
                        # Add this row to the overall pixel counts DataFrame
                        overall_pixel_counts_df = pd.concat([overall_pixel_counts_df, pd.DataFrame([row_data])], ignore_index=True)
                    
                    # Generate a sample visualization for this window
                    if window_idx % 5 == 0:  # Only visualize every 5th window to avoid too many images
                        plot_sample_predictions(
                            dem, slope, flood_cat, 
                            window_predictions, 
                            list(model_paths.keys()), 
                            output_dir,
                            window_idx=window_idx,
                            rainfall_level=rainfall_level
                        )
        
        # Store results for this rainfall level
        all_rainfall_results[rainfall_level] = {
            'results': rainfall_results,
            'predictions': rainfall_predictions,
            'pixel_counts': rainfall_pixel_counts
        }
        
        # Calculate average metrics across all windows for this rainfall level
        if rainfall_results:
            # Aggregate results across windows
            avg_results = []
            for model_idx in range(len(model_paths)):
                avg_jaccard = {
                    'binary': np.mean([window_results[model_idx]['binary'] for window_results in rainfall_results]),
                    'no_flood': np.mean([window_results[model_idx]['no_flood'] for window_results in rainfall_results]),
                    'nuisance': np.mean([window_results[model_idx]['nuisance'] for window_results in rainfall_results]),
                    'minor': np.mean([window_results[model_idx]['minor'] for window_results in rainfall_results]),
                    'medium': np.mean([window_results[model_idx]['medium'] for window_results in rainfall_results]),
                    'major': np.mean([window_results[model_idx]['major'] for window_results in rainfall_results]),
                }
                avg_results.append(avg_jaccard)
            
            # Plot histograms for this rainfall level using the average results
            plot_histograms(avg_results, list(model_paths.keys()), output_dir, rainfall_level=rainfall_level)
            
            # Create a CSV for this rainfall level
            rainfall_csv_path = os.path.join(output_dir, f'model_evaluation_{rainfall_level}_{timestamp}.csv')
            rainfall_results_df = pd.DataFrame({
                'Model': list(model_paths.keys()),
                'Epochs': [model_epochs[name] for name in model_paths.keys()],
                'Jaccard_No_Flood': [results['no_flood'] for results in avg_results],
                'Jaccard_Flood': [results['binary'] for results in avg_results],
                'Jaccard_Nuisance': [results['nuisance'] for results in avg_results],
                'Jaccard_Minor': [results['minor'] for results in avg_results],
                'Jaccard_Moderate': [results['medium'] for results in avg_results],
                'Jaccard_Major': [results['major'] for results in avg_results]
            })
            rainfall_results_df.to_csv(rainfall_csv_path, index=False)
            print(f"  Results for rainfall level {rainfall_level} saved to {rainfall_csv_path}")
    
    # Save the overall results
    overall_results_df.to_csv(overall_csv_path, index=False)
    overall_pixel_counts_df.to_csv(overall_pixel_csv_path, index=False)
    
    print(f"\nOverall results saved to:")
    print(f"- {overall_csv_path}")
    print(f"- {overall_pixel_csv_path}")
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
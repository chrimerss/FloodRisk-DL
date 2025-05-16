#!/usr/bin/env python
# coding: utf-8

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from datetime import datetime
import argparse
from pathlib import Path
import joblib
import rasterio
from rasterio.transform import Affine
from rasterio.windows import Window
import json

# Add parent directory to path for imports
sys.path.append('/home/users/li1995/global_flood/FloodRisk-DL/terratorch')

from test_metric import (
    load_model, classify_depths, calc_slope, load_full_dataset, 
    calculate_jaccard_scores, 
    run_bathtub_model, RAINFALL_DICT, FLOOD_COLORS, FloodCategory
)

# Import necessary TerraTorch components
from terratorch.tasks import SemanticSegmentationTask

# Import model arguments
from task_class import (
    model_args_res50, model_args_res101, model_args_res152, 
    model_args_tiny, model_args_100, model_args_300, model_args_600
)

# Import model paths
from model_pth import FloodCategory as ModelPaths

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_gaussian_weight_map(crop_size=512, sigma=0.5):
    """
    Create a Gaussian weight map for blending window predictions.
    
    Args:
        crop_size: Size of the window (default: 512)
        sigma: Standard deviation of the Gaussian kernel (default: 0.5)
        
    Returns:
        2D numpy array with Gaussian weights
    """
    y = np.linspace(-1, 1, crop_size)
    x = np.linspace(-1, 1, crop_size)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2) / np.sqrt(2)
    
    # Create Gaussian weight with specified sigma
    weights = np.exp(-(distance**2) / (2 * sigma**2))
    
    # Normalize weights
    weights = weights / np.max(weights)
    return weights



def extract_rainfall_levels(city_id):
    # Define the preferred order of return periods
    return_periods_ordered = [
        "1000-yr", "500-yr", "200-yr", "100-yr", "50-yr",
        "25-yr", "10-yr", "5-yr", "2-yr", "1-yr"
    ]
    with open("/home/users/li1995/global_flood/FloodBench/data/cities_rainfall.json", "r") as f:
        data = json.load(f)
    for entry in data:
        if entry["City ID"] == city_id:
            # Filter and order the rainfall values based on return periods
            rainfall_levels = [
                entry[rp].replace(" ", "") for rp in return_periods_ordered if rp in entry
            ]
            return rainfall_levels
    return []

def load_meta_model(meta_model_path):
    """
    Load the trained meta-model.
    
    Args:
        meta_model_path: Path to the saved meta-model
        
    Returns:
        Loaded meta-model
    """
    print(f"Loading meta-model from: {meta_model_path}")
    try:
        meta_model = joblib.load(meta_model_path)
        print("Meta-model loaded successfully.")
        print(f"Model type: {type(meta_model).__name__}")
        return meta_model
    except Exception as e:
        print(f"Error loading meta-model: {str(e)}")
        return None

def load_base_models():
    """
    Load all base deep learning models.
    
    Returns:
        Dictionary of models {model_name: model}
    """
    models = {}
    # Define model configurations
    model_config = {
        'RES50': model_args_res50,
        'RES101': model_args_res101,
        'RES152': model_args_res152,
        'TINY': model_args_tiny,
        '100M': model_args_100,
        '300M': model_args_300,
        '600M': model_args_600
    }
    
    # Load each model
    for model_name, model_args in model_config.items():
        try:
            model_path = getattr(ModelPaths, 'MODEL_'+model_name).value
            if os.path.exists(model_path):
                print(f"Loading {model_name} model...")
                models[model_name] = load_model(model_path, model_name)
            else:
                print(f"Model path not found for {model_name}: {model_path}")
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
    
    print(f"Loaded {len(models)} base models.")
    return models

def get_model_predictions(domain, rainfall_level, models, window_size=512, overlap=128, sigma=0.5):
    """
    Get predictions from all base models for a given domain and rainfall level,
    using Gaussian kernel smoothing at window boundaries.
    
    Args:
        domain: Domain name (e.g., 'HOU001')
        rainfall_level: Rainfall level (e.g., '100mm')
        models: Dictionary of loaded models
        window_size: Size of the window for processing (default: 512)
        overlap: Overlap between adjacent windows in pixels (default: 128)
        sigma: Standard deviation for the Gaussian kernel (default: 0.5)
        
    Returns:
        Dictionary with predictions and metadata
    """
    try:
        # Load the dataset
        print(f"Loading domain {domain} with rainfall {rainfall_level}...")
        dem, slope, rainfall, flood_cat, target, dem_mean, dem_std = load_full_dataset(domain, rainfall_level)
        
        # Get metadata for GeoTIFF export
        input_dir = f'/home/users/li1995/global_flood/UrbanFloods2D/dataset/{domain}'
        dem_input = os.path.join(input_dir, f"{domain}_DEM.tif")
        geotiff_meta = None
        try:
            with rasterio.open(dem_input) as src:
                geotiff_meta = {
                    'crs': src.crs,
                    'transform': src.transform,
                    'width': src.width,
                    'height': src.height,
                    'count': 1,
                    'dtype': 'uint8',
                    'nodata': 255
                }
        except Exception as e:
            print(f"Warning: Could not obtain GeoTIFF metadata from source file: {e}")
        
        height, width = dem.shape
        print(f"Dataset dimensions: {height} x {width}")
        
        # Initialize dictionary of predictions
        predictions = {}
        probs = {}  # For storing class probabilities
        
        # Get Gaussian weight map for window blending
        weight_map = create_gaussian_weight_map(window_size, sigma)
        
        # Calculate stride based on overlap
        stride = window_size - overlap
        
        # Calculate number of windows needed
        n_windows_y = max(1, int(np.ceil((height - overlap) / stride))) if height > window_size else 1
        n_windows_x = max(1, int(np.ceil((width - overlap) / stride))) if width > window_size else 1
        
        print(f"Using {n_windows_y}x{n_windows_x} windows with size {window_size}, stride={stride}, overlap={overlap}")
        
        # Process each ML model
        for model_name, model in models.items():
            try:
                print(f"  Running {model_name} model with Gaussian smoothing...")
                # Initialize output arrays for accumulating weighted predictions
                # We store class probabilities for better blending
                probs_out = np.zeros((5, height, width), dtype=np.float32)  # Store class probabilities
                weights_sum = np.zeros((height, width), dtype=np.float32)  # Store accumulated weights
                
                # Process image in windows with overlap
                for y_idx in range(n_windows_y):
                    for x_idx in range(n_windows_x):
                        # Calculate window coordinates
                        y_start = min(y_idx * stride, height - window_size) if height > window_size else 0
                        x_start = min(x_idx * stride, width - window_size) if width > window_size else 0
                        y_end = min(y_start + window_size, height)
                        x_end = min(x_start + window_size, width)
                        
                        # Extract window data
                        dem_window = dem[y_start:y_end, x_start:x_end]
                        slope_window = slope[y_start:y_end, x_start:x_end]
                        rainfall_window = rainfall[y_start:y_end, x_start:x_end]
                        
                        # Handle window size if smaller than crop_size
                        window_h, window_w = dem_window.shape
                        if window_h < window_size or window_w < window_size:
                            pad_h = max(0, window_size - window_h)
                            pad_w = max(0, window_size - window_w)
                            dem_window = np.pad(dem_window, ((0, pad_h), (0, pad_w)), mode='reflect')
                            slope_window = np.pad(slope_window, ((0, pad_h), (0, pad_w)), mode='reflect')
                            rainfall_window = np.pad(rainfall_window, ((0, pad_h), (0, pad_w)), mode='reflect')
                        
                        # Prepare input tensor
                        model_input = np.stack([dem_window, slope_window, rainfall_window])
                        input_tensor = torch.from_numpy(model_input).unsqueeze(0).to(DEVICE)
                        
                        # Get prediction
                        with torch.no_grad():
                            output = model(input_tensor)
                            pred = output.output.squeeze().detach().cpu().numpy()
                        
                        # Extract actual window dimensions
                        window_h = y_end - y_start
                        window_w = x_end - x_start
                        current_weight = weight_map[:window_h, :window_w]
                        
                        # Apply weights to prediction probabilities and add to accumulated predictions
                        for c in range(pred.shape[0]):  # For each class channel
                            weighted_pred = pred[c, :window_h, :window_w] * current_weight
                            probs_out[c, y_start:y_end, x_start:x_end] += weighted_pred
                        
                        # Add weights to sum for later normalization
                        weights_sum[y_start:y_end, x_start:x_end] += current_weight
                        
                        # Clear CUDA cache to free memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Normalize the predictions by the weight sum
                for c in range(probs_out.shape[0]):
                    probs_out[c] = np.divide(probs_out[c], weights_sum, 
                                            out=np.zeros_like(probs_out[c]), 
                                            where=weights_sum > 0)
                
                # Get class with highest probability for each pixel
                pred_cat = np.argmax(probs_out, axis=0).astype(np.int64)
                
                # Store results
                predictions[model_name] = pred_cat
                probs[model_name] = probs_out
                
                print(f"  Completed {model_name} model inference")
                
            except Exception as e:
                print(f"  Error running model {model_name}: {e}")
                import traceback
                traceback.print_exc()
                # Initialize with zeros if model fails
                predictions[model_name] = np.zeros_like(flood_cat, dtype=np.int64)
                probs[model_name] = np.zeros((5, height, width), dtype=np.float32)
        
        # Calculate jaccard scores and F1 scores
        jaccard_scores = {}
        for model_name, pred in predictions.items():
            jaccard_scores[model_name] = calculate_jaccard_scores(flood_cat, pred)
        
        return {
            'dem': dem,
            'slope': slope,
            'rainfall': rainfall,
            'flood_cat': flood_cat,
            'target': target,
            'predictions': predictions,
            'probabilities': probs,
            'jaccard_scores': jaccard_scores,
            'dem_mean': dem_mean,
            'dem_std': dem_std,
            'geotiff_meta': geotiff_meta
        }
    
    except Exception as e:
        print(f"Error getting predictions for {domain}, {rainfall_level}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'dem': np.array([]),
            'slope': np.array([]),
            'rainfall': np.array([]),
            'flood_cat': np.array([]),
            'target': np.array([]),
            'predictions': {},
            'probabilities': {},
            'jaccard_scores': {},
            'dem_mean': 0,
            'dem_std': 1,
            'geotiff_meta': None
        }

def extract_features(pred_result):
    """
    Extract features for meta-model inference from base model predictions.
    
    Args:
        pred_result: Dictionary with prediction results
        
    Returns:
        Feature array for meta-model input
    """
    # Extract base data and probabilities
    dem = pred_result['dem']
    slope = pred_result['slope']
    rainfall = pred_result['rainfall']
    probs = pred_result['probabilities']
    
    height, width = dem.shape
    
    # We want to extract features for the entire image
    # Features: DEM, slope, rainfall, model probabilities
    num_base_features = 3  # DEM, slope, rainfall
    
    # Get available DL models
    dl_models = list(probs.keys())
    num_model_features = len(dl_models) * 5  # 5 class probabilities per model
    num_features = num_base_features + num_model_features
    
    # Reshape the input data to a 2D array (pixels x features)
    X = np.zeros((height * width, num_features), dtype=np.float32)
    
    # Add base features (first 3 columns)
    X[:, 0] = dem.flatten()
    X[:, 1] = slope.flatten()
    X[:, 2] = rainfall.flatten()
    
    # Add model probabilities (remaining columns)
    feature_idx = num_base_features
    for model_name in dl_models:
        model_probs = probs[model_name]
        for class_idx in range(5):
            X[:, feature_idx] = model_probs[class_idx].flatten()
            feature_idx += 1
    
    return X

def run_meta_model_inference(domain, rainfall_level, meta_model, base_models, output_dir, 
                           window_size=512, overlap=128, sigma=0.5, save_geotiffs=True):
    """
    Run inference using the meta-model on a specific domain and rainfall level.
    
    Args:
        domain: Domain name
        rainfall_level: Rainfall level
        meta_model: Loaded meta-model
        base_models: Dictionary of loaded base models
        output_dir: Directory to save results
        window_size: Size of the window for processing (default: 512)
        overlap: Overlap between adjacent windows (default: 128)
        sigma: Sigma for Gaussian kernel (default: 0.5)
        save_geotiffs: Whether to save predictions as GeoTIFF files (default: True)
        
    Returns:
        Dictionary with meta-model evaluation results
    """
    print(f"\nRunning meta-model inference on {domain}, rainfall {rainfall_level}")
    
    # Get base model predictions with Gaussian kernel smoothing
    pred_result = get_model_predictions(domain, rainfall_level, base_models, 
                                      window_size=window_size, overlap=overlap, sigma=sigma)
    
    if pred_result['dem'].size == 0:
        print(f"No valid predictions for {domain}, {rainfall_level}. Skipping.")
        return None
    
    # Extract features for meta-model
    print("Extracting features for meta-model...")
    X = extract_features(pred_result)
    
    # Run meta-model prediction
    print("Running meta-model prediction...")
    height, width = pred_result['dem'].shape
    try:
        start_time = time.time()
        meta_pred = meta_model.predict(X)
        inference_time = time.time() - start_time
        print(f"Meta-model prediction completed in {inference_time:.2f} seconds")
        
        # Reshape prediction to 2D array
        meta_pred = meta_pred.reshape(height, width)
        
        # Calculate meta-model performance metrics
        meta_jaccard = calculate_jaccard_scores(pred_result['flood_cat'], meta_pred)
        
        # Store meta prediction in results
        pred_result['meta_prediction'] = meta_pred
        pred_result['meta_jaccard'] = meta_jaccard
        
        # Create and save visualization
        create_visualization(domain, rainfall_level, pred_result, meta_jaccard, output_dir)
        
        # Print binary jaccard and F1 score results
        print("\nMeta-model performance:")
        print(f"Binary Jaccard: {meta_jaccard['binary']:.4f}")
        print(f"Binary F1 Score: {meta_jaccard['binary_f1']:.4f}")
        
        # Print performance of all base models for comparison
        print("\nBase model performance:")
        for model_name, jaccard in pred_result['jaccard_scores'].items():
            print(f"{model_name} - Binary Jaccard: {jaccard['binary']:.4f}, Binary F1: {jaccard['binary_f1']:.4f}")
        
        # Save predictions as GeoTIFF files
        if save_geotiffs and pred_result['geotiff_meta'] is not None:
            print("\nSaving predictions as GeoTIFF files...")
            geotiff_dir = os.path.join(output_dir, 'geotiffs', domain, rainfall_level)
            os.makedirs(geotiff_dir, exist_ok=True)
            
            # Save meta-model prediction
            meta_geotiff_path = os.path.join(geotiff_dir, f'meta_model_prediction.tif')
            save_geotiff(meta_pred, meta_geotiff_path, pred_result['geotiff_meta'])
            
            # Save ground truth
            gt_geotiff_path = os.path.join(geotiff_dir, f'ground_truth.tif')
            save_geotiff(pred_result['flood_cat'], gt_geotiff_path, pred_result['geotiff_meta'])
            
            # Save base model predictions (top 3 only to save space)
            base_models_to_save = 3
            if pred_result['jaccard_scores']:
                # Sort models by binary F1 score
                model_scores = [(model, scores['binary_f1']) 
                               for model, scores in pred_result['jaccard_scores'].items()]
                model_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Save top models
                for i, (model_name, _) in enumerate(model_scores[:base_models_to_save]):
                    model_geotiff_path = os.path.join(geotiff_dir, f'{model_name}_prediction.tif')
                    save_geotiff(pred_result['predictions'][model_name], 
                               model_geotiff_path, pred_result['geotiff_meta'])
            
            print(f"Saved GeoTIFF files to {geotiff_dir}")
        
        return pred_result
    
    except Exception as e:
        print(f"Error in meta-model inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_visualization(domain, rainfall_level, pred_result, meta_jaccard, output_dir):
    """
    Create visualization comparing meta-model with base models.
    
    Args:
        domain: Domain name
        rainfall_level: Rainfall level
        pred_result: Dictionary with prediction results
        meta_jaccard: Meta-model Jaccard scores
        output_dir: Directory to save visualizations
    """
    print(f"Creating visualization for {domain}, {rainfall_level}")
    
    try:
        # Create colormap for flood categories
        flood_colors = [FLOOD_COLORS[FloodCategory(i)] for i in range(len(FloodCategory))]
        cmap = ListedColormap(flood_colors)
        
        # Get data
        flood_cat = pred_result['flood_cat']
        meta_pred = pred_result['meta_prediction']
        base_preds = pred_result['predictions']
        
        # Check if the visualization would be too large
        height, width = flood_cat.shape
        total_pixels = height * width
        max_viz_pixels = 2000000  # 2 million pixels max for visualization
        
        # If the image is too large, downsample it
        if total_pixels > max_viz_pixels:
            # Calculate downsample ratio
            downsample_ratio = int(np.ceil(np.sqrt(total_pixels / max_viz_pixels)))
            print(f"Image too large for visualization ({height}x{width}). Downsampling by factor of {downsample_ratio}.")
            
            # Downsample data
            flood_cat = flood_cat[::downsample_ratio, ::downsample_ratio]
            meta_pred = meta_pred[::downsample_ratio, ::downsample_ratio]
            
            # Downsample base predictions
            downsampled_base_preds = {}
            for model_name, pred in base_preds.items():
                downsampled_base_preds[model_name] = pred[::downsample_ratio, ::downsample_ratio]
            base_preds = downsampled_base_preds
        
        # Calculate how many subplots we need
        n_models = len(base_preds)
        total_plots = n_models + 2  # +2 for ground truth and meta-model
        
        # Limit the number of subplots to avoid memory issues
        max_plots = 9  # Max 9 subplots (3x3 grid)
        if total_plots > max_plots:
            print(f"Too many models to visualize ({n_models}). Limiting to {max_plots-2} top models.")
            
            # Find the best performing models based on binary F1 score
            model_scores = [(model, scores['binary_f1']) for model, scores in pred_result['jaccard_scores'].items()]
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only the top models
            top_models = [model for model, _ in model_scores[:max_plots-2]]
            
            # Filter base_preds to only include top models
            base_preds = {model: pred for model, pred in base_preds.items() if model in top_models}
            
            # Update n_models and total_plots
            n_models = len(base_preds)
            total_plots = n_models + 2
        
        # Create a grid of subplots
        ncols = min(3, total_plots)
        nrows = (total_plots + ncols - 1) // ncols
        
        # Create figure with smaller dpi to reduce memory usage
        plt.figure(figsize=(ncols*4, nrows*3), dpi=100)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), dpi=100)
        axes = axes.flatten() if total_plots > 1 else [axes]
        
        # Hide any unused subplots
        for i in range(total_plots, len(axes)):
            axes[i].axis('off')
        
        # Plot ground truth
        axes[0].set_title("Ground Truth")
        axes[0].imshow(flood_cat, cmap=cmap, vmin=0, vmax=4)
        
        # Plot meta-model prediction
        binary_f1 = meta_jaccard['binary_f1']
        axes[1].set_title(f"Meta-Model - F1: {binary_f1:.3f}")
        axes[1].imshow(meta_pred, cmap=cmap, vmin=0, vmax=4)
        
        # Plot base model predictions
        for i, (model_name, pred) in enumerate(base_preds.items()):
            idx = i + 2  # Offset for ground truth and meta-model
            binary_f1 = pred_result['jaccard_scores'][model_name]['binary_f1']
            axes[idx].set_title(f"{model_name} - F1: {binary_f1:.3f}")
            axes[idx].imshow(pred, cmap=cmap, vmin=0, vmax=4)
        
        # Add overall title
        fig.suptitle(f"Model Comparison: {domain}, Rainfall: {rainfall_level}", fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'meta_inference_{domain}_{rainfall_level}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close('all')  # Close all figures to free memory
        
        print(f"Saved visualization to {save_path}")
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def save_geotiff(data, output_path, meta=None, nodata=255):
    """
    Save a prediction array as a GeoTIFF file.
    
    Args:
        data: 2D numpy array with prediction data
        output_path: Path to save the GeoTIFF file
        meta: Dictionary with metadata for the GeoTIFF (crs, transform, etc.)
        nodata: Value to use for nodata pixels (default: 255)
    """
    try:
        print(f"Saving GeoTIFF to {output_path}")
        
        if meta is None:
            # Create default metadata if none provided
            height, width = data.shape
            meta = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,
                'dtype': 'uint8',
                'crs': None,
                'transform': Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),  # Default transform
                'nodata': nodata
            }
        else:
            # Ensure we have all required metadata
            meta = meta.copy()
            meta.update({
                'driver': 'GTiff',
                'count': 1,
                'dtype': 'uint8',
                'nodata': nodata
            })
        
        # Convert to uint8 (required for most GIS software to recognize categorical data)
        data_uint8 = data.astype(np.uint8)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write to GeoTIFF
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data_uint8, 1)
            
        print(f"Successfully saved GeoTIFF to {output_path}")
        
    except Exception as e:
        print(f"Error saving GeoTIFF: {str(e)}")
        import traceback
        traceback.print_exc()

def save_results(domain, rainfall_level, pred_result, output_dir):
    """
    Save inference results to CSV files.
    
    Args:
        domain: Domain name
        rainfall_level: Rainfall level
        pred_result: Dictionary with prediction results
        output_dir: Directory to save results
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract jaccard scores
        jaccard_data = []
        
        # Add meta-model row
        meta_jaccard = pred_result['meta_jaccard']
        jaccard_data.append({
            'Domain': domain,
            'Rainfall': rainfall_level,
            'Model': 'Meta-Model',
            'Binary_Jaccard': meta_jaccard['binary'],
            'NoFlood_Jaccard': meta_jaccard['no_flood'],
            'Nuisance_Jaccard': meta_jaccard['nuisance'],
            'Minor_Jaccard': meta_jaccard['minor'],
            'Medium_Jaccard': meta_jaccard['medium'],
            'Major_Jaccard': meta_jaccard['major'],
            'Binary_F1': meta_jaccard['binary_f1'],
            'NoFlood_F1': meta_jaccard['no_flood_f1'],
            'Nuisance_F1': meta_jaccard['nuisance_f1'],
            'Minor_F1': meta_jaccard['minor_f1'],
            'Medium_F1': meta_jaccard['medium_f1'],
            'Major_F1': meta_jaccard['major_f1']
        })
        
        # Add rows for each base model
        for model_name, model_jaccard in pred_result['jaccard_scores'].items():
            jaccard_data.append({
                'Domain': domain,
                'Rainfall': rainfall_level,
                'Model': model_name,
                'Binary_Jaccard': model_jaccard['binary'],
                'NoFlood_Jaccard': model_jaccard['no_flood'],
                'Nuisance_Jaccard': model_jaccard['nuisance'],
                'Minor_Jaccard': model_jaccard['minor'],
                'Medium_Jaccard': model_jaccard['medium'],
                'Major_Jaccard': model_jaccard['major'],
                'Binary_F1': model_jaccard['binary_f1'],
                'NoFlood_F1': model_jaccard['no_flood_f1'],
                'Nuisance_F1': model_jaccard['nuisance_f1'],
                'Minor_F1': model_jaccard['minor_f1'],
                'Medium_F1': model_jaccard['medium_f1'],
                'Major_F1': model_jaccard['major_f1']
            })
        
        # Save to CSV
        jaccard_df = pd.DataFrame(jaccard_data)
        csv_path = os.path.join(output_dir, f'meta_inference_results_{domain}_{rainfall_level}_{timestamp}.csv')
        jaccard_df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def main(args):
    """Main function to run meta-model inference."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"meta_inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Window size: {args.window_size}, Overlap: {args.overlap}, Sigma: {args.sigma}")
    print(f"Save GeoTIFF files: {args.save_geotiffs}")
    
    # Load meta-model
    meta_model = load_meta_model(args.meta_model)
    if meta_model is None:
        print("Failed to load meta-model. Exiting.")
        return
    
    # Load base models
    base_models = load_base_models()
    if len(base_models) == 0:
        print("Failed to load any base models. Exiting.")
        return
    
    # Get list of domains to process
    domains = args.domains.split(',') if args.domains else ["HOU001", "HOU007", "SF001", "SF002", "NYC002", "LA002", "DAL002", "AUS002", "MIA002"]

    
    
    
    # Process each domain and rainfall level
    for domain in domains:
        rainfall_levels = extract_rainfall_levels(domain)
        for rainfall_level in rainfall_levels:
            print(f"\n{'='*80}")
            print(f"Processing domain: {domain}, rainfall: {rainfall_level}")
            print(f"{'='*80}")
            
            # Run meta-model inference with Gaussian kernel smoothing
            result = run_meta_model_inference(
                domain, rainfall_level, meta_model, base_models, output_dir,
                window_size=args.window_size, 
                overlap=args.overlap,
                sigma=args.sigma,
                save_geotiffs=args.save_geotiffs
            )
            
            if result:
                save_results(domain, rainfall_level, result, output_dir)
                
                # Clear memory
                del result
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    print("\nInference completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference using a trained meta-model')
    parser.add_argument('--meta_model', type=str, 
                        default='/home/users/li1995/global_flood/FloodRisk-DL/terratorch/meta_learning_results_20250512_101047/meta_model_random_forest.joblib',
                        help='Path to the trained meta-model')
    parser.add_argument('--output_dir', type=str, default='./meta_inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--domains', type=str, default=None,
                        help='Comma-separated list of domains to process (e.g., HOU001,SF001)')
    parser.add_argument('--window_size', type=int, default=512,
                        help='Size of window for processing (default: 512)')
    parser.add_argument('--overlap', type=int, default=128,
                        help='Overlap between adjacent windows in pixels (default: 128)')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Standard deviation for Gaussian kernel (default: 0.5)')
    parser.add_argument('--save_geotiffs', action='store_true',
                        help='Save prediction results as GeoTIFF files')
    parser.add_argument('--no_save_geotiffs', dest='save_geotiffs', action='store_false',
                        help='Do not save prediction results as GeoTIFF files')
    parser.set_defaults(save_geotiffs=True)
    
    args = parser.parse_args()
    main(args)

#!/usr/bin/env python
# coding: utf-8

"""
Window-by-window DEM inference with immediate meta-learning
This implementation processes each window completely (all models + meta-learning) 
before moving to the next window for better memory efficiency.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from datetime import datetime
import argparse
from pathlib import Path
import joblib
import rasterio
from rasterio.transform import Affine
import json
import warnings
import xarray as xr
import tempfile
import threading
from numba import njit, prange

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_metric import (
    load_model, calc_slope,
    RAINFALL_DICT, FLOOD_COLORS, FloodCategory
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

def create_gaussian_weight_map(window_size=512, sigma=0.5):
    """
    Create a Gaussian weight map for blending window predictions.
    
    Args:
        window_size: Size of the window (default: 512)
        sigma: Standard deviation of the Gaussian kernel (default: 0.5)
        
    Returns:
        2D numpy array with Gaussian weights
    """
    y = np.linspace(-1, 1, window_size)
    x = np.linspace(-1, 1, window_size)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2) / np.sqrt(2)
    
    # Create Gaussian weight with specified sigma
    weights = np.exp(-(distance**2) / (2 * sigma**2))
    
    # Normalize weights
    weights = weights / np.max(weights)
    return weights

def load_meta_model(meta_model_path):
    """Load the trained meta-model."""
    print(f"Loading meta-model from: {meta_model_path}")
    try:
        meta_model = joblib.load(meta_model_path)
        print("Meta-model loaded successfully.")
        print(f"Model type: {type(meta_model).__name__}")
        return meta_model
    except Exception as e:
        print(f"Error loading meta-model: {str(e)}")
        return None

def load_all_base_models():
    """
    Load all base deep learning models in advance.
    
    Returns:
        Dictionary of models {model_name: model}
    """
    print("Loading all base models in advance...")
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
                print(f"  Loading {model_name} model...")
                models[model_name] = load_model(model_path, model_name)
            else:
                print(f"  Model path not found for {model_name}: {model_path}")
        except Exception as e:
            print(f"  Error loading {model_name}: {str(e)}")
    
    print(f"Loaded {len(models)} base models.")
    return models

def lazy_load_dem_from_zarr(zarr_file, rainfall_mm):
    """
    Lazy load DEM data from a Zarr file (step 1).
    
    Args:
        zarr_file: Path to the DEM Zarr file
        rainfall_mm: Rainfall amount in mm
        
    Returns:
        Dictionary with lazy-loaded DEM data and metadata
    """
    try:
        print(f"Lazy loading DEM from Zarr file: {zarr_file}")
        
        # Load Zarr file using xarray (lazy loading)
        ds = xr.open_zarr(zarr_file)
        
        # Get the first data variable (should be the DEM data)
        if len(ds.data_vars) > 0:
            var_name = list(ds.data_vars.keys())[0]
            da = ds[var_name]
            print(f"Loaded data variable: {var_name}")
        else:
            da = ds
        
        print(f"DataArray type: {type(da)}")
        print(f"DataArray shape: {da.shape}")
        
        # Keep as dask array - do NOT compute the entire array
        height, width = da.shape
        print(f"DEM dimensions: {height} rows × {width} columns")
        
        # Extract metadata
        crs_str = da.attrs.get('crs', 'EPSG:4326')
        transform_list = da.attrs.get('transform', None)
        nodata_value = da.attrs.get('nodata', -9999.0)
        
        print(f"CRS: {crs_str}")
        print(f"NoData value: {nodata_value}")
        
        # Create transform object
        if transform_list and len(transform_list) >= 6:
            transform = Affine(*transform_list[:6])
        else:
            # Calculate transform from coordinates
            x_coords = da.coords['x'].values
            y_coords = da.coords['y'].values
            x_res = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
            y_res = y_coords[0] - y_coords[1] if len(y_coords) > 1 else 1.0
            transform = Affine(x_res, 0, x_coords[0] - x_res/2, 0, -y_res, y_coords[0] + y_res/2)
        
        # Calculate statistics from a sample to avoid loading everything
        print("Calculating DEM statistics from sample data...")
        sample_step = 100
        sample_data = da[::sample_step, ::sample_step].values
        
        if nodata_value is not None and not np.isnan(nodata_value):
            sample_mask = np.logical_and(sample_data != nodata_value, ~np.isnan(sample_data))
        else:
            sample_mask = ~np.isnan(sample_data)
        
        sample_valid = sample_data[sample_mask]
        if len(sample_valid) > 0:
            dem_mean = np.mean(sample_valid)
            dem_std = np.std(sample_valid)
            print(f"DEM statistics (from sample) - Mean: {dem_mean:.2f}, Std: {dem_std:.2f}")
        else:
            print("Warning: No valid data found in sample!")
            dem_mean = 0
            dem_std = 1
        
        # Extract county information
        county_name = da.attrs.get('county', 'Unknown')
        state_name = da.attrs.get('state', 'Unknown')
        geoid = da.attrs.get('geoid', 'Unknown')
        
        print(f"County: {county_name}, State: {state_name}, GEOID: {geoid}")
        
        return {
            'dem_dask': da,  # Lazy-loaded dask array
            'rainfall_mm': rainfall_mm,
            'dem_mean': dem_mean,
            'dem_std': dem_std,
            'nodata_value': nodata_value,
            'height': height,
            'width': width,
            'transform': transform,
            'crs': crs_str,
            'county_info': {
                'county': county_name,
                'state': state_name,
                'geoid': geoid
            }
        }
    
    except Exception as e:
        print(f"Error lazy loading DEM data from Zarr: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

@njit
def extract_features_numba(dem_window, slope_window, window_mask, prob_output_window, 
                          rainfall_m, actual_window_h, actual_window_w, n_models):
    """
    Numba-optimized feature extraction for valid pixels.
    
    Returns:
        features_array: 2D array of shape (n_valid_pixels, n_features)
        valid_indices: 2D array of shape (n_valid_pixels, 2) with (y, x) coordinates
    """
    # Count valid pixels first
    n_valid = 0
    for y in range(actual_window_h):
        for x in range(actual_window_w):
            if window_mask[y, x]:
                n_valid += 1
    
    if n_valid == 0:
        return np.zeros((0, 3 + n_models * 5), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
    
    # Allocate arrays
    n_features = 3 + n_models * 5  # DEM, slope, rainfall + model probabilities
    features_array = np.zeros((n_valid, n_features), dtype=np.float32)
    valid_indices = np.zeros((n_valid, 2), dtype=np.int32)
    
    # Extract features for valid pixels (sequential for thread safety)
    idx = 0
    for y in range(actual_window_h):
        for x in range(actual_window_w):
            if window_mask[y, x]:
                # Store coordinates
                valid_indices[idx, 0] = y
                valid_indices[idx, 1] = x
                
                # Extract base features
                features_array[idx, 0] = dem_window[y, x]     # Normalized DEM
                features_array[idx, 1] = slope_window[y, x]   # Slope
                features_array[idx, 2] = rainfall_m           # Rainfall
                
                # Extract model probabilities
                feature_idx = 3
                for model_idx in range(n_models):
                    for class_idx in range(5):
                        features_array[idx, feature_idx] = prob_output_window[model_idx, class_idx, y, x]
                        feature_idx += 1
                
                idx += 1
    
    return features_array, valid_indices

@njit(parallel=True)
def majority_vote_predictions_numba(prob_output_window, valid_indices, n_models):
    """
    Numba-optimized majority vote fallback for failed meta-model predictions.
    """
    n_valid = valid_indices.shape[0]
    predictions = np.zeros(n_valid, dtype=np.uint8)
    
    for i in prange(n_valid):
        y = valid_indices[i, 0]
        x = valid_indices[i, 1]
        
        # Get predictions from all models
        votes = np.zeros(5, dtype=np.int32)
        for model_idx in range(n_models):
            # Find class with highest probability
            max_prob = -1.0
            pred_class = 0
            for class_idx in range(5):
                if prob_output_window[model_idx, class_idx, y, x] > max_prob:
                    max_prob = prob_output_window[model_idx, class_idx, y, x]
                    pred_class = class_idx
            votes[pred_class] += 1
        
        # Find majority vote
        max_votes = -1
        final_pred = 0
        for class_idx in range(5):
            if votes[class_idx] > max_votes:
                max_votes = votes[class_idx]
                final_pred = class_idx
        
        predictions[i] = final_pred
    
    return predictions

@njit(parallel=True)
def blend_window_predictions_numba(current_output, current_weights, new_predictions, new_weights):
    """
    Numba-optimized blending of window predictions with Gaussian weights.
    
    Args:
        current_output: Current prediction values
        current_weights: Current weight values  
        new_predictions: New prediction values
        new_weights: New weight values
        
    Returns:
        updated_output, updated_weights: Blended arrays
    """
    height, width = current_output.shape
    updated_output = current_output.copy()
    updated_weights = current_weights.copy()
    
    for y in prange(height):
        for x in range(width):
            if new_predictions[y, x] != 255:  # Valid new prediction
                old_weight = current_weights[y, x]
                new_weight = new_weights[y, x]
                total_weight = old_weight + new_weight
                
                if total_weight > 0:
                    if old_weight > 0 and current_output[y, x] != 255:
                        # Weighted blend of old and new predictions
                        old_pred = float(current_output[y, x])
                        new_pred = float(new_predictions[y, x])
                        blended_pred = (old_pred * old_weight + new_pred * new_weight) / total_weight
                        updated_output[y, x] = np.uint8(np.round(blended_pred))
                    else:
                        # First prediction for this pixel
                        updated_output[y, x] = new_predictions[y, x]
                    
                    # Update weight
                    updated_weights[y, x] = total_weight
    
    return updated_output, updated_weights

def extract_features_and_predict_vectorized(dem_window, slope_window, window_mask, prob_output_window, 
                                           rainfall_m, actual_window_h, actual_window_w, n_models, meta_model):
    """
    Vectorized feature extraction and meta-model prediction.
    
    Returns:
        flood_window: 2D array with flood predictions
    """
    # Initialize output
    flood_window = np.ones((actual_window_h, actual_window_w), dtype=np.uint8) * 255  # NoData
    
    # Extract features for all valid pixels using numba
    features_array, valid_indices = extract_features_numba(
        dem_window, slope_window, window_mask, prob_output_window,
        rainfall_m, actual_window_h, actual_window_w, n_models
    )
    
    if features_array.shape[0] == 0:
        return flood_window
    
    try:
        # Vectorized meta-model prediction
        predictions = meta_model.predict(features_array)
        
        # Place predictions back into the window
        for i in range(len(predictions)):
            y = valid_indices[i, 0]
            x = valid_indices[i, 1]
            flood_window[y, x] = predictions[i]
            
    except Exception as e:
        print(f"Meta-model prediction failed, using majority vote fallback: {e}")
        # Use majority vote fallback
        predictions = majority_vote_predictions_numba(prob_output_window, valid_indices, n_models)
        
        # Place predictions back into the window
        for i in range(len(predictions)):
            y = valid_indices[i, 0]
            x = valid_indices[i, 1]
            flood_window[y, x] = predictions[i]
    
    return flood_window

def process_window_with_all_models_and_metalearning(
    window_coords, dem_data, base_models, meta_model, 
    window_size=512, overlap=128, sigma=0.5
):
    """
    Process a single window with all models and meta-learning (steps 3.1-3.5).
    
    Args:
        window_coords: (y_start, x_start, y_end, x_end)
        dem_data: DEM data dictionary
        base_models: Dictionary of loaded base models
        meta_model: Loaded meta-model
        window_size: Window size for processing
        overlap: Overlap size
        sigma: Gaussian kernel sigma
        
    Returns:
        Tuple of (flood_prediction_window, gaussian_weights)
    """
    y_start, x_start, y_end, x_end = window_coords
    
    # Extract needed data
    dem_dask = dem_data['dem_dask']
    rainfall_mm = dem_data['rainfall_mm']
    dem_mean = dem_data['dem_mean']
    dem_std = dem_data['dem_std']
    nodata_value = dem_data['nodata_value']
    height, width = dem_data['height'], dem_data['width']
    
    # Step 3.1: Load DEM window into memory
    pad_size = 2  # For slope calculation
    y_start_padded = max(0, y_start - pad_size)
    x_start_padded = max(0, x_start - pad_size)
    y_end_padded = min(height, y_end + pad_size)
    x_end_padded = min(width, x_end + pad_size)
    
    # Load the padded window data
    dem_raw_padded = dem_dask[y_start_padded:y_end_padded, x_start_padded:x_end_padded].values.astype(np.float32)
    
    # Create mask for valid data
    if nodata_value is not None and not np.isnan(nodata_value):
        mask_padded = np.logical_and(dem_raw_padded != nodata_value, ~np.isnan(dem_raw_padded))
    else:
        mask_padded = ~np.isnan(dem_raw_padded)
    
    # Replace invalid values
    dem_raw_padded[~mask_padded] = -9999
    
    # Step 3.2: Compute slope (with padding handling)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        slope_padded = calc_slope(dem_raw_padded)
        slope_padded[~mask_padded] = 0
    
    # Normalize DEM
    dem_norm_padded = dem_raw_padded.copy()
    dem_norm_padded[mask_padded] = (dem_raw_padded[mask_padded] - dem_mean) / dem_std
    dem_norm_padded[~mask_padded] = 0
    
    # Extract actual window area from padded data
    pad_y_offset = y_start - y_start_padded
    pad_x_offset = x_start - x_start_padded
    actual_window_h = y_end - y_start
    actual_window_w = x_end - x_start
    
    dem_window = dem_norm_padded[
        pad_y_offset:pad_y_offset + actual_window_h,
        pad_x_offset:pad_x_offset + actual_window_w
    ]
    
    slope_window = slope_padded[
        pad_y_offset:pad_y_offset + actual_window_h,
        pad_x_offset:pad_x_offset + actual_window_w
    ]
    
    window_mask = mask_padded[
        pad_y_offset:pad_y_offset + actual_window_h,
        pad_x_offset:pad_x_offset + actual_window_w
    ]
    
    # Step 3.3: Prepare other model inputs including rainfall
    rainfall_m = rainfall_mm / 1000.0
    rainfall_window = np.ones_like(dem_window) * rainfall_m
    rainfall_window[~window_mask] = 0
    
    # Handle window size padding if needed
    window_h, window_w = dem_window.shape
    if window_h < window_size or window_w < window_size:
        pad_h = max(0, window_size - window_h)
        pad_w = max(0, window_size - window_w)
        dem_window = np.pad(dem_window, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        slope_window = np.pad(slope_window, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        rainfall_window = np.pad(rainfall_window, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    # Step 3.4: Compute logits for each model, save to prob_output_window (7, 512, 512)
    prob_output_window = np.zeros((len(base_models), 5, window_size, window_size), dtype=np.float32)
    
    model_input = np.stack([dem_window, slope_window, rainfall_window])
    input_tensor = torch.from_numpy(model_input).unsqueeze(0).to(DEVICE)
    
    model_names = list(base_models.keys())
    
    with torch.no_grad():
        for i, (model_name, model) in enumerate(base_models.items()):
            try:
                output = model(input_tensor)
                pred_probs = output.output.squeeze().detach().cpu().numpy()
                prob_output_window[i] = pred_probs
            except Exception as e:
                print(f"Error with model {model_name}: {e}")
                # Fill with uniform probabilities if model fails
                prob_output_window[i] = np.ones((5, window_size, window_size)) * 0.2
    
    # Step 3.5: Use meta-learner to produce output for the window
    flood_window = extract_features_and_predict_vectorized(
        dem_window, slope_window, window_mask, prob_output_window, 
        rainfall_m, actual_window_h, actual_window_w, len(base_models), meta_model
    )
    
    # Create Gaussian weights for this window
    gaussian_weights = create_gaussian_weight_map(window_size, sigma)
    actual_weights = gaussian_weights[:actual_window_h, :actual_window_w]
    
    return flood_window, actual_weights


def run_window_by_window_inference(
    dem_data, base_models, meta_model, output_file,
    window_size=512, overlap=128, sigma=0.5
):
    """
    Main window-by-window inference function (steps 3-3.6).
    
    Args:
        dem_data: DEM data dictionary
        base_models: Dictionary of loaded base models  
        meta_model: Loaded meta-model
        output_file: Output file path
        window_size: Size of processing windows
        overlap: Overlap between windows
        sigma: Gaussian kernel sigma
        
    Returns:
        Final prediction array
    """
    print("\nRunning window-by-window inference...")
    
    height = dem_data['height']
    width = dem_data['width']
    
    # Step 3: Initiate output with same size as DEM
    final_output = np.ones((height, width), dtype=np.uint8) * 255  # Initialize with NoData
    weight_accumulator = np.zeros((height, width), dtype=np.float32)
    
    print(f"Initialized output array: {height} x {width}")
    
    # Calculate stride and window coordinates
    stride = window_size - overlap
    n_windows_y = max(1, int(np.ceil((height - overlap) / stride))) if height > window_size else 1
    n_windows_x = max(1, int(np.ceil((width - overlap) / stride))) if width > window_size else 1
    
    print(f"Processing {n_windows_y} x {n_windows_x} windows with size {window_size}, overlap {overlap}")
    
    total_windows = n_windows_y * n_windows_x
    processed_windows = 0
    
    # Step 3: Break down DEM into window level
    start_time = time.time()
    
    for y_idx in range(n_windows_y):
        for x_idx in range(n_windows_x):
            # Calculate window coordinates
            y_start = min(y_idx * stride, height - window_size) if height > window_size else 0
            x_start = min(x_idx * stride, width - window_size) if width > window_size else 0
            y_end = min(y_start + window_size, height)
            x_end = min(x_start + window_size, width)
            
            window_coords = (y_start, x_start, y_end, x_end)
            
            try:
                # Process window with all models and meta-learning
                flood_window, weights = process_window_with_all_models_and_metalearning(
                    window_coords, dem_data, base_models, meta_model,
                    window_size, overlap, sigma
                )
                
                # Step 3.6: Blend with existing output using Gaussian weights
                window_h, window_w = flood_window.shape
                
                # Get current weights for this region
                current_weights = weight_accumulator[y_start:y_start+window_h, x_start:x_start+window_w].copy()
                current_output = final_output[y_start:y_start+window_h, x_start:x_start+window_w].copy()
                
                # Use numba-optimized blending
                updated_output, updated_weights = blend_window_predictions_numba(
                    current_output, current_weights, flood_window, weights
                )
                
                # Update arrays
                final_output[y_start:y_start+window_h, x_start:x_start+window_w] = updated_output
                weight_accumulator[y_start:y_start+window_h, x_start:x_start+window_w] = updated_weights
                
                processed_windows += 1
                
                # Progress reporting
                if processed_windows % max(1, total_windows // 20) == 0:
                    elapsed = time.time() - start_time
                    progress = processed_windows / total_windows * 100
                    print(f"Processed {processed_windows}/{total_windows} windows ({progress:.1f}%) - "
                          f"Elapsed: {elapsed:.1f}s")
                
                # Clear GPU cache periodically
                if processed_windows % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing window {y_idx}, {x_idx}: {e}")
                processed_windows += 1
                continue
    
    total_time = time.time() - start_time
    print(f"Completed all {total_windows} windows in {total_time:.1f} seconds")
    
    # Save final output
    save_prediction_zarr(final_output, output_file, dem_data)
    
    return final_output


def save_prediction_zarr(prediction_array, output_file, dem_data):
    """Save prediction array as Zarr with spatial reference."""
    try:
        print(f"Saving prediction to: {output_file}")
        
        height, width = prediction_array.shape
        transform = dem_data['transform']
        crs = dem_data['crs']
        
        # Create coordinate arrays
        x_coords = np.array([transform * (i + 0.5, 0) for i in range(width)])[:, 0]
        y_coords = np.array([transform * (0, j + 0.5) for j in range(height)])[:, 1]
        
        # Create xarray DataArray
        da = xr.DataArray(
            prediction_array,
            coords={
                'y': ('y', y_coords),
                'x': ('x', x_coords)
            },
            dims=['y', 'x'],
            name='flood_prediction',
            attrs={
                'crs': crs,
                'transform': list(transform),
                'nodata': 255,
                'description': 'Flood prediction categories (0=No_Flood, 1=Minor_Flood, 2=Moderate_Flood, 3=Major_Flood, 4=Extreme_Flood)',
                'units': 'category',
                'long_name': 'Flood Risk Categories',
                'created': datetime.now().isoformat(),
                'source': 'Window-by-window meta-learning flood prediction model'
            }
        )
        
        # Set spatial reference
        da = da.rio.write_crs(crs)
        da = da.rio.write_transform(transform)
        da = da.rio.write_nodata(255)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save to Zarr
        da.to_zarr(output_file, mode='w')
        
        print(f"Successfully saved prediction to {output_file}")
        
    except Exception as e:
        print(f"Error saving prediction: {str(e)}")
        import traceback
        traceback.print_exc()

def main(args):
    """Main function for window-by-window DEM inference."""
    print(f"Window-by-window DEM Inference")
    print(f"Input DEM Zarr: {args.zarr_file}")
    print(f"Rainfall: {args.rainfall_mm}mm")
    print(f"Output file: {args.output_file}")
    print(f"Window size: {args.window_size}, Overlap: {args.overlap}, Sigma: {args.sigma}")
    
    # Step 1: Lazy load DEM from zarr file
    dem_data = lazy_load_dem_from_zarr(args.zarr_file, args.rainfall_mm)
    if dem_data is None:
        print("Failed to load DEM data. Exiting.")
        return
    
    # Step 2: Load all seven models in advance
    base_models = load_all_base_models()
    if len(base_models) == 0:
        print("Failed to load base models. Exiting.")
        return
    
    # Load meta-model
    meta_model = load_meta_model(args.meta_model)
    if meta_model is None:
        print("Failed to load meta-model. Exiting.")
        return
    
    # Run window-by-window inference
    final_prediction = run_window_by_window_inference(
        dem_data, base_models, meta_model, args.output_file,
        window_size=args.window_size,
        overlap=args.overlap,
        sigma=args.sigma
    )
    
    print(f"\nWindow-by-window inference completed successfully!")
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Window-by-window DEM inference with meta-learning')
    parser.add_argument('--zarr_file', type=str, required=True,
                        help='Path to the DEM Zarr file')
    parser.add_argument('--rainfall_mm', type=float, required=True,
                        help='Rainfall amount in millimeters')
    parser.add_argument('--meta_model', type=str, 
                        default='/home/users/li1995/global_flood/FloodRisk-DL/terratorch/meta_learning_results_20250515_143320/meta_model_random_forest.joblib',
                        help='Path to the trained meta-model')
    parser.add_argument('--output_file', type=str, default='./window_by_window_results.zarr',
                        help='Output file path for the result')
    parser.add_argument('--window_size', type=int, default=512,
                        help='Size of window for processing (default: 512)')
    parser.add_argument('--overlap', type=int, default=128,
                        help='Overlap between adjacent windows in pixels (default: 128)')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Standard deviation for Gaussian kernel (default: 0.5)')
    
    args = parser.parse_args()
    main(args)

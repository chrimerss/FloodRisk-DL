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
import warnings
import xarray as xr
import rioxarray as rxr

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_metric import (
    load_model, classify_depths, calc_slope, 
    calculate_jaccard_scores, 
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

def load_meta_model(meta_model_path, n_jobs=-1):
    """
    Load the trained meta-model and enable parallel prediction.
    
    Args:
        meta_model_path: Path to the saved meta-model
        n_jobs: Number of parallel jobs for prediction (-1 for all cores)
        
    Returns:
        Loaded meta-model with parallel prediction enabled
    """
    print(f"Loading meta-model from: {meta_model_path}")
    try:
        meta_model = joblib.load(meta_model_path)
        print("Meta-model loaded successfully.")
        print(f"Model type: {type(meta_model).__name__}")
        
        # Enable parallel prediction for scikit-learn models
        if hasattr(meta_model, 'n_jobs'):
            original_n_jobs = getattr(meta_model, 'n_jobs', None)
            print(f"Original n_jobs: {original_n_jobs}")
            
            # Set n_jobs for parallel prediction
            meta_model.n_jobs = n_jobs
            print(f"Enabled parallel prediction with n_jobs={n_jobs}")
        
        # For ensemble models like RandomForest, also check if we can set parallel prediction
        # for individual estimators
        if hasattr(meta_model, 'estimators_') and hasattr(meta_model, 'n_jobs'):
            print(f"Ensemble model with {len(meta_model.estimators_)} estimators - parallel prediction enabled")
        
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

def load_dem_from_zarr(zarr_file, rainfall_mm):
    """
    Load DEM data from a Zarr file and prepare data for model inference.
    Note: Slope calculation is deferred to window-level processing to save memory.
    
    Args:
        zarr_file: Path to the DEM Zarr file
        rainfall_mm: Rainfall amount in mm
        
    Returns:
        Dictionary with input data and metadata
    """
    try:
        print(f"Loading DEM from Zarr file: {zarr_file}")
        
        # Load Zarr file using xarray
        ds = xr.open_zarr(zarr_file)
        
        # Get the first data variable (should be the DEM data)
        if len(ds.data_vars) > 0:
            # Get the first data variable
            var_name = list(ds.data_vars.keys())[0]
            da = ds[var_name]
            print(f"Loaded data variable: {var_name}")
        else:
            # Fallback: assume it's a single DataArray
            da = ds
        
        # Extract DEM data as numpy array
        print(f"DataArray type: {type(da)}")
        print(f"DataArray shape: {da.shape}")
        print(f"DataArray dims: {da.dims}")
        
        # Convert to numpy array and ensure it's float32
        if hasattr(da.values, 'compute'):
            # Handle dask arrays
            print("Computing dask array...")
            dem_data = da.values.compute().astype(np.float32)
        else:
            # Handle regular numpy arrays
            dem_data = da.values.astype(np.float32)
        
        print(f"DEM dimensions: {dem_data.shape[0]} rows × {dem_data.shape[1]} columns")
        
        # Extract spatial reference information from attributes
        crs_str = da.attrs.get('crs', 'EPSG:4326')
        transform_list = da.attrs.get('transform', None)
        nodata_value = da.attrs.get('nodata', -9999.0)
        
        print(f"CRS: {crs_str}")
        print(f"NoData value: {nodata_value}")
        
        # Create rasterio transform object
        if transform_list and len(transform_list) >= 6:
            transform = Affine(*transform_list[:6])
        else:
            # Calculate transform from coordinates
            x_coords = da.coords['x'].values
            y_coords = da.coords['y'].values
            x_res = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
            y_res = y_coords[0] - y_coords[1] if len(y_coords) > 1 else 1.0  # Negative for north-up
            transform = Affine(x_res, 0, x_coords[0] - x_res/2, 0, -y_res, y_coords[0] + y_res/2)
        
        print(f"Transform: {transform}")
        
        # Create metadata for output GeoTIFF
        geotiff_meta = {
            'driver': 'GTiff',
            'crs': crs_str,
            'transform': transform,
            'width': dem_data.shape[1],
            'height': dem_data.shape[0],
            'count': 1,
            'dtype': 'uint8',
            'nodata': 255  # Use 255 for output (uint8 compatible)
        }
        
        # Create a mask for valid data
        if nodata_value is not None and not np.isnan(nodata_value):
            # Handle both NaN and specific nodata values
            mask = np.logical_and(dem_data != nodata_value, ~np.isnan(dem_data))
        else:
            # Only check for NaN values
            mask = ~np.isnan(dem_data)
        
        # Make a copy of the mask for later use
        original_mask = mask.copy()
        
        # Calculate statistics from valid data
        dem_valid = dem_data[mask]
        if len(dem_valid) > 0:
            dem_mean = np.mean(dem_valid)
            dem_std = np.std(dem_valid)
            print(f"DEM statistics - Mean: {dem_mean:.2f}, Std: {dem_std:.2f}")
        else:
            print("Warning: No valid data found in DEM!")
            dem_mean = 0
            dem_std = 1
        
        print(f"Valid data: {np.sum(mask)}/{dem_data.size} pixels ({np.sum(mask)/dem_data.size*100:.2f}%)")
        
        # Replace invalid values with a sentinel value for processing
        dem_data[~mask] = -9999
        
        # NOTE: Slope calculation is now deferred to window-level processing to save memory
        print("Slope calculation deferred to window-level processing to save memory")
        
        # Create uniform rainfall input
        print(f"Creating rainfall input with {rainfall_mm}mm...")
        rainfall_m = rainfall_mm / 1000.0  # Convert to meters
        
        # Normalize the DEM (using only valid values for statistics)
        dem_norm = dem_data.copy()
        dem_norm[mask] = (dem_data[mask] - dem_mean) / dem_std
        dem_norm[~mask] = 0  # Set to 0 for processing
        
        # Extract county information from attributes if available
        county_name = da.attrs.get('county', 'Unknown')
        state_name = da.attrs.get('state', 'Unknown')
        geoid = da.attrs.get('geoid', 'Unknown')
        
        print(f"County: {county_name}, State: {state_name}, GEOID: {geoid}")
        
        return {
            'dem': dem_norm,
            'dem_raw': dem_data,  # Keep raw DEM for slope calculation
            'rainfall_mm': rainfall_mm,
            'mask': mask,
            'original_mask': original_mask,
            'dem_mean': dem_mean,
            'dem_std': dem_std,
            'geotiff_meta': geotiff_meta,
            'output_nodata': 255,  # Use 255 for output consistency
            'county_info': {
                'county': county_name,
                'state': state_name,
                'geoid': geoid
            }
        }
    
    except Exception as e:
        print(f"Error loading DEM data from Zarr: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def get_model_predictions(data, models, window_size=512, overlap=128, sigma=0.5):
    """
    Get predictions from all base models for the given DEM data,
    using Gaussian kernel smoothing at window boundaries.
    Calculates slope and rainfall at the window level to save memory.
    
    Args:
        data: Dictionary with DEM data and metadata
        models: Dictionary of loaded models
        window_size: Size of the window for processing (default: 512)
        overlap: Overlap between adjacent windows in pixels (default: 128)
        sigma: Standard deviation for the Gaussian kernel (default: 0.5)
        
    Returns:
        Dictionary with predictions
    """
    try:
        # Extract input data
        dem = data['dem']
        dem_raw = data['dem_raw']
        mask = data['mask']
        rainfall_mm = data['rainfall_mm']
        output_nodata = data['output_nodata']
        
        height, width = dem.shape
        print(f"Data dimensions: {height} x {width}")
        
        # Convert rainfall to meters
        rainfall_m = rainfall_mm / 1000.0
        
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
        print("Calculating slope and rainfall at window level to save memory")
        
        # Process each ML model
        for model_name, model in models.items():
            try:
                print(f"  Running {model_name} model with Gaussian smoothing...")
                # Initialize output arrays for accumulating weighted predictions
                probs_out = np.zeros((5, height, width), dtype=np.float32)
                weights_sum = np.zeros((height, width), dtype=np.float32)
                
                # Process image in windows with overlap
                for y_idx in range(n_windows_y):
                    for x_idx in range(n_windows_x):
                        # Calculate window coordinates
                        y_start = min(y_idx * stride, height - window_size) if height > window_size else 0
                        x_start = min(x_idx * stride, width - window_size) if width > window_size else 0
                        y_end = min(y_start + window_size, height)
                        x_end = min(x_start + window_size, width)
                        
                        # Calculate padded window coordinates for slope calculation
                        # Add padding around the window to properly calculate slope
                        pad_size = 2  # Need at least 1 pixel padding for slope calculation
                        y_start_padded = max(0, y_start - pad_size)
                        x_start_padded = max(0, x_start - pad_size)
                        y_end_padded = min(height, y_end + pad_size)
                        x_end_padded = min(width, x_end + pad_size)
                        
                        # Extract padded window data for slope calculation
                        dem_raw_padded = dem_raw[y_start_padded:y_end_padded, x_start_padded:x_end_padded]
                        mask_padded = mask[y_start_padded:y_end_padded, x_start_padded:x_end_padded]
                        
                        # Calculate slope for the padded window
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=RuntimeWarning)
                            slope_padded = calc_slope(dem_raw_padded)
                            # Set slope to 0 for invalid DEM cells
                            slope_padded[~mask_padded] = 0
                        
                        # Extract the actual window area from the padded slope
                        pad_y_offset = y_start - y_start_padded
                        pad_x_offset = x_start - x_start_padded
                        actual_window_h = y_end - y_start
                        actual_window_w = x_end - x_start
                        
                        slope_window = slope_padded[
                            pad_y_offset:pad_y_offset + actual_window_h,
                            pad_x_offset:pad_x_offset + actual_window_w
                        ]
                        
                        # Extract window data (normalized DEM)
                        dem_window = dem[y_start:y_end, x_start:x_end]
                        
                        # Create rainfall window
                        rainfall_window = np.ones_like(dem_window) * rainfall_m
                        # Set rainfall to 0 for invalid DEM cells
                        window_mask = mask[y_start:y_end, x_start:x_end]
                        rainfall_window[~window_mask] = 0
                        
                        # Handle window size if smaller than crop_size
                        window_h, window_w = dem_window.shape
                        if window_h < window_size or window_w < window_size:
                            pad_h = max(0, window_size - window_h)
                            pad_w = max(0, window_size - window_w)
                            dem_window = np.pad(dem_window, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                            slope_window = np.pad(slope_window, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                            rainfall_window = np.pad(rainfall_window, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
                        
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
                        for c in range(pred.shape[0]):
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
                
                # Apply mask - set invalid pixels to output nodata value
                pred_cat[~mask] = output_nodata
                
                # Store results
                predictions[model_name] = pred_cat
                probs[model_name] = probs_out
                
                print(f"  Completed {model_name} model inference")
                
            except Exception as e:
                print(f"  Error running model {model_name}: {e}")
                import traceback
                traceback.print_exc()
                # Initialize with zeros if model fails
                predictions[model_name] = np.zeros_like(dem, dtype=np.int64)
                probs[model_name] = np.zeros((5, height, width), dtype=np.float32)
        
        return {
            'dem': dem,
            'rainfall_mm': rainfall_mm,
            'predictions': predictions,
            'probabilities': probs,
            'mask': mask,
            'output_nodata': output_nodata
        }
    
    except Exception as e:
        print(f"Error getting predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def extract_features_batched(pred_result, batch_size=1000000):
    """
    Extract features for meta-model inference from base model predictions in batches.
    Calculates slope in chunks to avoid memory issues with large DEMs.
    
    Args:
        pred_result: Dictionary with prediction results
        batch_size: Maximum number of pixels to process in each batch
        
    Returns:
        Generator that yields (batch features, batch indices, height, width) for each batch
    """
    # Extract base data and probabilities
    dem = pred_result['dem']
    rainfall_mm = pred_result['rainfall_mm']
    probs = pred_result['probabilities']
    mask = pred_result['mask']
    
    height, width = dem.shape
    print(f"Preparing to extract features from image of size {height}x{width}")
    
    # Convert rainfall to meters for consistency
    rainfall_m = rainfall_mm / 1000.0
    
    # Features: DEM, slope (will be calculated), rainfall, model probabilities
    num_base_features = 3  # DEM, slope, rainfall
    
    # Get available DL models
    dl_models = list(probs.keys())
    num_model_features = len(dl_models) * 5  # 5 class probabilities per model
    num_features = num_base_features + num_model_features
    
    # Flatten mask to get indices of valid pixels
    flat_mask = mask.flatten()
    valid_indices = np.where(flat_mask)[0]
    valid_pixels = len(valid_indices)
    
    print(f"Total valid pixels to process: {valid_pixels:,}")
    
    # Calculate number of batches
    num_batches = int(np.ceil(valid_pixels / batch_size))
    print(f"Processing in {num_batches} batches of up to {batch_size:,} pixels each")
    print("Calculating slope in chunks to avoid memory issues...")
    
    # Calculate chunk size for slope computation (process in 2048x2048 chunks)
    chunk_size = 2048
    n_chunks_y = int(np.ceil(height / chunk_size))
    n_chunks_x = int(np.ceil(width / chunk_size))
    
    # Initialize slope array (only allocate memory for valid pixels would be ideal, but for simplicity, use full array)
    # For very large arrays, we'll calculate slope on-demand for each chunk
    print(f"Will calculate slope in {n_chunks_y}x{n_chunks_x} chunks of up to {chunk_size}x{chunk_size} pixels")
    
    # Pre-calculate slope for smaller images, or use chunk-based approach for larger ones
    total_pixels = height * width
    if total_pixels <= 50000000:  # ~7000x7000 pixels or smaller
        print("Image small enough - calculating slope for entire array...")
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            slope = calc_slope(dem)
            slope[~mask] = 0
        use_precalculated_slope = True
    else:
        print("Large image - will calculate slope on demand for each chunk...")
        slope = None
        use_precalculated_slope = False
    
    # Create rainfall array
    rainfall = np.ones_like(dem) * rainfall_m
    rainfall[~mask] = 0
    
    # Process in batches
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, valid_pixels)
        batch_size_actual = batch_end - batch_start
        
        print(f"Processing batch {batch_idx+1}/{num_batches}, {batch_size_actual:,} pixels")
        
        # Get indices for this batch
        batch_indices = valid_indices[batch_start:batch_end]
        
        # Create feature array for this batch
        X_batch = np.zeros((batch_size_actual, num_features), dtype=np.float32)
        
        # Add DEM features (first column)
        X_batch[:, 0] = dem.flatten()[batch_indices]
        
        # Add slope features (second column) - calculate on demand if needed
        if use_precalculated_slope:
            X_batch[:, 1] = slope.flatten()[batch_indices]
        else:
            # Calculate slope on demand for pixels in this batch
            # Group batch indices by chunks to minimize slope calculations
            y_indices = batch_indices // width
            x_indices = batch_indices % width
            
            # Find unique chunks that contain pixels from this batch
            chunk_y_indices = y_indices // chunk_size
            chunk_x_indices = x_indices // chunk_size
            unique_chunks = set(zip(chunk_y_indices, chunk_x_indices))
            
            # Calculate slope for each needed chunk and extract values
            slope_values = np.zeros(batch_size_actual, dtype=np.float32)
            
            for chunk_y, chunk_x in unique_chunks:
                # Calculate chunk boundaries
                y_start = chunk_y * chunk_size
                y_end = min((chunk_y + 1) * chunk_size, height)
                x_start = chunk_x * chunk_size
                x_end = min((chunk_x + 1) * chunk_size, width)
                
                # Extract DEM chunk with padding for slope calculation
                pad_size = 2
                y_start_padded = max(0, y_start - pad_size)
                x_start_padded = max(0, x_start - pad_size)
                y_end_padded = min(height, y_end + pad_size)
                x_end_padded = min(width, x_end + pad_size)
                
                dem_chunk_padded = dem[y_start_padded:y_end_padded, x_start_padded:x_end_padded]
                mask_chunk_padded = mask[y_start_padded:y_end_padded, x_start_padded:x_end_padded]
                
                # Calculate slope for this chunk
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    slope_chunk_padded = calc_slope(dem_chunk_padded)
                    slope_chunk_padded[~mask_chunk_padded] = 0
                
                # Extract the actual chunk area from the padded slope
                pad_y_offset = y_start - y_start_padded
                pad_x_offset = x_start - x_start_padded
                actual_chunk_h = y_end - y_start
                actual_chunk_w = x_end - x_start
                
                slope_chunk = slope_chunk_padded[
                    pad_y_offset:pad_y_offset + actual_chunk_h,
                    pad_x_offset:pad_x_offset + actual_chunk_w
                ]
                
                # Find pixels in this batch that belong to this chunk
                in_chunk = ((y_indices >= y_start) & (y_indices < y_end) & 
                           (x_indices >= x_start) & (x_indices < x_end))
                
                if np.any(in_chunk):
                    # Extract slope values for pixels in this batch
                    batch_y_in_chunk = y_indices[in_chunk] - y_start
                    batch_x_in_chunk = x_indices[in_chunk] - x_start
                    slope_values[in_chunk] = slope_chunk[batch_y_in_chunk, batch_x_in_chunk]
            
            X_batch[:, 1] = slope_values
        
        # Add rainfall features (third column)
        X_batch[:, 2] = rainfall.flatten()[batch_indices]
        
        # Add model probabilities (remaining columns)
        feature_idx = num_base_features
        for model_name in dl_models:
            model_probs = probs[model_name]
            for class_idx in range(5):
                X_batch[:, feature_idx] = model_probs[class_idx].flatten()[batch_indices]
                feature_idx += 1
        
        yield X_batch, batch_indices, height, width

def run_meta_model_inference(data, meta_model, base_models, output_file, 
                           window_size=512, overlap=128, sigma=0.5, batch_size=1000000):
    """
    Run inference using the meta-model on the provided DEM data.
    Processes large DEMs in batches to avoid memory issues.
    
    Args:
        data: Dictionary with DEM data and metadata
        meta_model: Loaded meta-model
        base_models: Dictionary of loaded models
        output_file: Path to save results
        window_size: Size of the window for processing (default: 512)
        overlap: Overlap between adjacent windows (default: 128)
        sigma: Sigma for Gaussian kernel (default: 0.5)
        batch_size: Maximum number of pixels to process in each batch (default: 1,000,000)
        
    Returns:
        Dictionary with meta-model evaluation results
    """
    print("\nRunning meta-model inference...")
    
    # Get base model predictions with Gaussian kernel smoothing
    pred_result = get_model_predictions(data, base_models, 
                                      window_size=window_size, overlap=overlap, sigma=sigma)
    
    if pred_result is None:
        print("No valid predictions. Exiting.")
        return None
    
    # Get nodata value from metadata
    output_nodata = data['output_nodata']
    
    # Get dimensions
    height, width = data['dem'].shape
    total_pixels = height * width
    
    # Initialize the full prediction array with nodata value
    print(f"Initializing prediction array of size {height}x{width}")
    meta_pred = np.ones((height, width), dtype=np.uint8) * output_nodata
    
    # Process features and make predictions in batches
    print("Extracting features and making predictions in batches...")
    start_time = time.time()
    
    try:
        # Get batched features and make predictions for each batch
        for batch_idx, (X_batch, batch_indices, h, w) in enumerate(
            extract_features_batched(pred_result, batch_size=batch_size)
        ):
            # Make predictions for this batch
            batch_start_time = time.time()
            meta_pred_batch = meta_model.predict(X_batch)
            batch_time = time.time() - batch_start_time
            
            # Place batch predictions into the full prediction array
            y_indices = batch_indices // width
            x_indices = batch_indices % width
            
            # Assign predictions
            for i in range(len(batch_indices)):
                meta_pred[y_indices[i], x_indices[i]] = meta_pred_batch[i]
            
            # Report progress
            print(f"Batch {batch_idx+1} processed in {batch_time:.2f} seconds")
            
            # Clear some memory
            del X_batch, meta_pred_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Apply the original mask (to handle any inconsistencies)
        meta_pred[~data['original_mask']] = output_nodata
        
        # Report total time
        inference_time = time.time() - start_time
        print(f"Meta-model prediction completed in {inference_time:.2f} seconds")
        
        # Add meta prediction to results
        pred_result['meta_prediction'] = meta_pred
        
        # Create visualization
        create_visualization(data, pred_result, output_file)
        
        # Save predictions as Zarr files
        print("\nSaving predictions as Zarr files...")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Ensure output file has .zarr extension
        if not output_file.endswith('.zarr'):
            output_file = output_file.replace('.tif', '.zarr').replace('.geotiff', '.zarr')
            if not output_file.endswith('.zarr'):
                output_file += '.zarr'
        
        # Save meta-model prediction
        meta_zarr_path = output_file
        save_zarr(meta_pred, meta_zarr_path, data['geotiff_meta'], original_mask=data['original_mask'])
        
        # Save top 3 base model predictions
        base_models_to_save = min(3, len(pred_result['predictions']))
        if pred_result['predictions']:
            model_names = list(pred_result['predictions'].keys())
            
            for i, model_name in enumerate(model_names[:base_models_to_save]):
                base_output_file = output_file.replace('.zarr', f'_{model_name}.zarr')
                save_zarr(pred_result['predictions'][model_name], 
                         base_output_file, data['geotiff_meta'], original_mask=data['original_mask'])
        
        return pred_result
    
    except Exception as e:
        print(f"Error in meta-model inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_visualization(data, pred_result, output_file):
    """
    Create visualization of meta-model prediction with base models.
    
    Args:
        data: Dictionary with DEM data and metadata
        pred_result: Dictionary with prediction results
        output_file: Path to save visualization
    """
    print(f"Creating visualization...")
    
    try:
        # Create colormap for flood categories
        flood_colors = [FLOOD_COLORS[FloodCategory(i)] for i in range(len(FloodCategory))]
        cmap = ListedColormap(flood_colors)
        
        # Get data
        dem = data['dem']
        mask = data['mask']
        rainfall_mm = data['rainfall_mm']  # Get rainfall from original data
        meta_pred = pred_result['meta_prediction']
        base_preds = pred_result['predictions']
        output_nodata = data['output_nodata']
        county_info = data['county_info']
        
        # Check if the visualization would be too large
        height, width = dem.shape
        total_pixels = height * width
        max_viz_pixels = 2000000  # 2 million pixels max for visualization
        
        # If the image is too large, downsample it
        if total_pixels > max_viz_pixels:
            downsample_ratio = int(np.ceil(np.sqrt(total_pixels / max_viz_pixels)))
            print(f"Image too large for visualization ({height}x{width}). Downsampling by factor of {downsample_ratio}.")
            
            # Downsample data
            dem_ds = dem[::downsample_ratio, ::downsample_ratio]
            mask_ds = mask[::downsample_ratio, ::downsample_ratio]
            meta_pred_ds = meta_pred[::downsample_ratio, ::downsample_ratio]
            
            # Downsample base predictions
            downsampled_base_preds = {}
            for model_name, pred in base_preds.items():
                downsampled_base_preds[model_name] = pred[::downsample_ratio, ::downsample_ratio]
            base_preds = downsampled_base_preds
        else:
            dem_ds = dem
            mask_ds = mask
            meta_pred_ds = meta_pred
        
        # Calculate how many subplots we need
        n_models = len(base_preds)
        total_plots = n_models + 2  # +2 for DEM and meta-model
        
        # Limit the number of subplots to avoid memory issues
        max_plots = 9
        if total_plots > max_plots:
            print(f"Too many models to visualize ({n_models}). Limiting to {max_plots-2} models.")
            base_preds = {k: base_preds[k] for k in list(base_preds.keys())[:max_plots-2]}
            n_models = len(base_preds)
            total_plots = n_models + 2
        
        # Create a grid of subplots
        ncols = min(3, total_plots)
        nrows = (total_plots + ncols - 1) // ncols
        
        # Create figure 
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), dpi=100)
        axes = axes.flatten() if total_plots > 1 else [axes]
        
        # Hide any unused subplots
        for i in range(total_plots, len(axes)):
            axes[i].axis('off')
        
        # Plot DEM
        dem_plot = dem_ds.copy()
        dem_plot[~mask_ds] = np.nan
        axes[0].set_title("DEM")
        im = axes[0].imshow(dem_plot, cmap='terrain')
        plt.colorbar(im, ax=axes[0], label='Elevation (normalized)')
        
        # Plot meta-model prediction
        meta_pred_masked = np.ma.masked_where(meta_pred_ds == output_nodata, meta_pred_ds)
        axes[1].set_title("Meta-Model Prediction")
        axes[1].imshow(meta_pred_masked, cmap=cmap, vmin=0, vmax=4)
        
        # Plot base model predictions
        for i, (model_name, pred) in enumerate(base_preds.items()):
            idx = i + 2
            pred_masked = np.ma.masked_where(pred == output_nodata, pred)
            axes[idx].set_title(f"{model_name} Prediction")
            axes[idx].imshow(pred_masked, cmap=cmap, vmin=0, vmax=4)
        
        # Add legend for flood categories
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=FLOOD_COLORS[FloodCategory(i)], 
                  label=FloodCategory(i).name.replace('_', ' ')) 
            for i in range(len(FloodCategory))
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0))
        
        # Add overall title with county and rainfall info
        title = f"Flood Prediction Results - {county_info['county']} County, {county_info['state']} ({rainfall_mm}mm rainfall)"
        if county_info['geoid'] != 'Unknown':
            title += f"\nGEOID: {county_info['geoid']}"
        fig.suptitle(title, fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        
        # Save figure
        viz_path = output_file.replace('.zarr', '_visualization.png').replace('.tif', '_visualization.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close('all')
        
        print(f"Saved visualization to {viz_path}")
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def save_zarr(data, output_path, meta=None, nodata=None, original_mask=None):
    """
    Save a prediction array as a Zarr file with spatial reference information.
    
    Args:
        data: 2D numpy array with prediction data
        output_path: Path to save the Zarr file (should end with .zarr)
        meta: Dictionary with metadata (crs, transform, etc.)
        nodata: Value to use for nodata pixels (overrides meta['nodata'] if provided)
        original_mask: Mask of valid data from original DEM
    """
    try:
        print(f"Saving Zarr to {output_path}")
        
        if meta is None:
            # Create default metadata if none provided
            height, width = data.shape
            meta = {
                'crs': 'EPSG:4326',
                'transform': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                'width': width,
                'height': height,
                'nodata': 255
            }
            
            if nodata is None:
                nodata = 255
            meta['nodata'] = nodata
        else:
            # Ensure we have all required metadata
            meta = meta.copy()
            
            if nodata is not None:
                meta['nodata'] = nodata
        
        # Check if the nodata value is compatible with uint8
        if 'nodata' in meta and (meta['nodata'] < 0 or meta['nodata'] > 255):
            print(f"Warning: Original nodata value {meta['nodata']} is not compatible with uint8.")
            print("Using 255 as the nodata value for the output instead.")
            meta['nodata'] = 255
        
        # Get the nodata value
        nodata_value = meta['nodata']
        
        # Convert to uint8
        data_clipped = np.clip(data, 0, 5)
        data_uint8 = data_clipped.astype(np.uint8)
        
        # Apply the original mask if provided
        if original_mask is not None:
            data_uint8[~original_mask] = nodata_value
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create coordinates based on transform
        height, width = data_uint8.shape
        if 'transform' in meta and len(meta['transform']) >= 6:
            transform = meta['transform']
            # Extract transform parameters
            x_res = transform[0]
            y_res = -transform[4]  # Usually negative for north-up
            x_origin = transform[2]
            y_origin = transform[5]
            
            # Create coordinate arrays
            x_coords = np.arange(width) * x_res + x_origin + x_res/2
            y_coords = np.arange(height) * y_res + y_origin + y_res/2
        else:
            # Default coordinates
            x_coords = np.arange(width)
            y_coords = np.arange(height)
        
        # Create xarray DataArray
        da = xr.DataArray(
            data_uint8,
            coords={
                'y': ('y', y_coords),
                'x': ('x', x_coords)
            },
            dims=['y', 'x'],
            name='flood_prediction',
            attrs={
                'crs': meta.get('crs', 'EPSG:4326'),
                'transform': meta.get('transform', [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
                'nodata': nodata_value,
                'description': 'Flood prediction categories (0=No_Flood, 1=Minor_Flood, 2=Moderate_Flood, 3=Major_Flood, 4=Extreme_Flood)',
                'units': 'category',
                'long_name': 'Flood Risk Categories',
                'created': datetime.now().isoformat(),
                'source': 'Meta-learning flood prediction model'
            }
        )
        
        # Set spatial reference with rioxarray
        da = da.rio.write_crs(meta.get('crs', 'EPSG:4326'))
        if 'transform' in meta:
            da = da.rio.write_transform(meta['transform'])
        da = da.rio.write_nodata(nodata_value)
        
        # Save to Zarr
        da.to_zarr(output_path, mode='w')
            
        print(f"Successfully saved Zarr to {output_path}")
        
    except Exception as e:
        print(f"Error saving Zarr: {str(e)}")
        import traceback
        traceback.print_exc()

def save_geotiff(data, output_path, meta=None, nodata=None, original_mask=None):
    """
    Save a prediction array as a GeoTIFF file.
    
    Args:
        data: 2D numpy array with prediction data
        output_path: Path to save the GeoTIFF file
        meta: Dictionary with metadata for the GeoTIFF (crs, transform, etc.)
        nodata: Value to use for nodata pixels (overrides meta['nodata'] if provided)
        original_mask: Mask of valid data from original DEM
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
                'transform': Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            }
            
            if nodata is None:
                nodata = 255
            meta['nodata'] = nodata
        else:
            # Ensure we have all required metadata
            meta = meta.copy()
            meta.update({
                'driver': 'GTiff',
                'count': 1,
                'dtype': 'uint8'
            })
            
            if nodata is not None:
                meta['nodata'] = nodata
        
        # Check if the nodata value is compatible with uint8
        if 'nodata' in meta and (meta['nodata'] < 0 or meta['nodata'] > 255):
            print(f"Warning: Original nodata value {meta['nodata']} is not compatible with uint8.")
            print("Using 255 as the nodata value for the output instead.")
            meta['nodata'] = 255
        
        # Get the nodata value
        nodata_value = meta['nodata']
        
        # Convert to uint8
        data_clipped = np.clip(data, 0, 5)
        data_uint8 = data_clipped.astype(np.uint8)
        
        # Apply the original mask if provided
        if original_mask is not None:
            data_uint8[~original_mask] = nodata_value
        
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

def main(args):
    """Main function to run meta-model inference on a DEM Zarr file."""
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    print(f"Input DEM Zarr: {args.zarr_file}")
    print(f"Rainfall: {args.rainfall_mm}mm")
    print(f"Output file: {args.output_file}")
    print(f"Window size: {args.window_size}, Overlap: {args.overlap}, Sigma: {args.sigma}")
    print(f"Batch size: {args.batch_size:,} pixels")
    print(f"Parallel jobs for meta-model: {args.n_jobs}")
    
    # Load meta-model with parallel prediction
    meta_model = load_meta_model(args.meta_model, n_jobs=args.n_jobs)
    if meta_model is None:
        print("Failed to load meta-model. Exiting.")
        return
    
    # Load base models
    base_models = load_base_models()
    if len(base_models) == 0:
        print("Failed to load any base models. Exiting.")
        return
    
    # Load DEM from Zarr and prepare data
    data = load_dem_from_zarr(args.zarr_file, args.rainfall_mm)
    if data is None:
        print("Failed to load DEM data from Zarr. Exiting.")
        return
    
    # Run meta-model inference
    result = run_meta_model_inference(
        data, meta_model, base_models, args.output_file,
        window_size=args.window_size, 
        overlap=args.overlap,
        sigma=args.sigma,
        batch_size=args.batch_size
    )
    
    if result:
        print(f"\nInference completed successfully. Results saved to {args.output_file}")
    else:
        print("\nInference failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run meta-model inference on a DEM Zarr file')
    parser.add_argument('--zarr_file', type=str, required=True,
                        help='Path to the DEM Zarr file')
    parser.add_argument('--rainfall_mm', type=float, required=True,
                        help='Rainfall amount in millimeters')
    parser.add_argument('--meta_model', type=str, 
                        default='/home/users/li1995/global_flood/FloodRisk-DL/terratorch/meta_learning_results_20250515_143320/meta_model_random_forest.joblib',
                        help='Path to the trained meta-model')
    parser.add_argument('--output_file', type=str, default='./dem_inference_results.zarr',
                        help='Output file path for the Zarr result')
    parser.add_argument('--window_size', type=int, default=512,
                        help='Size of window for processing (default: 512)')
    parser.add_argument('--overlap', type=int, default=128,
                        help='Overlap between adjacent windows in pixels (default: 128)')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Standard deviation for Gaussian kernel (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=5000000,
                        help='Maximum number of pixels to process in each batch (default: 5,000,000)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs for meta-model prediction (-1 for all cores, default: -1)')
    
    args = parser.parse_args()
    main(args)
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
import zarr
from numba import njit, prange
import multiprocessing as mp
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import tempfile
import shutil
import threading

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
    Note: Uses lazy loading to avoid loading entire DEM into memory.
    
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
        
        print(f"DataArray type: {type(da)}")
        print(f"DataArray shape: {da.shape}")
        print(f"DataArray dims: {da.dims}")
        
        # Keep as dask array - do NOT compute the entire array
        print(f"DEM dimensions: {da.shape[0]} rows × {da.shape[1]} columns")
        height, width = da.shape
        
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
        
        # Create metadata for output files
        output_meta = {
            'driver': 'GTiff',
            'crs': crs_str,
            'transform': transform,
            'width': width,
            'height': height,
            'count': 1,
            'dtype': 'uint8',
            'nodata': 255  # Use 255 for output (uint8 compatible)
        }
        
        # Calculate statistics from a sample of the data to avoid loading everything
        print("Calculating DEM statistics from sample data...")
        # Sample every 100th pixel in both dimensions to get statistics
        sample_step = 100
        sample_data = da[::sample_step, ::sample_step].values
        
        # Create a mask for valid data in the sample
        if nodata_value is not None and not np.isnan(nodata_value):
            sample_mask = np.logical_and(sample_data != nodata_value, ~np.isnan(sample_data))
        else:
            sample_mask = ~np.isnan(sample_data)
        
        # Calculate statistics from valid sample data
        sample_valid = sample_data[sample_mask]
        if len(sample_valid) > 0:
            dem_mean = np.mean(sample_valid)
            dem_std = np.std(sample_valid)
            print(f"DEM statistics (from sample) - Mean: {dem_mean:.2f}, Std: {dem_std:.2f}")
        else:
            print("Warning: No valid data found in sample!")
            dem_mean = 0
            dem_std = 1
        
        # Calculate approximate valid data percentage
        valid_ratio = np.sum(sample_mask) / sample_mask.size
        print(f"Estimated valid data: {valid_ratio*100:.2f}% (from sample)")
        
        # Extract county information from attributes if available
        county_name = da.attrs.get('county', 'Unknown')
        state_name = da.attrs.get('state', 'Unknown')
        geoid = da.attrs.get('geoid', 'Unknown')
        
        print(f"County: {county_name}, State: {state_name}, GEOID: {geoid}")
        
        return {
            'zarr_file': zarr_file,  # Add the original zarr file path
            'dem_dask': da,  # Keep as dask array for lazy loading
            'rainfall_mm': rainfall_mm,
            'dem_mean': dem_mean,
            'dem_std': dem_std,
            'nodata_value': nodata_value,
            'height': height,
            'width': width,
            'geotiff_meta': output_meta,
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

def get_model_predictions(data, models, output_file, window_size=512, overlap=128, sigma=0.5, n_workers=None):
    """
    Get predictions from all base models for the given DEM data,
    using Gaussian kernel smoothing at window boundaries with parallel processing.
    Processes models sequentially to save memory and uses zarr for intermediate storage.
    
    Args:
        data: Dictionary with DEM data and metadata
        models: Dictionary of loaded models
        output_file: Base output file path for saving intermediate results
        window_size: Size of the window for processing (default: 512)
        overlap: Overlap between adjacent windows in pixels (default: 128)
        sigma: Standard deviation for the Gaussian kernel (default: 0.5)
        n_workers: Number of parallel workers (default: min(16, cpu_count()))
        
    Returns:
        Dictionary with predictions
    """
    try:
        # Extract input data
        dem_dask = data['dem_dask']
        height = data['height']
        width = data['width']
        rainfall_mm = data['rainfall_mm']
        dem_mean = data['dem_mean']
        dem_std = data['dem_std']
        nodata_value = data['nodata_value']
        output_nodata = data['output_nodata']
        
        print(f"Data dimensions: {height} x {width}")
        
        # Convert rainfall to meters
        rainfall_m = rainfall_mm / 1000.0
        
        # Get Gaussian weight map for window blending
        weight_map = create_gaussian_weight_map(window_size, sigma)
        
        # Calculate stride based on overlap
        stride = window_size - overlap
        
        # Calculate number of windows needed
        n_windows_y = max(1, int(np.ceil((height - overlap) / stride))) if height > window_size else 1
        n_windows_x = max(1, int(np.ceil((width - overlap) / stride))) if width > window_size else 1
        
        print(f"Using {n_windows_y}x{n_windows_x} windows with size {window_size}, stride={stride}, overlap={overlap}")
        
        # Set up parallel processing
        if n_workers is None:
            n_workers = min(8, os.cpu_count())  # Use fewer threads to avoid GPU conflicts
            
        # Critical memory optimization for very large datasets
        total_pixels = height * width
        if total_pixels > 50_000_000:  # > 50M pixels (like Harris County with 7.6B pixels)
            if n_workers is None or n_workers > 2:
                n_workers = min(2, os.cpu_count())
                print(f"Large dataset detected ({total_pixels:,} pixels). Reducing to {n_workers} workers for memory conservation.")
        
        print(f"Using {n_workers} parallel threads")
        print("Processing models sequentially to save memory...")
        
        # Create temporary directory for intermediate zarr files
        temp_dir = tempfile.mkdtemp(prefix='flood_inference_')
        print(f"Using temporary directory: {temp_dir}")
        
        # Prepare window coordinates for all windows
        all_window_coords = []
        for y_idx in range(n_windows_y):
            for x_idx in range(n_windows_x):
                y_start = min(y_idx * stride, height - window_size) if height > window_size else 0
                x_start = min(x_idx * stride, width - window_size) if width > window_size else 0
                y_end = min(y_start + window_size, height)
                x_end = min(x_start + window_size, width)
                all_window_coords.append((y_idx, x_idx, y_start, x_start, y_end, x_end))
        
        total_windows = len(all_window_coords)
        print(f"Total windows to process: {total_windows}")
        
        # Create GPU lock for thread synchronization
        gpu_lock = threading.Lock()
        
        # Initialize dictionaries for final results
        predictions = {}
        probs = {}
        
        # Process each ML model sequentially
        for model_idx, (model_name, model) in enumerate(models.items()):
            try:
                print(f"  Running {model_name} model ({model_idx+1}/{len(models)}) with Gaussian smoothing...")
                
                # Create zarr arrays for this model's outputs
                probs_zarr_path = os.path.join(temp_dir, f'probs_{model_name}.zarr')
                weights_zarr_path = os.path.join(temp_dir, f'weights_{model_name}.zarr')
                
                # Initialize zarr arrays for accumulating weighted predictions
                probs_zarr = zarr.open(probs_zarr_path, mode='w', 
                                     shape=(5, height, width), 
                                     chunks=(1, min(2048, height), min(2048, width)),
                                     dtype=np.float32, fill_value=0.0)
                
                weights_zarr = zarr.open(weights_zarr_path, mode='w',
                                       shape=(height, width),
                                       chunks=(min(2048, height), min(2048, width)),
                                       dtype=np.float32, fill_value=0.0)
                
                # Process windows in parallel batches using threading
                batch_size = max(1, total_windows // (n_workers * 2))  # Smaller batches for threading
                
                # Use much smaller batches for very large datasets to prevent OOM
                if total_pixels > 50_000_000:  # Large datasets like Harris County
                    batch_size = max(1, total_windows // (n_workers * 8))  # Much smaller batches
                    print(f"    Large dataset: using smaller batch size of {batch_size} windows per batch")
                
                window_batches = [all_window_coords[i:i + batch_size] 
                                for i in range(0, len(all_window_coords), batch_size)]
                
                print(f"    Processing {len(window_batches)} batches of ~{batch_size} windows each using threads")
                
                # Process batches in parallel using threads
                with ThreadPoolExecutor(max_workers=n_workers) as executor:
                    # Submit all batches
                    future_to_batch = {}
                    for batch_idx, batch in enumerate(window_batches):
                        future = executor.submit(
                            process_window_batch_threaded,
                            batch,
                            model,  # Share the model instance
                            dem_dask,  # Share the DEM data
                            rainfall_m,
                            dem_mean,
                            dem_std,
                            nodata_value,
                            window_size,
                            weight_map,
                            gpu_lock
                        )
                        future_to_batch[future] = batch_idx
                    
                    # Collect results as they complete
                    completed_batches = 0
                    for future in as_completed(future_to_batch):
                        batch_idx = future_to_batch[future]
                        try:
                            batch_results = future.result(timeout=300)  # 5 minute timeout
                            completed_batches += 1
                            
                            if completed_batches % max(1, len(window_batches) // 10) == 0:
                                print(f"    Completed {completed_batches}/{len(window_batches)} batches "
                                      f"({completed_batches/len(window_batches)*100:.1f}%)")
                            
                            # Accumulate results into zarr arrays
                            for window_coords, weighted_probs, current_weight in batch_results:
                                _, _, y_start, x_start, y_end, x_end = window_coords
                                
                                # Optimized zarr updates - batch the operations
                                window_slice_y = slice(y_start, y_end)
                                window_slice_x = slice(x_start, x_end)
                                
                                # Read current values once for all classes
                                current_probs = probs_zarr[:, window_slice_y, window_slice_x]
                                current_weights = weights_zarr[window_slice_y, window_slice_x]
                                
                                # Add weighted probabilities
                                current_probs += weighted_probs
                                
                                # Update zarr arrays in batch
                                probs_zarr[:, window_slice_y, window_slice_x] = current_probs
                                weights_zarr[window_slice_y, window_slice_x] = current_weights + current_weight
                                
                                # Periodic memory cleanup for large datasets
                                if total_pixels > 50_000_000 and completed_batches % 5 == 0:
                                    import gc
                                    gc.collect()
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                
                        except Exception as e:
                            print(f"    Error processing batch {batch_idx}: {e}")
                            # Don't print full traceback for timeout errors
                            if "timeout" not in str(e).lower():
                                import traceback
                                traceback.print_exc()
                
                print(f"    Completed all {total_windows} windows for {model_name}")
                
                # Clear GPU cache after each model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"    Normalizing predictions for {model_name}...")
                
                # Normalize by weights to get final probabilities, process in chunks to save memory
                normalize_chunk_size = 2048
                n_norm_chunks_y = int(np.ceil(height / normalize_chunk_size))
                n_norm_chunks_x = int(np.ceil(width / normalize_chunk_size))
                
                for norm_y in range(n_norm_chunks_y):
                    for norm_x in range(n_norm_chunks_x):
                        y_norm_start = norm_y * normalize_chunk_size
                        y_norm_end = min((norm_y + 1) * normalize_chunk_size, height)
                        x_norm_start = norm_x * normalize_chunk_size
                        x_norm_end = min((norm_x + 1) * normalize_chunk_size, width)
                        
                        # Read chunk data
                        chunk_slice_y = slice(y_norm_start, y_norm_end)
                        chunk_slice_x = slice(x_norm_start, x_norm_end)
                        
                        probs_chunk = probs_zarr[:, chunk_slice_y, chunk_slice_x]
                        weights_chunk = weights_zarr[chunk_slice_y, chunk_slice_x]
                        
                        # Normalize (avoid division by zero)
                        weights_chunk[weights_chunk == 0] = 1e-8
                        probs_chunk = probs_chunk / weights_chunk[np.newaxis, :, :]
                        
                        # Write back normalized probabilities
                        probs_zarr[:, chunk_slice_y, chunk_slice_x] = probs_chunk
                
                # Create output arrays for this model - only keep predictions, save probabilities to disk
                pred_cat = np.zeros((height, width), dtype=np.uint8)
                
                # Get class with highest probability for each pixel - process in chunks
                print(f"    Computing final predictions for {model_name}...")
                
                classify_chunk_size = 4096  # Larger chunks for classification
                n_classify_chunks_y = int(np.ceil(height / classify_chunk_size))
                n_classify_chunks_x = int(np.ceil(width / classify_chunk_size))
                
                for class_y in range(n_classify_chunks_y):
                    for class_x in range(n_classify_chunks_x):
                        y_class_start = class_y * classify_chunk_size
                        y_class_end = min((class_y + 1) * classify_chunk_size, height)
                        x_class_start = class_x * classify_chunk_size
                        x_class_end = min((class_x + 1) * classify_chunk_size, width)
                        
                        # Read probability chunk and compute argmax
                        probs_chunk = probs_zarr[:, y_class_start:y_class_end, x_class_start:x_class_end]
                        pred_chunk = np.argmax(probs_chunk, axis=0).astype(np.uint8)
                        
                        # Store predictions
                        pred_cat[y_class_start:y_class_end, x_class_start:x_class_end] = pred_chunk
                
                # Store final results - only predictions, probabilities saved to temp zarr
                predictions[model_name] = pred_cat
                probs[model_name] = probs_zarr_path  # Store path instead of array
                
                # Clean up intermediate zarr files for this model
                shutil.rmtree(probs_zarr_path, ignore_errors=True)
                shutil.rmtree(weights_zarr_path, ignore_errors=True)
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                print(f"  Completed {model_name} model inference")
                
                # Optionally unload model to free memory (if processing very large data)
                # Note: This would require reloading models, which we skip for now
                # but could be added if memory is still an issue
                
            except Exception as e:
                print(f"  Error running model {model_name}: {e}")
                import traceback
                traceback.print_exc()
                # Initialize with nodata if model fails
                predictions[model_name] = np.ones((height, width), dtype=np.uint8) * output_nodata
                probs[model_name] = None
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            'rainfall_mm': rainfall_mm,
            'predictions': predictions,
            'probabilities': probs,
            'height': height,
            'width': width,
            'output_nodata': output_nodata,
            'dem_dask': dem_dask,
            'dem_mean': dem_mean,
            'dem_std': dem_std,
            'nodata_value': nodata_value
        }
    
    except Exception as e:
        print(f"Error getting predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def extract_features_batched(pred_result, batch_size=1000000):
    """
    Extract features for meta-model inference from base model predictions in batches.
    Works with lazy-loaded dask arrays and zarr-stored probabilities to avoid memory issues.
    
    Args:
        pred_result: Dictionary with prediction results
        batch_size: Maximum number of pixels to process in each batch
        
    Returns:
        Generator that yields (batch features, batch indices, height, width) for each batch
    """
    # Extract base data and probabilities
    dem_dask = pred_result['dem_dask']
    rainfall_mm = pred_result['rainfall_mm']
    probs_paths = pred_result['probabilities']  # Now contains file paths
    height = pred_result['height']
    width = pred_result['width']
    dem_mean = pred_result['dem_mean']
    dem_std = pred_result['dem_std']
    nodata_value = pred_result['nodata_value']
    
    print(f"Preparing to extract features from image of size {height}x{width}")
    
    # Convert rainfall to meters for consistency
    rainfall_m = rainfall_mm / 1000.0
    
    # Features: DEM, slope (will be calculated), rainfall, model probabilities
    num_base_features = 3  # DEM, slope, rainfall
    
    # Get available DL models - filter out failed models
    dl_models = [name for name, path in probs_paths.items() if path is not None]
    num_model_features = len(dl_models) * 5  # 5 class probabilities per model
    num_features = num_base_features + num_model_features
    
    print(f"Using {len(dl_models)} models for meta-learning: {dl_models}")
    
    # Calculate mask for valid pixels by processing in larger chunks for efficiency
    print("Identifying valid pixels in chunks...")
    chunk_size = 4096  # Increased chunk size for better efficiency
    n_chunks_y = int(np.ceil(height / chunk_size))
    n_chunks_x = int(np.ceil(width / chunk_size))
    
    valid_indices_list = []
    total_valid_pixels = 0
    
    for chunk_y in range(n_chunks_y):
        for chunk_x in range(n_chunks_x):
            y_start = chunk_y * chunk_size
            y_end = min((chunk_y + 1) * chunk_size, height)
            x_start = chunk_x * chunk_size
            x_end = min((chunk_x + 1) * chunk_size, width)
            
            # Load DEM chunk and create mask
            dem_chunk = dem_dask[y_start:y_end, x_start:x_end].values.astype(np.float32)
            
            # Create mask for this chunk
            if nodata_value is not None and not np.isnan(nodata_value):
                mask_chunk = np.logical_and(dem_chunk != nodata_value, ~np.isnan(dem_chunk))
            else:
                mask_chunk = ~np.isnan(dem_chunk)
            
            # Find valid pixels in this chunk
            y_chunk_indices, x_chunk_indices = np.where(mask_chunk)
            
            if len(y_chunk_indices) > 0:
                # Convert to global coordinates
                global_y_indices = y_chunk_indices + y_start
                global_x_indices = x_chunk_indices + x_start
                
                # Store indices as a list of tuples
                chunk_indices = list(zip(global_y_indices, global_x_indices))
                valid_indices_list.extend(chunk_indices)
                total_valid_pixels += len(chunk_indices)
    
    print(f"Found {total_valid_pixels:,} valid pixels")
    
    # Convert to arrays for easier processing
    if total_valid_pixels == 0:
        print("No valid pixels found!")
        return
    
    valid_indices = np.array(valid_indices_list)
    y_indices = valid_indices[:, 0].astype(np.int32)
    x_indices = valid_indices[:, 1].astype(np.int32)
    
    # Calculate number of batches
    num_batches = int(np.ceil(total_valid_pixels / batch_size))
    print(f"Processing {total_valid_pixels:,} valid pixels in {num_batches} batches")
    
    # Process each batch
    for batch_idx in range(num_batches):
        print(f"Processing batch {batch_idx + 1}/{num_batches}...")
        
        # Get batch indices
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_valid_pixels)
        
        batch_y_indices = y_indices[start_idx:end_idx]
        batch_x_indices = x_indices[start_idx:end_idx]
        batch_size_actual = end_idx - start_idx
        
        # Initialize feature array for this batch
        X_batch = np.zeros((batch_size_actual, num_features), dtype=np.float32)
        
        # Extract base features (DEM, slope, rainfall) using chunked processing
        print(f"  Extracting base features for {batch_size_actual:,} pixels...")
        
        # Use larger chunks for feature extraction to reduce overhead
        feature_chunk_size = 2048
        n_feature_chunks_y = int(np.ceil(height / feature_chunk_size))
        n_feature_chunks_x = int(np.ceil(width / feature_chunk_size))
        
        # Initialize feature arrays
        dem_features = np.zeros(batch_size_actual, dtype=np.float32)
        slope_features = np.zeros(batch_size_actual, dtype=np.float32)
        
        # Process each chunk and extract features for pixels in this batch
        for chunk_y in range(n_feature_chunks_y):
            for chunk_x in range(n_feature_chunks_x):
                y_chunk_start = chunk_y * feature_chunk_size
                y_chunk_end = min((chunk_y + 1) * feature_chunk_size, height)
                x_chunk_start = chunk_x * feature_chunk_size
                x_chunk_end = min((chunk_x + 1) * feature_chunk_size, width)
                
                # Find pixels in this batch that belong to this chunk
                in_chunk = ((batch_y_indices >= y_chunk_start) & (batch_y_indices < y_chunk_end) & 
                           (batch_x_indices >= x_chunk_start) & (batch_x_indices < x_chunk_end))
                
                if not np.any(in_chunk):
                    continue
                
                # Load chunk data with padding for slope calculation
                pad_size = 2
                y_start_padded = max(0, y_chunk_start - pad_size)
                x_start_padded = max(0, x_chunk_start - pad_size)
                y_end_padded = min(height, y_chunk_end + pad_size)
                x_end_padded = min(width, x_chunk_end + pad_size)
                
                dem_chunk_padded = dem_dask[y_start_padded:y_end_padded, x_start_padded:x_end_padded].values.astype(np.float32)
                
                # Create mask for this chunk
                if nodata_value is not None and not np.isnan(nodata_value):
                    mask_chunk_padded = np.logical_and(dem_chunk_padded != nodata_value, ~np.isnan(dem_chunk_padded))
                else:
                    mask_chunk_padded = ~np.isnan(dem_chunk_padded)
                
                # Replace invalid values
                dem_chunk_padded[~mask_chunk_padded] = -9999
                
                # Normalize DEM
                dem_norm_padded = dem_chunk_padded.copy()
                dem_norm_padded[mask_chunk_padded] = (dem_chunk_padded[mask_chunk_padded] - dem_mean) / dem_std
                dem_norm_padded[~mask_chunk_padded] = 0
                
                # Calculate slope for this chunk
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    slope_chunk_padded = calc_slope(dem_chunk_padded)
                    slope_chunk_padded[~mask_chunk_padded] = 0
                
                # Extract the actual chunk area from the padded data
                pad_y_offset = y_chunk_start - y_start_padded
                pad_x_offset = x_chunk_start - x_start_padded
                actual_chunk_h = y_chunk_end - y_chunk_start
                actual_chunk_w = x_chunk_end - x_chunk_start
                
                dem_chunk = dem_norm_padded[
                    pad_y_offset:pad_y_offset + actual_chunk_h,
                    pad_x_offset:pad_x_offset + actual_chunk_w
                ]
                
                slope_chunk = slope_chunk_padded[
                    pad_y_offset:pad_y_offset + actual_chunk_h,
                    pad_x_offset:pad_x_offset + actual_chunk_w
                ]
                
                # Extract features for pixels in this batch that belong to this chunk
                batch_y_in_chunk = batch_y_indices[in_chunk] - y_chunk_start
                batch_x_in_chunk = batch_x_indices[in_chunk] - x_chunk_start
                
                dem_features[in_chunk] = dem_chunk[batch_y_in_chunk, batch_x_in_chunk]
                slope_features[in_chunk] = slope_chunk[batch_y_in_chunk, batch_x_in_chunk]
        
        # Add base features to batch
        X_batch[:, 0] = dem_features
        X_batch[:, 1] = slope_features
        X_batch[:, 2] = rainfall_m  # Uniform rainfall for all pixels
        
        # Add model probabilities (remaining columns) from zarr files - vectorized approach
        feature_idx = num_base_features
        
        print(f"  Extracting model probabilities for {batch_size_actual:,} pixels...")
        for model_name in dl_models:
            probs_path = probs_paths[model_name]
            if probs_path is None:
                print(f"    Skipping {model_name} (failed)")
                continue
                
            print(f"    Loading probabilities from {model_name}...")
            
            # Load zarr array
            probs_zarr = zarr.open(probs_path, mode='r')
            
            # Extract probabilities for all 5 classes in batches to avoid memory issues
            prob_batch_size = min(50000, batch_size_actual)  # Process in smaller chunks
            
            for prob_start in range(0, batch_size_actual, prob_batch_size):
                prob_end = min(prob_start + prob_batch_size, batch_size_actual)
                
                prob_y_indices = batch_y_indices[prob_start:prob_end]
                prob_x_indices = batch_x_indices[prob_start:prob_end]
                
                # Read all 5 class probabilities at once using advanced indexing
                probs_batch = probs_zarr[:, prob_y_indices, prob_x_indices]  # Shape: (5, n_pixels)
                
                # Add to feature matrix
                for class_idx in range(5):
                    X_batch[prob_start:prob_end, feature_idx + class_idx] = probs_batch[class_idx]
            
            feature_idx += 5
        
        # Return this batch
        batch_indices = np.column_stack((batch_y_indices, batch_x_indices))
        yield X_batch, batch_indices, height, width

def run_meta_model_inference(data, meta_model, base_models, output_file, 
                           window_size=512, overlap=128, sigma=0.5, batch_size=1000000, n_workers=None):
    """
    Run inference using the meta-model on the provided DEM data.
    Processes large DEMs with memory-efficient approaches.
    
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
    
    # Get base model predictions with memory-efficient approach
    pred_result = get_model_predictions(data, base_models, output_file, 
                                      window_size=window_size, overlap=overlap, sigma=sigma, n_workers=n_workers)
    
    if pred_result is None:
        print("No valid predictions. Exiting.")
        return None
    
    # Get nodata value from metadata
    output_nodata = data['output_nodata']
    
    # Get dimensions
    height = data['height']
    width = data['width']
    
    # Initialize the full prediction array with nodata value
    print(f"Initializing prediction array of size {height}x{width}")
    meta_pred = np.ones((height, width), dtype=np.uint8) * output_nodata
    
    # Use smaller batch size for very large arrays to be more conservative
    if height * width > 50000000:  # >50M pixels
        batch_size = min(batch_size, 500000)  # Use smaller batches
        print(f"Large DEM detected, using conservative batch size: {batch_size:,}")
    
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
            y_indices = batch_indices[:, 0]
            x_indices = batch_indices[:, 1]
            
            # Assign predictions
            for i in range(len(batch_indices)):
                meta_pred[y_indices[i], x_indices[i]] = meta_pred_batch[i]
            
            # Report progress
            print(f"Batch {batch_idx+1} processed in {batch_time:.2f} seconds")
            
            # Clear some memory
            del X_batch, meta_pred_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
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
        save_zarr(meta_pred, meta_zarr_path, data['geotiff_meta'], original_mask=None)
        
        # Save top 3 base model predictions
        base_models_to_save = min(3, len(pred_result['predictions']))
        if pred_result['predictions']:
            model_names = list(pred_result['predictions'].keys())
            
            for i, model_name in enumerate(model_names[:base_models_to_save]):
                base_output_file = output_file.replace('.zarr', f'_{model_name}.zarr')
                save_zarr(pred_result['predictions'][model_name], 
                         base_output_file, data['geotiff_meta'], original_mask=None)
        
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
        
        # Get data from the lazy-loaded structure
        dem_dask = data['dem_dask']
        height = data['height']
        width = data['width']
        rainfall_mm = data['rainfall_mm']
        meta_pred = pred_result['meta_prediction']
        base_preds = pred_result['predictions']
        output_nodata = data['output_nodata']
        county_info = data['county_info']
        nodata_value = data['nodata_value']
        dem_mean = data['dem_mean']
        dem_std = data['dem_std']
        
        # Check if the visualization would be too large
        total_pixels = height * width
        max_viz_pixels = 2000000  # 2 million pixels max for visualization
        
        # Calculate downsampling ratio if needed
        if total_pixels > max_viz_pixels:
            downsample_ratio = int(np.ceil(np.sqrt(total_pixels / max_viz_pixels)))
            print(f"Image too large for visualization ({height}x{width}). Downsampling by factor of {downsample_ratio}.")
            
            # Downsample DEM data
            dem_sampled = dem_dask[::downsample_ratio, ::downsample_ratio].values.astype(np.float32)
            
            # Create mask and normalize sampled DEM
            if nodata_value is not None and not np.isnan(nodata_value):
                mask_sampled = np.logical_and(dem_sampled != nodata_value, ~np.isnan(dem_sampled))
            else:
                mask_sampled = ~np.isnan(dem_sampled)
            
            # Normalize DEM for display
            dem_norm_sampled = dem_sampled.copy()
            dem_norm_sampled[mask_sampled] = (dem_sampled[mask_sampled] - dem_mean) / dem_std
            dem_norm_sampled[~mask_sampled] = np.nan
            
            # Downsample predictions
            meta_pred_ds = meta_pred[::downsample_ratio, ::downsample_ratio]
            
            downsampled_base_preds = {}
            for model_name, pred in base_preds.items():
                downsampled_base_preds[model_name] = pred[::downsample_ratio, ::downsample_ratio]
            base_preds = downsampled_base_preds
        else:
            # Load full DEM for smaller images
            print("Loading full DEM for visualization...")
            dem_full = dem_dask.values.astype(np.float32)
            
            # Create mask
            if nodata_value is not None and not np.isnan(nodata_value):
                mask_full = np.logical_and(dem_full != nodata_value, ~np.isnan(dem_full))
            else:
                mask_full = ~np.isnan(dem_full)
            
            # Normalize DEM for display
            dem_norm_sampled = dem_full.copy()
            dem_norm_sampled[mask_full] = (dem_full[mask_full] - dem_mean) / dem_std
            dem_norm_sampled[~mask_full] = np.nan
            mask_sampled = mask_full
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
        axes[0].set_title("DEM")
        im = axes[0].imshow(dem_norm_sampled, cmap='terrain')
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

def process_window_batch_threaded(window_coords_batch, model, dem_dask, 
                                 rainfall_m, dem_mean, dem_std, nodata_value, 
                                 window_size, weight_map, gpu_lock):
    """
    Process a batch of windows using shared model and data (thread-safe).
    
    Args:
        window_coords_batch: List of (y_idx, x_idx, y_start, x_start, y_end, x_end) tuples
        model: Shared model instance
        dem_dask: Shared DEM dask array
        rainfall_m: Rainfall value in meters
        dem_mean: DEM mean for normalization
        dem_std: DEM standard deviation for normalization
        nodata_value: No data value
        window_size: Window size for processing
        weight_map: Gaussian weight map
        gpu_lock: Threading lock for GPU access
        
    Returns:
        List of (window_coords, predictions, weights) tuples
    """
    results = []
    
    for window_coords in window_coords_batch:
        try:
            y_idx, x_idx, y_start, x_start, y_end, x_end = window_coords
            
            # Calculate padded window coordinates for slope calculation
            pad_size = 2
            height, width = dem_dask.shape
            y_start_padded = max(0, y_start - pad_size)
            x_start_padded = max(0, x_start - pad_size)
            y_end_padded = min(height, y_end + pad_size)
            x_end_padded = min(width, x_end + pad_size)
            
            # Load only the padded window data from the dask array
            dem_raw_padded = dem_dask[y_start_padded:y_end_padded, x_start_padded:x_end_padded].values.astype(np.float32)
            
            # Create a mask for valid data in the padded window
            if nodata_value is not None and not np.isnan(nodata_value):
                mask_padded = np.logical_and(dem_raw_padded != nodata_value, ~np.isnan(dem_raw_padded))
            else:
                mask_padded = ~np.isnan(dem_raw_padded)
            
            # Replace invalid values with a sentinel value for processing
            dem_raw_padded[~mask_padded] = -9999
            
            # Calculate slope for the padded window
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                slope_padded = calc_slope(dem_raw_padded)
                slope_padded[~mask_padded] = 0
            
            # Normalize the DEM data for the padded window
            dem_norm_padded = dem_raw_padded.copy()
            dem_norm_padded[mask_padded] = (dem_raw_padded[mask_padded] - dem_mean) / dem_std
            dem_norm_padded[~mask_padded] = 0
            
            # Extract the actual window area from the padded data
            pad_y_offset = y_start - y_start_padded
            pad_x_offset = x_start - x_start_padded
            actual_window_h = y_end - y_start
            actual_window_w = x_end - x_start
            
            slope_window = slope_padded[
                pad_y_offset:pad_y_offset + actual_window_h,
                pad_x_offset:pad_x_offset + actual_window_w
            ]
            
            dem_window = dem_norm_padded[
                pad_y_offset:pad_y_offset + actual_window_h,
                pad_x_offset:pad_x_offset + actual_window_w
            ]
            
            window_mask = mask_padded[
                pad_y_offset:pad_y_offset + actual_window_h,
                pad_x_offset:pad_x_offset + actual_window_w
            ]
            
            # Create rainfall window
            rainfall_window = np.ones_like(dem_window) * rainfall_m
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
            
            # Use GPU lock to prevent conflicts
            with gpu_lock:
                with torch.no_grad():
                    output = model(input_tensor)
                    pred = output.output.squeeze().detach().cpu().numpy()
            
            # Extract actual window dimensions and weights
            window_h = y_end - y_start
            window_w = x_end - x_start
            current_weight = weight_map[:window_h, :window_w]
            
            # Apply weights to prediction probabilities
            weighted_probs = np.zeros((pred.shape[0], window_h, window_w), dtype=np.float32)
            for c in range(pred.shape[0]):
                weighted_probs[c] = pred[c, :window_h, :window_w] * current_weight
            
            results.append((window_coords, weighted_probs, current_weight))
            
        except Exception as e:
            print(f"Error processing window {window_coords}: {e}")
            continue
    
    return results

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
    print(f"Parallel workers for window processing: {args.n_workers or 'auto'}")
    
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
        batch_size=args.batch_size,
        n_workers=args.n_workers
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
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap between adjacent windows in pixels (default: 128)')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Standard deviation for Gaussian kernel (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=5000000,
                        help='Maximum number of pixels to process in each batch (default: 5,000,000)')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of parallel jobs for meta-model prediction (-1 for all cores, default: -1)')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel threads for window processing (default: min(8, cpu_count()))')
    
    args = parser.parse_args()
    main(args)
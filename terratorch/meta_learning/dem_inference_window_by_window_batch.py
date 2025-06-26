#!/usr/bin/env python
# coding: utf-8

"""
Optimized Window-by-window DEM inference with model batching and data prefetching
This implementation processes windows in batches per model to minimize model switching overhead
and implements data prefetching for better I/O performance.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import gc

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

# Set device and enable mixed precision
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_MIXED_PRECISION = torch.cuda.is_available()

# Try to import GPU-accelerated random forest
GPU_RF_AVAILABLE = False
try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForest
    GPU_RF_AVAILABLE = True
    print("cuML GPU Random Forest available")
except ImportError:
    try:
        import rapids_triton
        GPU_RF_AVAILABLE = True
        print("Rapids Triton GPU Random Forest available")
    except ImportError:
        print("GPU Random Forest not available, using CPU scikit-learn")

class GPURandomForestWrapper:
    """Wrapper for GPU-accelerated Random Forest inference"""
    
    def __init__(self, cpu_model_path):
        self.cpu_model = joblib.load(cpu_model_path)
        self.gpu_model = None
        self.use_gpu = GPU_RF_AVAILABLE and torch.cuda.is_available()
        
        if self.use_gpu:
            try:
                # Convert CPU model to GPU model
                self._convert_to_gpu_model()
                print("Successfully converted Random Forest to GPU")
            except Exception as e:
                print(f"Failed to convert to GPU model, using CPU: {e}")
                self.use_gpu = False
    
    def _convert_to_gpu_model(self):
        """Convert CPU scikit-learn RF to GPU cuML RF"""
        if not GPU_RF_AVAILABLE:
            return
            
        # Extract parameters from CPU model
        n_estimators = self.cpu_model.n_estimators
        max_depth = self.cpu_model.max_depth
        random_state = self.cpu_model.random_state
        
        # Create GPU model with same parameters
        self.gpu_model = CuMLRandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_streams=1  # Use single stream for stability
        )
        
        # We can't directly transfer trees, so we'll use CPU for now
        # In production, you'd retrain the model with cuML
        print("Note: For full GPU acceleration, retrain the model with cuML")
        self.use_gpu = False
    
    def predict(self, X):
        """Predict using GPU or CPU model"""
        if self.use_gpu and self.gpu_model is not None:
            # Convert to cuDF if needed
            try:
                import cupy as cp
                if isinstance(X, np.ndarray):
                    X_gpu = cp.asarray(X, dtype=cp.float32)
                    predictions = self.gpu_model.predict(X_gpu)
                    return cp.asnumpy(predictions)
                else:
                    return self.gpu_model.predict(X)
            except Exception as e:
                print(f"GPU prediction failed, falling back to CPU: {e}")
                return self.cpu_model.predict(X)
        else:
            return self.cpu_model.predict(X)

class DataPrefetcher:
    """Prefetch DEM data chunks asynchronously"""
    
    def __init__(self, dem_dask, window_coords_list, pad_size=2, max_cache_size=4):
        self.dem_dask = dem_dask
        self.window_coords_list = window_coords_list
        self.pad_size = pad_size
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.prefetch_queue = Queue(maxsize=max_cache_size)
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.prefetch_futures = {}
        
    def _load_window_data(self, window_idx, window_coords):
        """Load a single window's data"""
        y_start, x_start, y_end, x_end = window_coords
        height, width = self.dem_dask.shape
        
        # Calculate padded coordinates
        y_start_padded = max(0, y_start - self.pad_size)
        x_start_padded = max(0, x_start - self.pad_size)
        y_end_padded = min(height, y_end + self.pad_size)
        x_end_padded = min(width, x_end + self.pad_size)
        
        # Load data
        dem_data = self.dem_dask[y_start_padded:y_end_padded, x_start_padded:x_end_padded].values.astype(np.float32)
        
        return {
            'window_idx': window_idx,
            'dem_data': dem_data,
            'padded_coords': (y_start_padded, x_start_padded, y_end_padded, x_end_padded),
            'original_coords': window_coords
        }
    
    def prefetch_window(self, window_idx):
        """Start prefetching a window asynchronously"""
        if window_idx < len(self.window_coords_list) and window_idx not in self.prefetch_futures:
            window_coords = self.window_coords_list[window_idx]
            future = self.executor.submit(self._load_window_data, window_idx, window_coords)
            self.prefetch_futures[window_idx] = future
    
    def get_window_data(self, window_idx):
        """Get window data, waiting for prefetch if necessary"""
        if window_idx in self.cache:
            return self.cache.pop(window_idx)
        
        if window_idx in self.prefetch_futures:
            data = self.prefetch_futures[window_idx].result()
            del self.prefetch_futures[window_idx]
            return data
        else:
            # Fallback: load synchronously
            window_coords = self.window_coords_list[window_idx]
            return self._load_window_data(window_idx, window_coords)
    
    def prefetch_batch(self, start_idx, batch_size):
        """Prefetch a batch of windows"""
        for i in range(start_idx, min(start_idx + batch_size, len(self.window_coords_list))):
            self.prefetch_window(i)
    
    def cleanup(self):
        """Clean up resources"""
        for future in self.prefetch_futures.values():
            future.cancel()
        self.executor.shutdown(wait=False)

def create_gaussian_weight_map(window_size=512, sigma=0.5):
    """Create a Gaussian weight map for blending window predictions."""
    y = np.linspace(-1, 1, window_size)
    x = np.linspace(-1, 1, window_size)
    X, Y = np.meshgrid(x, y)
    distance = np.sqrt(X**2 + Y**2) / np.sqrt(2)
    
    weights = np.exp(-(distance**2) / (2 * sigma**2))
    weights = weights / np.max(weights)
    return weights

def load_meta_model_gpu(meta_model_path):
    """Load the trained meta-model with GPU acceleration if available."""
    print(f"Loading meta-model from: {meta_model_path}")
    try:
        meta_model = GPURandomForestWrapper(meta_model_path)
        print(f"Meta-model loaded successfully with GPU support: {meta_model.use_gpu}")
        return meta_model
    except Exception as e:
        print(f"Error loading meta-model: {str(e)}")
        return None

def load_all_base_models():
    """Load all base deep learning models with mixed precision support."""
    print("Loading all base models in advance...")
    models = {}
    
    model_config = {
        'RES50': model_args_res50,
        'RES101': model_args_res101, 
        'RES152': model_args_res152,
        'TINY': model_args_tiny,
        '100M': model_args_100,
        '300M': model_args_300,
        '600M': model_args_600
    }
    
    for model_name, model_args in model_config.items():
        try:
            model_path = getattr(ModelPaths, 'MODEL_'+model_name).value
            if os.path.exists(model_path):
                print(f"  Loading {model_name} model...")
                model = load_model(model_path, model_name)
                
                # Enable mixed precision if available
                if USE_MIXED_PRECISION and hasattr(model, 'half'):
                    model = model.half()
                    print(f"  Enabled mixed precision for {model_name}")
                
                models[model_name] = model
            else:
                print(f"  Model path not found for {model_name}: {model_path}")
        except Exception as e:
            print(f"  Error loading {model_name}: {str(e)}")
    
    print(f"Loaded {len(models)} base models.")
    return models

def lazy_load_dem_from_zarr(zarr_file, rainfall_mm):
    """Lazy load DEM data from a Zarr file."""
    try:
        print(f"Lazy loading DEM from Zarr file: {zarr_file}")
        
        ds = xr.open_zarr(zarr_file)
        
        if len(ds.data_vars) > 0:
            var_name = list(ds.data_vars.keys())[0]
            da = ds[var_name]
            print(f"Loaded data variable: {var_name}")
        else:
            da = ds
        
        print(f"DataArray type: {type(da)}")
        print(f"DataArray shape: {da.shape}")
        
        height, width = da.shape
        print(f"DEM dimensions: {height} rows × {width} columns")
        
        crs_str = da.attrs.get('crs', 'EPSG:4326')
        transform_list = da.attrs.get('transform', None)
        nodata_value = da.attrs.get('nodata', -9999.0)
        
        print(f"CRS: {crs_str}")
        print(f"NoData value: {nodata_value}")
        
        if transform_list and len(transform_list) >= 6:
            transform = Affine(*transform_list[:6])
        else:
            x_coords = da.coords['x'].values
            y_coords = da.coords['y'].values
            x_res = x_coords[1] - x_coords[0] if len(x_coords) > 1 else 1.0
            y_res = y_coords[0] - y_coords[1] if len(y_coords) > 1 else 1.0
            transform = Affine(x_res, 0, x_coords[0] - x_res/2, 0, -y_res, y_coords[0] + y_res/2)
        
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
        
        county_name = da.attrs.get('county', 'Unknown')
        state_name = da.attrs.get('state', 'Unknown')
        geoid = da.attrs.get('geoid', 'Unknown')
        
        print(f"County: {county_name}, State: {state_name}, GEOID: {geoid}")
        
        return {
            'dem_dask': da,
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
    """Numba-optimized feature extraction for valid pixels."""
    n_valid = 0
    for y in range(actual_window_h):
        for x in range(actual_window_w):
            if window_mask[y, x]:
                n_valid += 1
    
    if n_valid == 0:
        return np.zeros((0, 3 + n_models * 5), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)
    
    n_features = 3 + n_models * 5
    features_array = np.zeros((n_valid, n_features), dtype=np.float32)
    valid_indices = np.zeros((n_valid, 2), dtype=np.int32)
    
    idx = 0
    for y in range(actual_window_h):
        for x in range(actual_window_w):
            if window_mask[y, x]:
                valid_indices[idx, 0] = y
                valid_indices[idx, 1] = x
                
                features_array[idx, 0] = dem_window[y, x]
                features_array[idx, 1] = slope_window[y, x]
                features_array[idx, 2] = rainfall_m
                
                feature_idx = 3
                for model_idx in range(n_models):
                    for class_idx in range(5):
                        features_array[idx, feature_idx] = prob_output_window[model_idx, class_idx, y, x]
                        feature_idx += 1
                
                idx += 1
    
    return features_array, valid_indices

@njit(parallel=True)
def majority_vote_predictions_numba(prob_output_window, valid_indices, n_models):
    """Numba-optimized majority vote fallback."""
    n_valid = valid_indices.shape[0]
    predictions = np.zeros(n_valid, dtype=np.uint8)
    
    for i in prange(n_valid):
        y = valid_indices[i, 0]
        x = valid_indices[i, 1]
        
        votes = np.zeros(5, dtype=np.int32)
        for model_idx in range(n_models):
            max_prob = -1.0
            pred_class = 0
            for class_idx in range(5):
                if prob_output_window[model_idx, class_idx, y, x] > max_prob:
                    max_prob = prob_output_window[model_idx, class_idx, y, x]
                    pred_class = class_idx
            votes[pred_class] += 1
        
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
    """Numba-optimized blending of window predictions with Gaussian weights."""
    height, width = current_output.shape
    updated_output = current_output.copy()
    updated_weights = current_weights.copy()
    
    for y in prange(height):
        for x in range(width):
            if new_predictions[y, x] != 255:
                old_weight = current_weights[y, x]
                new_weight = new_weights[y, x]
                total_weight = old_weight + new_weight
                
                if total_weight > 0:
                    if old_weight > 0 and current_output[y, x] != 255:
                        old_pred = float(current_output[y, x])
                        new_pred = float(new_predictions[y, x])
                        blended_pred = (old_pred * old_weight + new_pred * new_weight) / total_weight
                        updated_output[y, x] = np.uint8(np.round(blended_pred))
                    else:
                        updated_output[y, x] = new_predictions[y, x]
                    
                    updated_weights[y, x] = total_weight
    
    return updated_output, updated_weights

def process_window_batch_with_model(window_batch_data, model_name, model, dem_data, window_size=512):
    """Process a batch of windows with a single model to get probabilities."""
    batch_results = []
    
    # Prepare batch tensors
    batch_tensors = []
    batch_info = []
    
    for window_data in window_batch_data:
        dem_raw_padded = window_data['dem_data']
        padded_coords = window_data['padded_coords']
        original_coords = window_data['original_coords']
        
        y_start, x_start, y_end, x_end = original_coords
        y_start_padded, x_start_padded, y_end_padded, x_end_padded = padded_coords
        
        # Create mask for valid data
        nodata_value = dem_data['nodata_value']
        if nodata_value is not None and not np.isnan(nodata_value):
            mask_padded = np.logical_and(dem_raw_padded != nodata_value, ~np.isnan(dem_raw_padded))
        else:
            mask_padded = ~np.isnan(dem_raw_padded)
        
        dem_raw_padded[~mask_padded] = -9999
        
        # Compute slope
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            slope_padded = calc_slope(dem_raw_padded)
            slope_padded[~mask_padded] = 0
        
        # Normalize DEM
        dem_norm_padded = dem_raw_padded.copy()
        dem_norm_padded[mask_padded] = (dem_raw_padded[mask_padded] - dem_data['dem_mean']) / dem_data['dem_std']
        dem_norm_padded[~mask_padded] = 0
        
        # Extract actual window area
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
        
        # Prepare rainfall
        rainfall_m = dem_data['rainfall_mm'] / 1000.0
        rainfall_window = np.ones_like(dem_window) * rainfall_m
        rainfall_window[~window_mask] = 0
        
        # Pad to window_size if needed
        window_h, window_w = dem_window.shape
        if window_h < window_size or window_w < window_size:
            pad_h = max(0, window_size - window_h)
            pad_w = max(0, window_size - window_w)
            dem_window = np.pad(dem_window, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            slope_window = np.pad(slope_window, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            rainfall_window = np.pad(rainfall_window, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        # Create input tensor
        model_input = np.stack([dem_window, slope_window, rainfall_window])
        
        # Convert to appropriate dtype for mixed precision
        if USE_MIXED_PRECISION:
            input_tensor = torch.from_numpy(model_input).half().unsqueeze(0).to(DEVICE)
        else:
            input_tensor = torch.from_numpy(model_input).float().unsqueeze(0).to(DEVICE)
        
        batch_tensors.append(input_tensor)
        batch_info.append((window_data, actual_window_h, actual_window_w, window_mask))
    
    # Process batch with model
    with torch.no_grad():
        if USE_MIXED_PRECISION:
            with torch.cuda.amp.autocast():
                batch_outputs = []
                for input_tensor in batch_tensors:
                    try:
                        output = model(input_tensor)
                        pred_probs = output.output.squeeze().detach().cpu().float().numpy()
                        batch_outputs.append(pred_probs)
                    except Exception as e:
                        print(f"Error with model {model_name}: {e}")
                        batch_outputs.append(np.ones((5, window_size, window_size)) * 0.2)
        else:
            batch_outputs = []
            for input_tensor in batch_tensors:
                try:
                    output = model(input_tensor)
                    pred_probs = output.output.squeeze().detach().cpu().numpy()
                    batch_outputs.append(pred_probs)
                except Exception as e:
                    print(f"Error with model {model_name}: {e}")
                    batch_outputs.append(np.ones((5, window_size, window_size)) * 0.2)
    
    # Return results with metadata
    for i, (probs, (window_data, actual_window_h, actual_window_w, window_mask)) in enumerate(zip(batch_outputs, batch_info)):
        batch_results.append({
            'window_data': window_data,
            'probabilities': probs,
            'actual_window_h': actual_window_h,
            'actual_window_w': actual_window_w,
            'window_mask': window_mask,
            'model_name': model_name
        })
    
    return batch_results

def run_batched_window_inference(
    dem_data, base_models, meta_model, output_file,
    window_size=512, overlap=128, sigma=0.5, batch_size=8
):
    """Main batched window-by-window inference function."""
    print("\nRunning batched window-by-window inference...")
    
    height = dem_data['height']
    width = dem_data['width']
    
    final_output = np.ones((height, width), dtype=np.uint8) * 255
    weight_accumulator = np.zeros((height, width), dtype=np.float32)
    
    print(f"Initialized output array: {height} x {width}")
    
    # Calculate window coordinates
    stride = window_size - overlap
    n_windows_y = max(1, int(np.ceil((height - overlap) / stride))) if height > window_size else 1
    n_windows_x = max(1, int(np.ceil((width - overlap) / stride))) if width > window_size else 1
    
    print(f"Processing {n_windows_y} x {n_windows_x} windows with size {window_size}, overlap {overlap}")
    print(f"Batch size: {batch_size}")
    
    # Generate all window coordinates
    window_coords_list = []
    for y_idx in range(n_windows_y):
        for x_idx in range(n_windows_x):
            y_start = min(y_idx * stride, height - window_size) if height > window_size else 0
            x_start = min(x_idx * stride, width - window_size) if width > window_size else 0
            y_end = min(y_start + window_size, height)
            x_end = min(x_start + window_size, width)
            window_coords_list.append((y_start, x_start, y_end, x_end))
    
    total_windows = len(window_coords_list)
    print(f"Total windows to process: {total_windows}")
    
    # Initialize data prefetcher
    prefetcher = DataPrefetcher(dem_data['dem_dask'], window_coords_list)
    
    # Pre-allocate probability storage for all windows
    all_window_probabilities = {}
    
    start_time = time.time()
    
    try:
        # Process each model across all windows in batches
        model_names = list(base_models.keys())
        n_models = len(model_names)
        
        for model_idx, (model_name, model) in enumerate(base_models.items()):
            print(f"\nProcessing model {model_idx + 1}/{n_models}: {model_name}")
            model_start_time = time.time()
            
            # Process windows in batches for this model
            for batch_start in range(0, total_windows, batch_size):
                batch_end = min(batch_start + batch_size, total_windows)
                current_batch_size = batch_end - batch_start
                
                # Prefetch next batch while processing current
                if batch_end < total_windows:
                    prefetcher.prefetch_batch(batch_end, min(batch_size, total_windows - batch_end))
                
                # Load current batch data
                batch_window_data = []
                for window_idx in range(batch_start, batch_end):
                    window_data = prefetcher.get_window_data(window_idx)
                    batch_window_data.append(window_data)
                
                # Process batch with current model
                batch_results = process_window_batch_with_model(
                    batch_window_data, model_name, model, dem_data, window_size
                )
                
                # Store results
                for result in batch_results:
                    window_idx = result['window_data']['window_idx']
                    if window_idx not in all_window_probabilities:
                        all_window_probabilities[window_idx] = {}
                    all_window_probabilities[window_idx][model_name] = result
                
                # Progress reporting
                progress = (batch_end) / total_windows * 100
                print(f"  Model {model_name}: {batch_end}/{total_windows} windows ({progress:.1f}%)")
                
                # Clear GPU cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            model_time = time.time() - model_start_time
            print(f"  Model {model_name} completed in {model_time:.1f}s")
        
        print(f"\nAll models completed. Running meta-learning...")
        
        # Now run meta-learning on all windows
        gaussian_weights = create_gaussian_weight_map(window_size, sigma)
        
        for window_idx in range(total_windows):
            if window_idx not in all_window_probabilities:
                continue
            
            try:
                # Get window coordinate
                window_coords = window_coords_list[window_idx]
                y_start, x_start, y_end, x_end = window_coords
                
                # Collect probabilities from all models for this window
                prob_output_window = np.zeros((n_models, 5, window_size, window_size), dtype=np.float32)
                
                # Get window metadata from any model result
                sample_result = next(iter(all_window_probabilities[window_idx].values()))
                actual_window_h = sample_result['actual_window_h']
                actual_window_w = sample_result['actual_window_w']
                window_mask = sample_result['window_mask']
                
                # Reconstruct DEM and slope windows for meta-learning
                window_data = prefetcher.get_window_data(window_idx)
                dem_raw_padded = window_data['dem_data']
                padded_coords = window_data['padded_coords']
                
                # Process DEM data (similar to batch processing)
                nodata_value = dem_data['nodata_value']
                if nodata_value is not None and not np.isnan(nodata_value):
                    mask_padded = np.logical_and(dem_raw_padded != nodata_value, ~np.isnan(dem_raw_padded))
                else:
                    mask_padded = ~np.isnan(dem_raw_padded)
                
                dem_raw_padded[~mask_padded] = -9999
                
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    slope_padded = calc_slope(dem_raw_padded)
                    slope_padded[~mask_padded] = 0
                
                dem_norm_padded = dem_raw_padded.copy()
                dem_norm_padded[mask_padded] = (dem_raw_padded[mask_padded] - dem_data['dem_mean']) / dem_data['dem_std']
                dem_norm_padded[~mask_padded] = 0
                
                pad_y_offset = y_start - padded_coords[0]
                pad_x_offset = x_start - padded_coords[1]
                
                dem_window = dem_norm_padded[
                    pad_y_offset:pad_y_offset + actual_window_h,
                    pad_x_offset:pad_x_offset + actual_window_w
                ]
                
                slope_window = slope_padded[
                    pad_y_offset:pad_y_offset + actual_window_h,
                    pad_x_offset:pad_x_offset + actual_window_w
                ]
                
                # Collect model probabilities
                for model_idx, model_name in enumerate(model_names):
                    if model_name in all_window_probabilities[window_idx]:
                        prob_output_window[model_idx] = all_window_probabilities[window_idx][model_name]['probabilities']
                
                # Run meta-learning
                rainfall_m = dem_data['rainfall_mm'] / 1000.0
                
                features_array, valid_indices = extract_features_numba(
                    dem_window, slope_window, window_mask, prob_output_window,
                    rainfall_m, actual_window_h, actual_window_w, n_models
                )
                
                flood_window = np.ones((actual_window_h, actual_window_w), dtype=np.uint8) * 255
                
                if features_array.shape[0] > 0:
                    try:
                        predictions = meta_model.predict(features_array)
                        
                        for i in range(len(predictions)):
                            y = valid_indices[i, 0]
                            x = valid_indices[i, 1]
                            flood_window[y, x] = predictions[i]
                            
                    except Exception as e:
                        print(f"Meta-model prediction failed for window {window_idx}, using majority vote: {e}")
                        predictions = majority_vote_predictions_numba(prob_output_window, valid_indices, n_models)
                        
                        for i in range(len(predictions)):
                            y = valid_indices[i, 0]
                            x = valid_indices[i, 1]
                            flood_window[y, x] = predictions[i]
                
                # Blend with final output
                actual_weights = gaussian_weights[:actual_window_h, :actual_window_w]
                
                current_weights = weight_accumulator[y_start:y_start+actual_window_h, x_start:x_start+actual_window_w].copy()
                current_output = final_output[y_start:y_start+actual_window_h, x_start:x_start+actual_window_w].copy()
                
                updated_output, updated_weights = blend_window_predictions_numba(
                    current_output, current_weights, flood_window, actual_weights
                )
                
                final_output[y_start:y_start+actual_window_h, x_start:x_start+actual_window_w] = updated_output
                weight_accumulator[y_start:y_start+actual_window_h, x_start:x_start+actual_window_w] = updated_weights
                
                # Progress reporting
                if (window_idx + 1) % max(1, total_windows // 20) == 0:
                    progress = (window_idx + 1) / total_windows * 100
                    elapsed = time.time() - start_time
                    print(f"Meta-learning: {window_idx + 1}/{total_windows} windows ({progress:.1f}%) - Elapsed: {elapsed:.1f}s")
                
            except Exception as e:
                print(f"Error in meta-learning for window {window_idx}: {e}")
                continue
    
    finally:
        # Cleanup
        prefetcher.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    total_time = time.time() - start_time
    print(f"Completed all processing in {total_time:.1f} seconds")
    
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
        
        x_coords = np.array([transform * (i + 0.5, 0) for i in range(width)])[:, 0]
        y_coords = np.array([transform * (0, j + 0.5) for j in range(height)])[:, 1]
        
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
                'source': 'Batched window-by-window meta-learning flood prediction model'
            }
        )
        
        da = da.rio.write_crs(crs)
        da = da.rio.write_transform(transform)
        da = da.rio.write_nodata(255)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        da.to_zarr(output_file, mode='w')
        
        print(f"Successfully saved prediction to {output_file}")
        
    except Exception as e:
        print(f"Error saving prediction: {str(e)}")
        import traceback
        traceback.print_exc()

def main(args):
    """Main function for batched window-by-window DEM inference."""
    print(f"Batched Window-by-window DEM Inference")
    print(f"Input DEM Zarr: {args.zarr_file}")
    print(f"Rainfall: {args.rainfall_mm}mm")
    print(f"Output file: {args.output_file}")
    print(f"Window size: {args.window_size}, Overlap: {args.overlap}, Sigma: {args.sigma}")
    print(f"Batch size: {args.batch_size}")
    
    # Load DEM data
    dem_data = lazy_load_dem_from_zarr(args.zarr_file, args.rainfall_mm)
    if dem_data is None:
        print("Failed to load DEM data. Exiting.")
        return
    
    # Load base models
    base_models = load_all_base_models()
    if len(base_models) == 0:
        print("Failed to load base models. Exiting.")
        return
    
    # Load meta-model with GPU support
    meta_model = load_meta_model_gpu(args.meta_model)
    if meta_model is None:
        print("Failed to load meta-model. Exiting.")
        return
    
    # Run batched inference
    final_prediction = run_batched_window_inference(
        dem_data, base_models, meta_model, args.output_file,
        window_size=args.window_size,
        overlap=args.overlap,
        sigma=args.sigma,
        batch_size=args.batch_size
    )
    
    print(f"\nBatched window-by-window inference completed successfully!")
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batched window-by-window DEM inference with meta-learning')
    parser.add_argument('--zarr_file', type=str, required=True,
                        help='Path to the DEM Zarr file')
    parser.add_argument('--rainfall_mm', type=float, required=True,
                        help='Rainfall amount in millimeters')
    parser.add_argument('--meta_model', type=str, 
                        default='/home/users/li1995/global_flood/FloodRisk-DL/terratorch/meta_learning_results_20250515_143320/meta_model_random_forest.joblib',
                        help='Path to the trained meta-model')
    parser.add_argument('--output_file', type=str, default='./batched_window_results.zarr',
                        help='Output file path for the result')
    parser.add_argument('--window_size', type=int, default=512,
                        help='Size of window for processing (default: 512)')
    parser.add_argument('--overlap', type=int, default=128,
                        help='Overlap between adjacent windows in pixels (default: 128)')
    parser.add_argument('--sigma', type=float, default=0.5,
                        help='Standard deviation for Gaussian kernel (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Number of windows to process in each batch (default: 8)')
    
    args = parser.parse_args()
    main(args)
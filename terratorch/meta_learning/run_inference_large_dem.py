#!/usr/bin/env python
"""
Memory-optimized inference script for very large DEM files.
This script uses conservative settings to prevent out-of-memory errors.
"""

import subprocess
import sys
import os
import argparse
import xarray as xr

def get_dem_info(zarr_file):
    """Get basic information about the DEM file."""
    try:
        ds = xr.open_zarr(zarr_file)
        if len(ds.data_vars) > 0:
            var_name = list(ds.data_vars.keys())[0]
            da = ds[var_name]
        else:
            da = ds
        
        height, width = da.shape
        total_pixels = height * width
        
        return {
            'height': height,
            'width': width,
            'total_pixels': total_pixels,
            'size_gb': total_pixels * 4 / (1024**3)  # Estimate size in GB (float32)
        }
    except Exception as e:
        print(f"Error reading DEM info: {e}")
        return None

def get_optimized_params(dem_info):
    """Get optimized parameters based on DEM size."""
    total_pixels = dem_info['total_pixels']
    
    if total_pixels > 100_000_000:  # > 100M pixels (very large)
        return {
            'window_size': 256,  # Smaller windows
            'overlap': 32,       # Minimal overlap
            'batch_size': 500_000,  # Small batches
            'n_workers': 1,      # Single worker
            'n_jobs': 4          # Limited parallel jobs for meta-model
        }
    elif total_pixels > 50_000_000:  # > 50M pixels (large)
        return {
            'window_size': 384,
            'overlap': 48,
            'batch_size': 750_000,
            'n_workers': 2,
            'n_jobs': 8
        }
    elif total_pixels > 10_000_000:  # > 10M pixels (medium)
        return {
            'window_size': 512,
            'overlap': 64,
            'batch_size': 1_000_000,
            'n_workers': 4,
            'n_jobs': -1
        }
    else:  # Small datasets
        return {
            'window_size': 512,
            'overlap': 128,
            'batch_size': 5_000_000,
            'n_workers': None,  # Auto
            'n_jobs': -1
        }

def run_inference(zarr_file, rainfall_mm, output_file, meta_model=None, force_params=None):
    """Run inference with optimized parameters."""
    
    # Get DEM information
    print(f"Analyzing DEM file: {zarr_file}")
    dem_info = get_dem_info(zarr_file)
    
    if dem_info is None:
        print("Failed to analyze DEM file.")
        return False
    
    print(f"DEM dimensions: {dem_info['height']} × {dem_info['width']}")
    print(f"Total pixels: {dem_info['total_pixels']:,}")
    print(f"Estimated size: {dem_info['size_gb']:.2f} GB")
    
    # Get optimized parameters
    if force_params:
        params = force_params
        print("Using user-specified parameters:")
    else:
        params = get_optimized_params(dem_info)
        print("Using optimized parameters for this dataset size:")
    
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Build command
    cmd = [
        'python', 
        os.path.join(os.path.dirname(__file__), 'dem_inference_zarr.py'),
        '--zarr_file', zarr_file,
        '--rainfall_mm', str(rainfall_mm),
        '--output_file', output_file,
        '--window_size', str(params['window_size']),
        '--overlap', str(params['overlap']),
        '--batch_size', str(params['batch_size']),
        '--n_jobs', str(params['n_jobs'])
    ]
    
    if params['n_workers'] is not None:
        cmd.extend(['--n_workers', str(params['n_workers'])])
    
    if meta_model:
        cmd.extend(['--meta_model', meta_model])
    
    print(f"\nRunning command:")
    print(' '.join(cmd))
    print()
    
    # Run the inference
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\nInference completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nInference failed with return code: {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Memory-optimized inference for large DEM files')
    parser.add_argument('--zarr_file', type=str, required=True,
                        help='Path to the DEM Zarr file')
    parser.add_argument('--rainfall_mm', type=float, required=True,
                        help='Rainfall amount in millimeters')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Output file path for the result')
    parser.add_argument('--meta_model', type=str, default=None,
                        help='Path to the trained meta-model (optional)')
    
    # Optional parameter overrides
    parser.add_argument('--force_window_size', type=int, default=None,
                        help='Force specific window size (overrides optimization)')
    parser.add_argument('--force_overlap', type=int, default=None,
                        help='Force specific overlap (overrides optimization)')
    parser.add_argument('--force_batch_size', type=int, default=None,
                        help='Force specific batch size (overrides optimization)')
    parser.add_argument('--force_n_workers', type=int, default=None,
                        help='Force specific number of workers (overrides optimization)')
    parser.add_argument('--force_n_jobs', type=int, default=None,
                        help='Force specific number of jobs (overrides optimization)')
    
    args = parser.parse_args()
    
    # Check if user wants to override parameters
    force_params = None
    if any([args.force_window_size, args.force_overlap, args.force_batch_size, 
            args.force_n_workers, args.force_n_jobs]):
        
        # Get default optimized params first
        dem_info = get_dem_info(args.zarr_file)
        if dem_info:
            force_params = get_optimized_params(dem_info)
            
            # Override with user values
            if args.force_window_size:
                force_params['window_size'] = args.force_window_size
            if args.force_overlap:
                force_params['overlap'] = args.force_overlap
            if args.force_batch_size:
                force_params['batch_size'] = args.force_batch_size
            if args.force_n_workers:
                force_params['n_workers'] = args.force_n_workers
            if args.force_n_jobs:
                force_params['n_jobs'] = args.force_n_jobs
    
    # Run inference
    success = run_inference(
        args.zarr_file, 
        args.rainfall_mm, 
        args.output_file,
        meta_model=args.meta_model,
        force_params=force_params
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
National Map Data Downloader
============================

This script uses the dem_getter.py module to download elevation and other geospatial data 
from The National Map based on county geometry bounding boxes.

Usage:
    python download_national_map_data.py

Author: Generated for FloodRisk-DL project
"""

import os
import sys
from typing import List, Tuple, Optional
from terratorch.dem_getter import (
    get_aws_paths, 
    batch_download, 
    DATASETS_DICT,
    merge_warp_dems
)

def print_available_datasets():
    """Print all available datasets from The National Map"""
    print("Available datasets:")
    print("-" * 50)
    for short_name, full_name in DATASETS_DICT.items():
        print(f"{short_name:12} : {full_name}")
    print("-" * 50)

def validate_bounding_box(bbox: Tuple[float, float, float, float]) -> bool:
    """
    Validate bounding box coordinates
    
    Args:
        bbox: Tuple of (xMin, yMin, xMax, yMax)
        
    Returns:
        bool: True if valid, False otherwise
    """
    xMin, yMin, xMax, yMax = bbox
    
    # Check if coordinates are reasonable for US bounds
    if not (-180 <= xMin < xMax <= -60):  # Longitude bounds for US
        print(f"Warning: Longitude values seem outside US bounds: {xMin}, {xMax}")
        
    if not (20 <= yMin < yMax <= 70):  # Latitude bounds for US
        print(f"Warning: Latitude values seem outside US bounds: {yMin}, {yMax}")
        
    return xMin < xMax and yMin < yMax

def download_county_data(
    bbox: Tuple[float, float, float, float],
    dataset: str = 'DEM_1m',
    output_folder: str = 'national_map_data',
    data_type: str = '',
    input_epsg: int = 4326,
    exclude_redundant: bool = True,
    force_download: bool = False,
    merge_files: bool = False
) -> List[str]:
    """
    Download National Map data for a county bounding box
    
    Args:
        bbox: Tuple of (xMin, yMin, xMax, yMax) in decimal degrees or specified CRS
        dataset: Dataset to download (default: 'DEM_1m')
        output_folder: Folder to save downloaded data
        data_type: Specific data format (leave empty for default)
        input_epsg: EPSG code for input coordinates (default: 4326 for WGS84)
        exclude_redundant: Remove duplicate/overlapping datasets
        force_download: Skip download size confirmation
        merge_files: Merge downloaded rasters into single file
        
    Returns:
        List of downloaded file paths
    """
    
    print(f"Downloading {dataset} data for bounding box: {bbox}")
    print(f"Output folder: {output_folder}")
    print(f"Input CRS: EPSG:{input_epsg}")
    
    # Validate bounding box
    if not validate_bounding_box(bbox):
        raise ValueError("Invalid bounding box coordinates")
    
    # Unpack bounding box
    xMin, yMin, xMax, yMax = bbox
    
    try:
        # Get AWS download paths
        print("\nQuerying The National Map API...")
        aws_paths = get_aws_paths(
            dataset=dataset,
            xMin=xMin,
            yMin=yMin, 
            xMax=xMax,
            yMax=yMax,
            filePath=None,  # Don't save paths to file
            dataType=data_type,
            inputEPSG=input_epsg,
            doExcludeRedundantData=exclude_redundant
        )
        
        if not aws_paths:
            print("No data found for the specified bounding box and dataset.")
            return []
            
        print(f"Found {len(aws_paths)} datasets to download")
        
        # Download the data
        print(f"\nDownloading data to folder: {output_folder}")
        downloaded_files = batch_download(
            dlList=aws_paths,
            folderName=output_folder,
            doForceDownload=force_download
        )
        
        print(f"\nSuccessfully downloaded {len(downloaded_files)} files")
        
        # Optionally merge raster files
        if merge_files and downloaded_files:
            print("\nMerging downloaded raster files...")
            
            # Filter for raster files (common extensions)
            raster_extensions = ['.tif', '.tiff', '.img', '.bil', '.adf']
            raster_files = [
                f for f in downloaded_files 
                if any(f.lower().endswith(ext) for ext in raster_extensions)
            ]
            
            if len(raster_files) > 1:
                merged_filename = os.path.join(output_folder, f"merged_{dataset}.tif")
                print(f"Merging {len(raster_files)} raster files into: {merged_filename}")
                
                merge_warp_dems(
                    inFileNames=raster_files,
                    outFileName=merged_filename,
                    outExtent=[[xMin, xMax], [yMin, yMax]],
                    outEPSG=input_epsg
                )
                
                downloaded_files.append(merged_filename)
                print(f"Merged file saved as: {merged_filename}")
            else:
                print("Only one raster file found, skipping merge.")
        
        return downloaded_files
        
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        raise

def get_user_input():
    """Get user input for download parameters"""
    
    print("National Map Data Downloader")
    print("=" * 40)
    
    # Show available datasets
    print_available_datasets()
    
    # Get dataset choice
    dataset = input(f"\nEnter dataset (default: DEM_1m): ").strip()
    if not dataset:
        dataset = 'DEM_1m'
    
    # Validate dataset choice
    if dataset not in DATASETS_DICT:
        print(f"Invalid dataset. Using default: DEM_1m")
        dataset = 'DEM_1m'
    
    # Get bounding box coordinates
    print(f"\nEnter bounding box coordinates (in decimal degrees, WGS84):")
    print("Format: xMin yMin xMax yMax (space-separated)")
    print("Example for San Francisco County: -122.515 37.708 -122.357 37.833")
    
    bbox_input = input("Bounding box: ").strip()
    try:
        coords = [float(x) for x in bbox_input.split()]
        if len(coords) != 4:
            raise ValueError("Must provide exactly 4 coordinates")
        bbox = tuple(coords)
    except ValueError as e:
        print(f"Invalid coordinates: {e}")
        return None
    
    # Get output folder
    output_folder = input(f"\nOutput folder (default: national_map_data): ").strip()
    if not output_folder:
        output_folder = 'national_map_data'
    
    # Ask about merging files
    merge_choice = input(f"\nMerge downloaded raster files? (y/N): ").strip().lower()
    merge_files = merge_choice in ['y', 'yes']
    
    return {
        'bbox': bbox,
        'dataset': dataset,
        'output_folder': output_folder,
        'merge_files': merge_files
    }

def main():
    """Main function for interactive usage"""
    
    # Example usage with predefined bounding box (commented out)
    # Uncomment and modify for programmatic usage
    """
    # Example: San Francisco County bounding box
    sf_bbox = (-122.515, 37.708, -122.357, 37.833)
    
    downloaded_files = download_county_data(
        bbox=sf_bbox,
        dataset='DEM_1m',
        output_folder='sf_dem_data',
        merge_files=True
    )
    
    print(f"Downloaded files: {downloaded_files}")
    """
    
    # Interactive mode
    try:
        params = get_user_input()
        if params is None:
            return
        
        downloaded_files = download_county_data(**params)
        
        print(f"\nDownload complete!")
        print(f"Files saved to: {params['output_folder']}")
        print(f"Number of files: {len(downloaded_files)}")
        
        if downloaded_files:
            print("\nDownloaded files:")
            for file_path in downloaded_files:
                print(f"  - {file_path}")
                
    except KeyboardInterrupt:
        print("\nDownload cancelled by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
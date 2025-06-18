import os
import glob
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import matplotlib.colors as mcolors
import numpy as np
import gc
from rasterio.enums import Resampling
import rasterio.warp

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Try to import scipy for advanced smoothing, fallback to basic operations if not available
try:
    from scipy import ndimage
    from scipy.ndimage import median_filter, uniform_filter
    SCIPY_AVAILABLE = True
    print("SciPy available - advanced tile boundary smoothing enabled")
except ImportError:
    SCIPY_AVAILABLE = False
    print("SciPy not available - using basic tile boundary smoothing")
    
    # Fallback implementations using numpy only
    def median_filter(input_array, size=3):
        """Simple median filter fallback using numpy"""
        # Simple sliding window median
        pad_width = size // 2
        padded = np.pad(input_array, pad_width, mode='edge')
        result = np.zeros_like(input_array)
        
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                window = padded[i:i+size, j:j+size]
                result[i, j] = np.median(window)
        return result
    
    def uniform_filter(input_array, size=3):
        """Simple uniform filter fallback using numpy"""
        # Simple sliding window average
        pad_width = size // 2
        padded = np.pad(input_array, pad_width, mode='edge')
        result = np.zeros_like(input_array)
        
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                window = padded[i:i+size, j:j+size]
                result[i, j] = np.mean(window)
        return result

# =====================
# CONFIGURATION SECTION
# =====================
# Set your data directory (where dem_data_* folders are)
DATA_DIR = "/home/users/li1995/global_flood/FloodRisk-DL/data/download_US_DEM"  # Change to your base directory if needed
# Set your county shapefile path
COUNTY_SHAPEFILE = "/home/users/li1995/global_flood/FloodRisk-DL/data/download_US_DEM/national_map_downloader/cb_2018_us_county_20m.shp"
# Set your output directory for merged/cropped results
OUTPUT_DIR = "merged_results"

# =====================
# TILE BLENDING SETTINGS
# =====================
# Smoothing kernel size (3-7 recommended, higher = more smoothing)
SMOOTHING_KERNEL_SIZE = 3
# Edge detection threshold (0.5-0.9, lower = more sensitive to boundaries)
EDGE_DETECTION_THRESHOLD = 0.7
# Enable/disable boundary smoothing
ENABLE_BOUNDARY_SMOOTHING = False



def get_state_fips_mapping():
    """Get state FIPS code to abbreviation mapping"""
    return {
        '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE',
        '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA',
        '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN',
        '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM',
        '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
        '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA',
        '54': 'WV', '55': 'WI', '56': 'WY', '72': 'PR'
    }

def load_county_shapefile(shapefile_path):
    """Load the US county shapefile and return GeoDataFrame with readable state names"""
    print("Loading US county shapefile...")
    
    # Load the county shapefile
    counties_gdf = gpd.read_file(shapefile_path)
    
    # Add readable state abbreviations
    state_fips_to_abbrev = get_state_fips_mapping()
    counties_gdf['STATE_ABBREV'] = counties_gdf['STATEFP'].map(state_fips_to_abbrev)
    
    print(f"Loaded {len(counties_gdf)} counties from shapefile")
    print(f"Shapefile CRS: {counties_gdf.crs}")
    
    return counties_gdf

def parse_county_directory_name(county_dir_name):
    """
    Parse county directory name to extract county info and GEOID
    
    Args:
        county_dir_name: Directory name like 'dem_data_San_Francisco_CA_geoid_06075'
        
    Returns:
        dict with county_name, state_abbrev, geoid, or None if parsing fails
    """
    print(f"Parsing directory name: {county_dir_name}")
    
    # Handle new format with GEOID: dem_data_San_Francisco_CA_geoid_06075
    if '_geoid_' in county_dir_name:
        parts = county_dir_name.split('_geoid_')
        if len(parts) == 2:
            geoid = parts[1]
            prefix_parts = parts[0].split('_')
            if len(prefix_parts) >= 4:  # dem_data_San_Francisco_CA
                county_name = '_'.join(prefix_parts[2:-1])  # San_Francisco -> San Francisco
                county_name = county_name.replace('_', ' ')
                state_abbrev = prefix_parts[-1]  # CA
                
                result = {
                    'county_name': county_name,
                    'state_abbrev': state_abbrev,
                    'geoid': geoid,
                    'method': 'geoid'
                }
                print(f"Parsed with GEOID: {result}")
                return result
    
    # Handle legacy format: dem_data_San_Francisco_CA
    parts = county_dir_name.split('_')
    if len(parts) >= 4 and parts[0] == 'dem' and parts[1] == 'data':
        county_name = '_'.join(parts[2:-1])  # Everything except dem_data and last part (state)
        county_name = county_name.replace('_', ' ')
        state_abbrev = parts[-1]  # Last part should be state
        
        result = {
            'county_name': county_name,
            'state_abbrev': state_abbrev,
            'geoid': None,
            'method': 'name'
        }
        print(f"Parsed legacy format: {result}")
        return result
    
    print(f"Could not parse directory name: {county_dir_name}")
    return None

def smooth_tile_boundaries(mosaic_data, kernel_size=3, preserve_categories=True):
    """
    Apply smoothing to reduce tile boundary artifacts while preserving flood categories.
    
    Args:
        mosaic_data: 3D numpy array (bands, height, width) with flood predictions
        kernel_size: Size of smoothing kernel (default: 3)
        preserve_categories: Whether to preserve integer flood categories (default: True)
        
    Returns:
        Smoothed mosaic data
    """
    print(f"Applying boundary smoothing with kernel size {kernel_size}...")
    
    smoothed_data = mosaic_data.copy()
    
    for band in range(mosaic_data.shape[0]):
        band_data = mosaic_data[band].astype(np.float32)
        
        # Create mask for valid data (not nodata)
        valid_mask = band_data != 255
        
        if not np.any(valid_mask):
            continue
            
        # Apply median filter to reduce noise while preserving boundaries
        # Median filter is better for categorical data than mean filters
        smoothed_band = median_filter(band_data, size=kernel_size)
        
        # For areas with high variability (likely tile boundaries), apply additional smoothing
        # Calculate local variance to identify boundary areas
        local_variance = uniform_filter((band_data - uniform_filter(band_data, size=5))**2, size=5)
        high_variance_mask = (local_variance > np.percentile(local_variance[valid_mask], 90)) & valid_mask
        
        # Apply stronger smoothing to high variance areas (tile boundaries)
        if np.any(high_variance_mask):
            # Use a slightly larger kernel for boundary areas
            boundary_smoothed = median_filter(band_data, size=kernel_size + 2)
            smoothed_band[high_variance_mask] = boundary_smoothed[high_variance_mask]
        
        # Preserve nodata values
        smoothed_band[~valid_mask] = 255
        
        # If preserving categories, round to nearest integer
        if preserve_categories:
            smoothed_band = np.round(smoothed_band).astype(np.uint8)
            # Ensure values stay within valid range (0-4 for flood categories, 255 for nodata)
            smoothed_band = np.clip(smoothed_band, 0, 4)
            smoothed_band[~valid_mask] = 255
        
        smoothed_data[band] = smoothed_band
    
    return smoothed_data

def create_weighted_mosaic(src_files_to_mosaic, method='weighted_average'):
    """
    Create a mosaic using weighted averaging to reduce tile boundary artifacts.
    
    Args:
        src_files_to_mosaic: List of open rasterio datasets
        method: Blending method ('weighted_average', 'max', 'median')
        
    Returns:
        tuple: (mosaic_array, transform)
    """
    print(f"Creating weighted mosaic using method: {method}")
    
    if len(src_files_to_mosaic) == 1:
        # Single file, no blending needed
        single_src = src_files_to_mosaic[0]
        data = single_src.read()
        return data, single_src.transform
    
    # Use rasterio's merge with appropriate method
    if method == 'weighted_average':
        # For flood categories, max is often better than mean
        mosaic, out_trans = merge(src_files_to_mosaic, method='max')
    elif method == 'median':
        # Median can help with outliers but is computationally expensive
        mosaic, out_trans = merge(src_files_to_mosaic, method='first')  # Fallback to first
        # TODO: Implement true median merging if needed
    else:
        mosaic, out_trans = merge(src_files_to_mosaic, method=method)
    
    return mosaic, out_trans

def detect_and_fix_tile_boundaries(mosaic_data, threshold=0.8):
    """
    Detect and fix obvious tile boundary artifacts using edge detection.
    
    Args:
        mosaic_data: 3D numpy array (bands, height, width)
        threshold: Threshold for detecting artificial boundaries
        
    Returns:
        Fixed mosaic data
    """
    print("Detecting and fixing tile boundary artifacts...")
    
    fixed_data = mosaic_data.copy()
    
    for band in range(mosaic_data.shape[0]):
        band_data = mosaic_data[band].astype(np.float32)
        valid_mask = band_data != 255
        
        if not np.any(valid_mask):
            continue
        
        # Detect artificial edges (likely tile boundaries)
        # Use Sobel operator to find strong edges
        if SCIPY_AVAILABLE:
            sobel_x = ndimage.sobel(band_data, axis=1, mode='constant', cval=0)
            sobel_y = ndimage.sobel(band_data, axis=0, mode='constant', cval=0)
        else:
            # Simple edge detection fallback
            sobel_x = np.diff(band_data, axis=1, prepend=0)
            sobel_y = np.diff(band_data, axis=0, prepend=0)
        
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Normalize edge magnitude
        edge_magnitude[~valid_mask] = 0
        if np.max(edge_magnitude) > 0:
            edge_magnitude = edge_magnitude / np.max(edge_magnitude)
        
        # Identify strong artificial edges (likely tile boundaries)
        artificial_edges = (edge_magnitude > threshold) & valid_mask
        
        # Apply stronger smoothing to areas near artificial edges
        if np.any(artificial_edges):
            # Dilate the edge mask to include nearby pixels
            if SCIPY_AVAILABLE:
                dilated_edges = ndimage.binary_dilation(artificial_edges, iterations=2)
            else:
                # Simple dilation fallback using numpy
                dilated_edges = artificial_edges.copy()
                for _ in range(2):
                    temp = np.zeros_like(dilated_edges)
                    # 3x3 dilation
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            temp[max(0, di):temp.shape[0]+min(0, di), 
                                 max(0, dj):temp.shape[1]+min(0, dj)] |= \
                                dilated_edges[max(0, -di):dilated_edges.shape[0]+min(0, -di),
                                             max(0, -dj):dilated_edges.shape[1]+min(0, -dj)]
                    dilated_edges = temp
            
            # Apply median filter to edge areas
            smoothed_edges = median_filter(band_data, size=5)
            band_data[dilated_edges] = smoothed_edges[dilated_edges]
        
        # Preserve nodata
        band_data[~valid_mask] = 255
        fixed_data[band] = band_data.astype(np.uint8)
    
    return fixed_data

def find_county_in_shapefile(counties_gdf, county_info):
    """
    Find county in shapefile using GEOID (preferred) or name-based search (fallback)
    
    Args:
        counties_gdf: Loaded county shapefile GeoDataFrame
        county_info: Dictionary from parse_county_directory_name()
        
    Returns:
        Single row GeoDataFrame with the matched county, or None if not found
    """
    
    if county_info['method'] == 'geoid' and county_info['geoid']:
        # Use GEOID for precise matching (preferred method)
        print(f"Looking up county by GEOID: {county_info['geoid']}")
        county_row = counties_gdf[counties_gdf['GEOID'] == county_info['geoid']]
        
        if not county_row.empty:
            county_name = county_row.iloc[0]['NAME']
            state_abbrev = county_row.iloc[0]['STATE_ABBREV']
            print(f"✅ Found county by GEOID: {county_name} County, {state_abbrev}")
            return county_row.iloc[0:1]  # Return single row
        else:
            print(f"❌ County with GEOID {county_info['geoid']} not found in shapefile")
            # Fall back to name-based search
            print(f"Falling back to name-based search...")
    
    # Name-based search (fallback or legacy format)
    if county_info['method'] == 'name' or county_info['geoid'] is None:
        county_name = county_info['county_name']
        state_abbrev = county_info['state_abbrev']
        
        print(f"Looking up county by name: {county_name}, {state_abbrev}")
        
        # Try exact match with state abbreviation
        county_row = counties_gdf[
            (counties_gdf['NAME'].str.contains(county_name, case=False, na=False)) &
            (counties_gdf['STATE_ABBREV'] == state_abbrev)
        ]
        
        if not county_row.empty:
            if len(county_row) > 1:
                print(f"Multiple counties found: {county_row['NAME'].tolist()}")
                county_row = county_row.iloc[0:1]  # Take the first one
            
            found_name = county_row.iloc[0]['NAME']
            found_state = county_row.iloc[0]['STATE_ABBREV']
            print(f"✅ Found county by name: {found_name} County, {found_state}")
            return county_row.iloc[0:1]
        else:
            print(f"❌ County '{county_name}' not found in state '{state_abbrev}'")
            
            # Try broader search within state
            partial_matches = counties_gdf[
                (counties_gdf['NAME'].str.contains(county_name.split()[0], case=False, na=False)) &
                (counties_gdf['STATE_ABBREV'] == state_abbrev)
            ]
            
            if not partial_matches.empty:
                print(f"Possible matches in {state_abbrev}:")
                for idx, row in partial_matches.head(5).iterrows():
                    print(f"  - {row['NAME']} County (GEOID: {row['GEOID']})")
            
            return None
    
    return None

def load_dem_reference(county_dir, county_info):
    """
    Load DEM reference data (either Zarr or TIFF) to get spatial reference information
    
    Args:
        county_dir: County directory path
        county_info: Parsed county information
        
    Returns:
        dict with CRS, transform, bounds information, or None if not found
    """
    
    # Look for TIFF files (both merged and cropped versions)
    tiff_patterns = ["*_merged_dem.tif", "*_cropped_dem*.tif", "USGS_*.tif", "*.tif"]
    
    for pattern in tiff_patterns:
        tif_files = glob.glob(os.path.join(county_dir, pattern))
        # Filter out inference files
        tif_files = [f for f in tif_files if not os.path.basename(f).startswith('inference_')]
        
        if tif_files:
            print(f"Found TIFF DEM file: {os.path.basename(tif_files[0])}")
            try:
                with rasterio.open(tif_files[0]) as src:
                    return {
                        'crs': src.crs,
                        'transform': src.transform,
                        'bounds': src.bounds,
                        'width': src.width,
                        'height': src.height,
                        'source': 'tiff'
                    }
            except Exception as e:
                print(f"Error reading TIFF file: {e}")
                continue
    
    print("No DEM reference file found")
    return None

def merge_and_crop_county(county_dir, shapefile, counties_gdf, output_dir):
    """Merge inference results for a county and crop to county boundary using GEOID lookup."""
    tif_files = sorted(glob.glob(os.path.join(county_dir, "inference_*.tif")))
    if not tif_files:
        print(f"No inference GeoTIFFs found in {county_dir}")
        return

    print(f"Processing {len(tif_files)} inference files in {county_dir}")

    # Parse county directory name
    county_dir_name = os.path.basename(county_dir)
    county_info = parse_county_directory_name(county_dir_name)
    
    if not county_info:
        print(f"❌ Could not parse directory name: {county_dir_name}")
        return
    
    # Find county in shapefile using GEOID or name
    county_row = find_county_in_shapefile(counties_gdf, county_info)
    
    if county_row is None:
        print(f"❌ County not found in shapefile for directory: {county_dir_name}")
        return
    
    # Get county information
    county_geom = county_row.geometry.iloc[0]
    county_name_found = county_row['NAME'].iloc[0]
    state_abbrev_found = county_row['STATE_ABBREV'].iloc[0]
    geoid_found = county_row['GEOID'].iloc[0]
    
    print(f"✅ Processing: {county_name_found} County, {state_abbrev_found} (GEOID: {geoid_found})")

    # Load DEM reference for spatial information
    dem_ref = load_dem_reference(county_dir, county_info)
    if dem_ref:
        print(f"Using {dem_ref['source']} file for spatial reference")
        print(f"Reference CRS: {dem_ref['crs']}")
    else:
        print("Warning: No DEM reference found, using inference file CRS")

    # Merge all rasters with memory optimization
    print("Merging inference rasters...")
    src_files_to_mosaic = []
    target_crs = None
    temp_files = []  # Initialize temp_files list
    
    try:
        # First pass: check CRS of all files
        print("Checking CRS of all input files...")
        crs_info = {}
        for fp in tif_files:
            with rasterio.open(fp) as src:
                file_crs = src.crs
                if file_crs not in crs_info:
                    crs_info[file_crs] = []
                crs_info[file_crs].append(fp)
        
        print(f"Found {len(crs_info)} different CRS:")
        for crs, files in crs_info.items():
            print(f"  {crs}: {len(files)} files")
        
        # Determine target CRS (prefer DEM reference CRS if available)
        if dem_ref and dem_ref['crs']:
            try:
                target_crs = rasterio.crs.CRS.from_string(dem_ref['crs'])
                print(f"Using DEM reference CRS: {target_crs}")
            except:
                target_crs = max(crs_info.keys(), key=lambda x: len(crs_info[x]))
                print(f"Could not parse DEM CRS, using most common inference CRS: {target_crs}")
        else:
            target_crs = max(crs_info.keys(), key=lambda x: len(crs_info[x]))
            print(f"Using most common inference CRS: {target_crs}")
        
        # Second pass: open files and reproject if needed
        # temp_files already initialized above
        
        for fp in tif_files:
            with rasterio.open(fp) as src:
                if src.crs == target_crs:
                    # Same CRS, use directly
                    src_files_to_mosaic.append(rasterio.open(fp))
                else:
                    # Different CRS, need to reproject
                    print(f"Reprojecting {os.path.basename(fp)} from {src.crs} to {target_crs}")
                    
                    # Create temporary reprojected file
                    temp_fp = fp.replace('.tif', '_reprojected_temp.tif')
                    temp_files.append(temp_fp)
                    
                    # Calculate transform and dimensions for target CRS
                    transform, width, height = rasterio.warp.calculate_default_transform(
                        src.crs, target_crs, src.width, src.height, *src.bounds
                    )
                    
                    # Create reprojected raster
                    kwargs = src.meta.copy()
                    kwargs.update({
                        'crs': target_crs,
                        'transform': transform,
                        'width': width,
                        'height': height
                    })
                    
                    with rasterio.open(temp_fp, 'w', **kwargs) as dst:
                        rasterio.warp.reproject(
                            source=rasterio.band(src, 1),
                            destination=rasterio.band(dst, 1),
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=transform,
                            dst_crs=target_crs,
                            resampling=rasterio.warp.Resampling.nearest
                        )
                    
                    # Open the reprojected file for merging
                    src_files_to_mosaic.append(rasterio.open(temp_fp))
        
        # Create seamless mosaic with improved blending
        mosaic, out_trans = create_weighted_mosaic(src_files_to_mosaic, method='max')
        out_meta = src_files_to_mosaic[0].meta.copy()
        
        # Apply boundary smoothing to reduce tile artifacts if enabled
        if ENABLE_BOUNDARY_SMOOTHING:
            mosaic = smooth_tile_boundaries(mosaic, kernel_size=SMOOTHING_KERNEL_SIZE, preserve_categories=True)
            
            # Additional edge detection and fixing for stubborn tile boundaries
            mosaic = detect_and_fix_tile_boundaries(mosaic, threshold=EDGE_DETECTION_THRESHOLD)
        
        # Get CRS from the raster (should be target_crs)
        raster_crs = target_crs
        print(f"Merged inference CRS: {raster_crs}")
        
        # Get raster bounds
        raster_bounds = rasterio.transform.array_bounds(mosaic.shape[1], mosaic.shape[2], out_trans)
        print(f"Merged inference bounds: {raster_bounds}")
        
    except Exception as e:
        print(f"Error during raster merging: {e}")
        import traceback
        traceback.print_exc()
        # Ensure variables are defined for cleanup
        if 'mosaic' not in locals():
            mosaic = None
        if 'out_trans' not in locals():
            out_trans = None
        if 'raster_crs' not in locals():
            raster_crs = None
        if 'raster_bounds' not in locals():
            raster_bounds = None
        if 'out_meta' not in locals():
            out_meta = None
    finally:
        # Close all opened files immediately to free memory
        if 'src_files_to_mosaic' in locals():
            for src in src_files_to_mosaic:
                if src and not src.closed:
                    src.close()
            src_files_to_mosaic.clear()
        
        # Clean up temporary reprojected files
        if 'temp_files' in locals():
            for temp_fp in temp_files:
                if os.path.exists(temp_fp):
                    os.remove(temp_fp)
                    print(f"Cleaned up temporary file: {temp_fp}")
        
        gc.collect()  # Force garbage collection
    
    # Check if we have the required variables to continue
    if mosaic is None or out_trans is None or raster_crs is None:
        print("Failed to merge rasters, cannot continue processing")
        return

    # Reproject county geometry to match raster CRS if needed
    original_county_geom = county_geom
    if counties_gdf.crs != raster_crs:
        print(f"Reprojecting county geometry from {counties_gdf.crs} to {raster_crs}")
        county_geom = gpd.GeoSeries([county_geom], crs=counties_gdf.crs).to_crs(raster_crs).iloc[0]
        print(f"Original county bounds: {original_county_geom.bounds}")
        print(f"Reprojected county bounds: {county_geom.bounds}")
    else:
        print(f"County geometry and inference raster already in same CRS: {raster_crs}")
    
    # Check if geometries overlap
    raster_box = box(*raster_bounds)
    if not county_geom.intersects(raster_box):
        print(f"ERROR: County boundary does not intersect with raster bounds!")
        print(f"County bounds: {county_geom.bounds}")
        print(f"Raster bounds: {raster_bounds}")
        return

    # Update metadata for merged raster
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "crs": raster_crs,
        "compress": "lzw"
    })

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use GEOID in output filename for uniqueness
    output_prefix = f"{county_name_found.replace(' ', '_')}_{state_abbrev_found}_geoid_{geoid_found}"
    
    # Crop to county boundary directly from in-memory mosaic
    print("Cropping inference to county boundary...")
    
    try:
        # Create a temporary in-memory rasterio dataset from the mosaic
        from rasterio.io import MemoryFile
        
        # Create an in-memory rasterio dataset
        with MemoryFile() as memfile:
            with memfile.open(**out_meta) as mem_dataset:
                mem_dataset.write(mosaic)
                
                print(f"Source raster CRS: {mem_dataset.crs}")
                print(f"Source raster bounds: {mem_dataset.bounds}")
                print(f"Source raster transform: {mem_dataset.transform}")
                
                try:
                    # Crop (mask) the merged raster to the county boundary
                    out_image, out_transform = mask(
                        mem_dataset,
                        [county_geom],
                        crop=True,
                        nodata=255
                    )
                    print(f"Successfully cropped raster to county boundary")
                    print(f"Output raster shape: {out_image.shape}")
                    print(f"Output transform: {out_transform}")
                    
                    # Calculate bounds of cropped raster for debugging
                    cropped_bounds = rasterio.transform.array_bounds(
                        out_image.shape[1], out_image.shape[2], out_transform
                    )
                    print(f"Cropped raster bounds: {cropped_bounds}")
                    
                except ValueError as e:
                    print(f"Error during masking: {e}")
                    # Try without cropping
                    out_image, out_transform = mask(
                        mem_dataset,
                        [county_geom],
                        crop=False,
                        nodata=255
                    )
                    print(f"Masked without cropping")
                    print(f"Output raster shape: {out_image.shape}")
                    print(f"Output transform: {out_transform}")
    except Exception as e:
        print(f"Error during in-memory cropping: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Clear mosaic from memory
    del mosaic
    gc.collect()

    # Save cropped inference as GeoTIFF
    out_tif = os.path.join(output_dir, f"{output_prefix}_inference_merged_cropped_smoothed.tif")
    print(f"Saving cropped inference as GeoTIFF to: {out_tif}")

    # Update metadata for the full resolution output
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": 255,
        "compress": "lzw"
    })

    # Save the full resolution GeoTIFF
    with rasterio.open(out_tif, "w", **out_meta) as dest:
        dest.write(out_image)
    
    print(f"Cropped inference saved as GeoTIFF successfully")

    # Also save a small TIFF version for visualization if the image is very large
    out_tif_small = os.path.join(output_dir, f"{output_prefix}_inference_merged_cropped_small.tif")
    print("Saving small TIFF for visualization...")
    
    # Downsample for visualization if too large
    if out_image.shape[1] > 5000 or out_image.shape[2] > 5000:
        print("Creating downsampled version for visualization...")
        downsample_factor = max(out_image.shape[1] // 5000, out_image.shape[2] // 5000, 1)
        small_image = out_image[:, ::downsample_factor, ::downsample_factor]
        small_transform = rasterio.transform.Affine(
            out_transform.a * downsample_factor, out_transform.b, out_transform.c,
            out_transform.d, out_transform.e * downsample_factor, out_transform.f
        )
        
        # Update metadata for small TIFF
        small_meta = out_meta.copy()
        small_meta.update({
            "height": small_image.shape[1],
            "width": small_image.shape[2],
            "transform": small_transform
        })

        with rasterio.open(out_tif_small, "w", **small_meta) as dest:
            dest.write(small_image)
        
        # Use small version for visualization
        visualization_file = out_tif_small
    else:
        print("Image size is reasonable, using full resolution for visualization")
        visualization_file = out_tif

    # Create and save visualization using the small TIFF
    print("Creating visualization...")
    out_png = os.path.join(output_dir, f"{output_prefix}_inference_merged_cropped.png")
    
    # Use the appropriate TIFF file for visualization
    try:
        with rasterio.open(visualization_file) as vis_src:
            vis_data = vis_src.read(1)
            vis_transform = vis_src.transform
            vis_bounds = vis_src.bounds
            
            # Get bounds for plotting
            left, bottom, right, top = vis_bounds.left, vis_bounds.bottom, vis_bounds.right, vis_bounds.top
            
            print(f"Visualization raster size: {vis_data.shape}")
            print(f"Geographic extent: left={left}, right={right}, bottom={bottom}, top={top}")
            print(f"County geometry bounds: {county_geom.bounds}")
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create a custom colormap for flood categories
            colors = ['blue', 'lightblue', 'yellow', 'orange', 'red', 'white']  # 0-4 + nodata
            cmap = mcolors.ListedColormap(colors[:-1])
            cmap.set_bad(colors[-1])  # nodata color
            
            # Mask nodata values
            plot_data = vis_data.astype(float)
            plot_data[plot_data == 255] = float('nan')
            
            print(f"Using extent for plot: left={left}, right={right}, bottom={bottom}, top={top}")
            
            # Display the raster
            im = ax.imshow(plot_data, cmap=cmap, vmin=0, vmax=4, 
                           extent=[left, right, bottom, top], origin='upper')
            
            # Debug: Check if the extent makes sense
            print(f"Raster extent width: {right - left}")
            print(f"Raster extent height: {top - bottom}")
            county_bounds = county_geom.bounds
            print(f"County bounds width: {county_bounds[2] - county_bounds[0]}")
            print(f"County bounds height: {county_bounds[3] - county_bounds[1]}")
            
            # Plot county boundary - ensure it's in the same coordinate system
            try:
                if hasattr(county_geom, 'exterior'):
                    # Single polygon
                    x, y = county_geom.exterior.xy
                    ax.plot(x, y, color='red', linewidth=3, label='County Boundary')
                    print(f"Plotted single polygon boundary with {len(x)} points")
                    print(f"County boundary X range: {min(x)} to {max(x)}")
                    print(f"County boundary Y range: {min(y)} to {max(y)}")
                elif hasattr(county_geom, 'geoms'):
                    # MultiPolygon
                    for i, geom in enumerate(county_geom.geoms):
                        if hasattr(geom, 'exterior'):
                            x, y = geom.exterior.xy
                            ax.plot(x, y, color='red', linewidth=3)
                            print(f"Plotted polygon {i} with {len(x)} points")
                    ax.plot([], [], color='red', linewidth=3, label='County Boundary')  # For legend
                else:
                    print(f"Unknown geometry type: {type(county_geom)}")
            except Exception as e:
                print(f"Error plotting county boundary: {e}")
            
            # Set the axis limits to the county bounds to ensure proper zoom
            ax.set_xlim(county_bounds[0], county_bounds[2])
            ax.set_ylim(county_bounds[1], county_bounds[3])
            
            ax.set_title(f"{county_name_found} County - Merged & Cropped Flood Inference (GEOID: {geoid_found})", fontsize=14)
            ax.set_xlabel('Easting (m)' if not raster_crs.is_geographic else 'Longitude')
            ax.set_ylabel('Northing (m)' if not raster_crs.is_geographic else 'Latitude')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, label='Flood Category (0-4)', shrink=0.8)
            
            # Add legend for county boundary
            ax.legend(loc='upper right')
            
            # Set aspect ratio to equal for proper geographic display
            ax.set_aspect('equal')
            
            plt.savefig(out_png, bbox_inches='tight', dpi=150)
            plt.close()
            
            # Clear variables to free memory
            del vis_data, plot_data
            gc.collect()
            
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"✅ Saved merged and cropped inference as GeoTIFF to {out_tif}")
    if out_image.shape[1] > 5000 or out_image.shape[2] > 5000:
        print(f"✅ Saved small TIFF for visualization to {out_tif_small}")
    print(f"✅ Saved visualization to {out_png}")
    if ENABLE_BOUNDARY_SMOOTHING:
        print(f"✅ Applied tile boundary smoothing (kernel size: {SMOOTHING_KERNEL_SIZE}, edge threshold: {EDGE_DETECTION_THRESHOLD})")
    print("-" * 70)

def main():
    """Main function to process all county directories."""
    print("Inference Results Merge and Crop Tool (with GEOID-based County Lookup)")
    print("=" * 80)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Data directory not found: {DATA_DIR}")
        print("Please update the DATA_DIR variable in the script.")
        return
    
    # Check if shapefile exists
    if not os.path.exists(COUNTY_SHAPEFILE):
        print(f"County shapefile not found: {COUNTY_SHAPEFILE}")
        print("Please update the COUNTY_SHAPEFILE variable in the script.")
        return
    
    # Load county shapefile once for all processing
    print("Loading county shapefile for lookup...")
    try:
        counties_gdf = load_county_shapefile(COUNTY_SHAPEFILE)
    except Exception as e:
        print(f"Error loading county shapefile: {e}")
        return
    
    print(f"Looking for county directories in: {DATA_DIR}")
    county_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "dem_data_Harris_TX_geoid*")))
    print(f"Found {len(county_dirs)} county directories")
    
    if not county_dirs:
        print(f"No county directories found matching pattern: dem_data_*")
        return
    
    print(f"Processing counties...")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Process each county
    for i, county_dir in enumerate(county_dirs, 1):
        print(f"\n{'='*70}")
        print(f"Processing {i}/{len(county_dirs)}: {os.path.basename(county_dir)}")
        print(f"{'='*70}")
        
        try:
            merge_and_crop_county(county_dir, COUNTY_SHAPEFILE, counties_gdf, OUTPUT_DIR)
            print(f"✅ Successfully processed {os.path.basename(county_dir)}")
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(county_dir)}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print("Processing complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Output files use GEOID for unique identification")
    print(f"{'='*70}")

if __name__ == "__main__":
    main() 
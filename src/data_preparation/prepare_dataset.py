import os
import json
import numpy as np
import rasterio
import h5py
from tqdm import tqdm
import glob
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing

# Define HALO size for boundary handling
HALO = 10
OVERLAY=500

def load_city_rainfall_data(json_file):
    """
    Load the rainfall data from the JSON file.
    
    Args:
        json_file (str): Path to the JSON file containing rainfall data
        
    Returns:
        dict: Dictionary mapping city IDs to rainfall values
    """
    with open(json_file, 'r') as f:
        rainfall_data = json.load(f)

    # Create a dictionary mapping city ID to rainfall values
    city_rainfall = {}
    for city in rainfall_data:
        city_id = city['City ID']
        # Convert rainfall values from string to float (remove 'mm' and convert)
        rainfall_values = {
            '100-yr': float(city['100-yr'].split('m')[0]),
            '50-yr': float(city['50-yr'].split('m')[0]),
            '25-yr': float(city['25-yr'].split('m')[0]),
            '10-yr': float(city['10-yr'].split('m')[0])
        }
        city_rainfall[city_id] = rainfall_values
    
    return city_rainfall

def get_timesteps():
    """
    Generate the timestep strings used in the filenames.
    
    Returns:
        list: List of timestep strings
    """
    return [f"{t:07d}" for t in range(0, 2160000 + 1, 30000) if t != 0]

def apply_halo_boundary(data):
    """
    Apply HALO boundary by slicing the array to remove edge effects.
    
    Args:
        data (numpy.ndarray): Input array
        
    Returns:
        numpy.ndarray: Array with HALO boundary removed
    """
    return data[HALO:-HALO, HALO:-HALO]

def create_tiles(data, tile_size=1024, overlap=OVERLAY):
    """
    Create overlapping tiles from a large array.
    
    Args:
        data (numpy.ndarray): Input array to tile
        tile_size (int): Size of each tile
        overlap (int): Number of pixels to overlap between tiles
        
    Returns:
        list: List of (tile, start_h, start_w) tuples
    """
    h, w = data.shape
    tiles = []
    
    # Calculate stride (distance between tile centers)
    stride = tile_size - overlap
    
    # Create tiles
    for h_start in range(0, h, stride):
        for w_start in range(0, w, stride):
            # Calculate tile boundaries
            h_end = min(h_start + tile_size, h)
            w_end = min(w_start + tile_size, w)
            
            # Extract tile
            tile = data[h_start:h_end, w_start:w_end]
            
            # Pad if necessary
            if tile.shape != (tile_size, tile_size):
                padded_tile = np.zeros((tile_size, tile_size), dtype=data.dtype)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
            
            tiles.append((tile, h_start, w_start))
    
    return tiles

def process_tile(args):
    """
    Process a single tile of data.
    
    Args:
        args (tuple): Tuple containing (tile_data, tile_info, rainfall_value)
        
    Returns:
        tuple: (tile_data, tile_info, rainfall_value)
    """
    tile_data, tile_info, rainfall_value = args
    return tile_data, tile_info, rainfall_value

def process_timestep(args):
    """
    Process a single timestep of flood depth data.
    
    Args:
        args (tuple): Tuple containing (flood_path, ts)
        
    Returns:
        tuple: (ts, flood_data) or None if file doesn't exist
    """
    flood_path, ts = args
    if not os.path.exists(flood_path):
        return None
    
    with rasterio.open(flood_path) as src:
        flood_data = src.read(1)  # Read the first band
    return ts, flood_data

def extract_max_flood_depth(dataset_path, city_id, rainfall_type):
    """
    Extract the maximum flood depth across all time steps for a specific city and rainfall.
    
    Args:
        dataset_path (str): Path to the dataset directory
        city_id (str): City ID
        rainfall_type (str): Type of rainfall (e.g., '100-yr')
        
    Returns:
        numpy.ndarray: Maximum flood depth array
    """
    city_dir = os.path.join(dataset_path, city_id)
    timesteps = get_timesteps()
    
    # Create arguments for parallel processing
    args = []
    for ts in timesteps:
        flood_filename = f"{city_id}_{rainfall_type}_WaterDepth_{ts}.tif"
        flood_path = os.path.join(city_dir, flood_filename)
        args.append((flood_path, ts))
    
    # Process timesteps in parallel
    num_workers = multiprocessing.cpu_count()
    max_flood_depth = None
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_timestep, arg) for arg in args]
        for future in as_completed(futures):
            # try:
            result = future.result()
            if result is None:
                continue
                
            ts, flood_data = result
            
            # Initialize max_flood_depth if this is the first valid timestep
            if max_flood_depth is None:
                max_flood_depth = flood_data
            else:
                # Update maximum values
                max_flood_depth = np.maximum(max_flood_depth, flood_data)
            # except Exception as e:
            #     print(f"Error processing timestep for {city_id}, {rainfall_type}: {e}")
    
    # Apply HALO boundary to the maximum flood depth
    if max_flood_depth is not None:
        max_flood_depth = apply_halo_boundary(max_flood_depth)
    
    return max_flood_depth

def load_dem(dataset_path, city_id):
    """
    Load the DEM data for a specific city.
    
    Args:
        dataset_path (str): Path to the dataset directory
        city_id (str): City ID
        
    Returns:
        numpy.ndarray: DEM data
    """
    city_dir = os.path.join(dataset_path, city_id)
    dem_filename = f"{city_id}_DEM.tif"
    dem_path = os.path.join(city_dir, dem_filename)
    
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)  # Read the first band
    
    # Apply HALO boundary to the DEM
    dem_data = apply_halo_boundary(dem_data)
    
    return dem_data

def read_flood_depth_files(args):
    """
    Read all flood depth files for a specific city and rainfall type.
    
    Args:
        args (tuple): Tuple containing (city_dir, city_id, rainfall_type)
        
    Returns:
        tuple: (city_id, rainfall_type, flood_data_list) or None if error
    """
    city_dir, city_id, rainfall_type = args
    timesteps = get_timesteps()
    flood_data_list = []
    
    # try:
    for ts in timesteps:
        flood_filename = f"{city_id}_{rainfall_type}_WaterDepth_{ts}.tif"
        flood_path = os.path.join(city_dir, flood_filename)
        
        if not os.path.exists(flood_path):
            continue
            
        with rasterio.open(flood_path) as src:
            flood_data = src.read(1)  # Read the first band
            flood_data_list.append(flood_data)
    
    if not flood_data_list:
        print(f"Warning: No flood data found for {city_id}, {rainfall_type}")
        return None
        
    # Calculate maximum flood depth across all timesteps
    max_flood_depth = np.maximum.reduce(flood_data_list)
    
    # Apply HALO boundary
    max_flood_depth = apply_halo_boundary(max_flood_depth)
    
    return city_id, max_flood_depth
        
    # except Exception as e:
    #     print(f"Error reading flood depth files for {city_id}, {rainfall_type}: {e}")
    #     return None

def process_city_data(args):
    """
    Process a single city's data without H5 writing.
    
    Args:
        args (tuple): Tuple containing (city_id, dataset_path, city_rainfall)
        
    Returns:
        tuple: (city_id, dem_tiles, flood_tiles_dict)
    """
    city_id, dataset_path, city_rainfall = args
    city_dir = os.path.join(dataset_path, city_id)
    
    # Load and process DEM data
    dem_data = load_dem(dataset_path, city_id)
    dem_tiles = create_tiles(dem_data)
    
    # Process each rainfall type in parallel
    flood_tiles_dict = {}
    rainfall_types = ['100-yr', '50-yr', '25-yr', '10-yr']
    
    # Create arguments for parallel processing
    process_args = []
    for rainfall_type in rainfall_types:
        rain_value_str = f'{int(city_rainfall[city_id][rainfall_type])}mm'
        process_args.append((city_dir, city_id, rain_value_str))
    
    # Process rainfall types in parallel
    num_workers = multiprocessing.cpu_count()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(read_flood_depth_files, arg) for arg in process_args]
        for future in as_completed(futures):
            # try:
            result = future.result()
            if result is None:
                continue
                
            city_id, max_depth = result
            
            # Create tiles for flood depth
            flood_tiles = create_tiles(max_depth)
            flood_tiles_dict[rainfall_type] = {
                'tiles': flood_tiles,
                'rainfall_value': city_rainfall[city_id][rainfall_type]
            }
            # except Exception as e:
            #     print(f"Error processing rainfall type for {city_id}: {e}")
    
    return city_id, dem_tiles, flood_tiles_dict

def create_h5_dataset(dataset_path, output_path, city_rainfall_file, test_cities=None, val_ratio=0.1):
    """
    Create an H5 dataset containing the DEM and maximum flood depth for each city and rainfall level.
    
    Args:
        dataset_path (str): Path to the dataset directory
        output_path (str): Path where the H5 file will be saved
        city_rainfall_file (str): Path to the JSON file containing rainfall data
        test_cities (list): List of city IDs to be used for testing
        val_ratio (float): Ratio of validation data
    """
    # Load rainfall data
    city_rainfall = load_city_rainfall_data(city_rainfall_file)
    
    # Get all city IDs from the dataset directory
    all_cities = list(city_rainfall.keys())
    
    # Filter out test cities
    if test_cities is None:
        test_cities = []
    
    train_val_cities = [city for city in all_cities if city not in test_cities]
    
    # Split into train and validation sets
    random.seed(42)  # For reproducibility
    random.shuffle(train_val_cities)
    val_size = max(1, int(len(train_val_cities) * val_ratio))
    val_cities = train_val_cities[:val_size]
    train_cities = train_val_cities[val_size:]
    
    print(f"Train cities: {train_cities}")
    print(f"Validation cities: {val_cities}")
    print(f"Test cities: {test_cities}")
    
    # Process cities sequentially
    city_results = {}
    for city_id in tqdm(train_cities + val_cities + test_cities, desc="Processing cities"):
        # try:
        city_id, dem_tiles, flood_tiles_dict = process_city_data((city_id, dataset_path, city_rainfall))
        city_results[city_id] = (dem_tiles, flood_tiles_dict)
        # except Exception as e:
        #     print(f"Error processing city: {e}")
    
    # Write results to H5 file
    print("Writing results to H5 file...")
    with h5py.File(output_path, 'w') as h5f:
        # Create groups for train, val, and test
        train_group = h5f.create_group('train')
        val_group = h5f.create_group('val')
        test_group = h5f.create_group('test')
        
        # Write results for each city
        for city_id, (dem_tiles, flood_tiles_dict) in city_results.items():
            # Determine which group to write to
            if city_id in train_cities:
                group = train_group
            elif city_id in val_cities:
                group = val_group
            else:
                group = test_group
            
            # Write data for each rainfall type
            for rainfall_type, data in flood_tiles_dict.items():
                # Create subgroup for city and rainfall level
                subgroup_name = f"{city_id}_{rainfall_type}"
                subgroup = group.create_group(subgroup_name)
                
                # Store rainfall value as attribute
                subgroup.attrs['rainfall_value'] = data['rainfall_value']
                
                # Write tiles
                for i, (dem_tile, dem_h_start, dem_w_start) in enumerate(dem_tiles, start=1):
                    # Get corresponding flood depth tile
                    flood_tile, flood_h_start, flood_w_start = data['tiles'][i-1]
                    
                    # Create combined dataset (2, 1024, 1024)
                    combined_data = np.stack([dem_tile, flood_tile], axis=0)
                    subgroup.create_dataset(f'tile_{i:06d}', data=combined_data, compression='gzip')
                    
                    # Store tile position info
                    subgroup.create_dataset(f'tile_info_{i:06d}', 
                                         data=(dem_h_start, dem_w_start, flood_h_start, flood_w_start),
                                         compression='gzip')
                
                # Store number of tiles
                subgroup.attrs['num_tiles'] = len(dem_tiles)
        
        # Store number of cities in each split
        train_group.attrs['num_cities'] = len(train_cities)
        val_group.attrs['num_cities'] = len(val_cities)
        test_group.attrs['num_cities'] = len(test_cities)
        
        print(f"Dataset created with:")
        print(f"Train cities: {len(train_cities)}")
        print(f"Validation cities: {len(val_cities)}")
        print(f"Test cities: {len(test_cities)}")

def main():
    # Define paths
    dataset_path = '/home/users/li1995/global_flood/UrbanFloods2D/dataset'  # Path to the dataset directory
    output_path = 'flood_data.h5'  # Path where the H5 file will be saved
    city_rainfall_file = '../cities_rainfall.json'  # Path to the JSON file
    
    # Define test cities (you can change this)
    test_cities = ['HOU007', 'LA002', 'NYC002']
    
    # Create H5 dataset
    create_h5_dataset(dataset_path, output_path, city_rainfall_file, test_cities)
    
    print(f"Dataset created at {output_path}")

if __name__ == "__main__":
    main() 
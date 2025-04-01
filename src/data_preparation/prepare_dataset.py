import os
import json
import numpy as np
import rasterio
import h5py
from tqdm import tqdm
import glob
import random
from pathlib import Path

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
            '100-yr': float(city['100-yr'].split(' ')[0]),
            '50-yr': float(city['50-yr'].split(' ')[0]),
            '25-yr': float(city['25-yr'].split(' ')[0]),
            '10-yr': float(city['10-yr'].split(' ')[0])
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
    max_flood_depth = None
    
    for ts in timesteps:
        # Construct filename for this timestep
        flood_filename = f"{city_id}_{rainfall_type}_WaterDepth_{ts}.tif"
        flood_path = os.path.join(city_dir, flood_filename)
        
        # Skip if file doesn't exist
        if not os.path.exists(flood_path):
            print(flood_path)
            continue
        
        # Read the flood depth data
        with rasterio.open(flood_path) as src:
            flood_data = src.read(1)  # Read the first band
            
            # Initialize max_flood_depth if this is the first valid timestep
            if max_flood_depth is None:
                max_flood_depth = flood_data
            else:
                # Update maximum values
                max_flood_depth = np.maximum(max_flood_depth, flood_data)
    
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
    
    return dem_data

def create_h5_dataset(dataset_path, output_path, city_rainfall_file, test_cities=None, val_ratio=0.1):
    """
    Create an H5 dataset containing the DEM, rainfall, and maximum flood depth.
    
    Args:
        dataset_path (str): Path to the dataset directory
        output_path (str): Path where the H5 file will be saved
        city_rainfall_file (str): Path to the JSON file containing rainfall data
        test_cities (list): List of city IDs to be used for testing (will be excluded from training)
        val_ratio (float): Ratio of validation data (taken from non-test cities)
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
    
    # Create H5 file
    with h5py.File(output_path, 'w') as h5f:
        # Create groups for train, val, and test
        train_group = h5f.create_group('train')
        val_group = h5f.create_group('val')
        test_group = h5f.create_group('test')
        
        # Process each city
        city_groups = [
            (train_cities, train_group),
            (val_cities, val_group),
            (test_cities, test_group)
        ]
        
        for cities, group in city_groups:
            for city_id in tqdm(cities, desc=f"Processing {group.name} cities"):
                
                # Create a group for this city
                city_group = group.create_group(city_id)
                

                dem_data = load_dem(dataset_path, city_id)
                city_group.create_dataset('dem', data=dem_data, compression='gzip')

                
                # Process each rainfall type
                for rainfall_type in ['100-yr', '50-yr', '25-yr', '10-yr']:
                    rain_value_str= f'{int(city_rainfall[city_id][rainfall_type])}mm'
                    # Extract max flood depth
                    max_depth = extract_max_flood_depth(dataset_path, city_id, rain_value_str)
                    
                    if max_depth is None:
                        print(f"Warning: No flood data found for {city_id}, {rainfall_type}")
                        continue
                    
                    # Create a group for this rainfall type
                    rainfall_group = city_group.create_group(rainfall_type)
                    
                    # Store rainfall value
                    rainfall_value = city_rainfall[city_id][rainfall_type]
                    rainfall_group.attrs['rainfall_value'] = rainfall_value
                    
                    # Store max flood depth
                    rainfall_group.create_dataset('max_flood_depth', data=max_depth, compression='gzip')
                        


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
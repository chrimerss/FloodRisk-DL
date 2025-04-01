import os
import json
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse
from pathlib import Path
import glob

#make HALO to exclude nan values
HALO=10

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

def get_last_timestep(dataset_path, city_id, rainfall_amount):
    """
    Get the last available timestep for a city and rainfall amount.
    
    Args:
        dataset_path (str): Path to the dataset directory
        city_id (str): City ID
        rainfall_amount (float): Rainfall amount in mm
        
    Returns:
        str: Last timestep string
    """
    city_dir = os.path.join(dataset_path, city_id)
    pattern = f"{city_id}_{rainfall_amount}mm_WaterDepth_*.tif"
    files = glob.glob(os.path.join(city_dir, pattern))
    
    if not files:
        return None
    
    # Extract timesteps from filenames and find the last one
    timesteps = [f.split('_')[-1].split('.')[0] for f in files]
    return max(timesteps)

def visualize_raw_data(dataset_path, city_id, rainfall_type, rainfall_amount, city_rainfall, output_dir="raw_visualizations"):
    """
    Visualize raw DEM, rainfall, and flood depth data for a city.
    
    Args:
        dataset_path (str): Path to the dataset directory
        city_id (str): City ID
        rainfall_type (str): Type of rainfall (e.g., '100-yr')
        rainfall_amount (float): Rainfall amount in mm
        city_rainfall (dict): Dictionary mapping city IDs to rainfall values
        output_dir (str): Directory to save visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    city_dir = os.path.join(dataset_path, city_id)
    
    # Get the last timestep
    last_timestep = get_last_timestep(dataset_path, city_id, rainfall_amount)
    if last_timestep is None:
        print(f"No data found for {city_id}, {rainfall_amount}mm")
        return
    
    # Load DEM data
    dem_path = os.path.join(city_dir, f"{city_id}_DEM.tif")
    with rasterio.open(dem_path) as src:
        dem = src.read(1)[HALO:-HALO, HALO:-HALO]
        dem_transform = src.transform
        dem_crs = src.crs
    
    # Load flood depth data at last timestep
    flood_path = os.path.join(city_dir, f"{city_id}_{rainfall_amount}mm_WaterDepth_{last_timestep}.tif")
    with rasterio.open(flood_path) as src:
        flood_depth = src.read(1)[HALO:-HALO, HALO:-HALO]
        flood_transform = src.transform
        flood_crs = src.crs
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot DEM
    im0 = axs[0].imshow(dem, cmap='terrain')
    axs[0].set_title(f'DEM - {city_id}')
    plt.colorbar(im0, ax=axs[0], label='Elevation (m)')
    
    # Plot flood depth at last timestep
    im1 = axs[1].imshow(flood_depth, cmap='Blues')
    axs[1].set_title(f'Flood Depth at t={last_timestep}\n{rainfall_type}: {rainfall_amount} mm')
    plt.colorbar(im1, ax=axs[1], label='Depth (m)')
    
    # Plot overlay (DEM with flood depth)
    # Create a masked array for flood depth (mask where depth is zero)
    masked_depth = np.ma.masked_where(flood_depth <= 0.01, flood_depth)
    
    # Plot DEM first
    axs[2].imshow(dem, cmap='terrain')
    
    # Add flood depth overlay
    flood_overlay = axs[2].imshow(masked_depth, cmap='Blues', alpha=0.7, 
                                norm=Normalize(vmin=0, vmax=np.max(flood_depth)))
    axs[2].set_title('DEM + Flood Depth Overlay')
    plt.colorbar(flood_overlay, ax=axs[2], label='Depth (m)')
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, f'{city_id}_{rainfall_type}_{rainfall_amount}mm_raw.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_file}")
    
    # Print some statistics
    print(f"\nStatistics for {city_id}, {rainfall_type} ({rainfall_amount}mm):")
    print(f"DEM - Min: {dem.min():.2f}m, Max: {dem.max():.2f}m, Mean: {dem.mean():.2f}m")
    print(f"Flood Depth - Min: {flood_depth.min():.2f}m, Max: {flood_depth.max():.2f}m, Mean: {flood_depth.mean():.2f}m")
    print(f"Last timestep: {last_timestep}")

def main():
    parser = argparse.ArgumentParser(description='Visualize raw flood prediction data')
    parser.add_argument('--dataset_path', type=str, default='/home/users/li1995/global_flood/UrbanFloods2D/dataset', help='Path to the dataset directory')
    parser.add_argument('--city', type=str, default=None, help='City ID to visualize')
    parser.add_argument('--rainfall_type', type=str, default='100-yr', help='Rainfall type to visualize (100-yr, 50-yr, 25-yr, 10-yr)')
    parser.add_argument('--output_dir', type=str, default='raw_visualizations', help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Load rainfall data
    city_rainfall = load_city_rainfall_data('../cities_rainfall.json')
    
    # Get all city IDs from the dataset directory
    all_cities = [d for d in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path, d))]
    
    # If no city is specified, use all cities
    if args.city is None:
        cities = all_cities
    else:
        cities = [args.city]
    
    # If no rainfall type is specified, use all types
    if args.rainfall_type is None:
        rainfall_types = ['100-yr', '50-yr', '25-yr', '10-yr']
    else:
        rainfall_types = [args.rainfall_type]
    
    # Visualize data for each city and rainfall type
    for city_id in cities:
        if city_id not in city_rainfall:
            print(f"Warning: Rainfall data for {city_id} not found in JSON file. Skipping.")
            continue
        
        for rainfall_type in rainfall_types:
            rainfall_amount = int(city_rainfall[city_id][rainfall_type])
            print(f"\nProcessing {city_id}, {rainfall_type} ({rainfall_amount}mm)...")
            visualize_raw_data(args.dataset_path, city_id, rainfall_type, rainfall_amount, city_rainfall, args.output_dir)

if __name__ == "__main__":
    main()
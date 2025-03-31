import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib.colors import Normalize

def explore_h5_structure(h5_file):
    """
    Print the structure of the H5 file to understand its organization.
    
    Args:
        h5_file (str): Path to the H5 file
    """
    def print_attrs(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: Dataset, shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{name}: Group")
            for attr_name, attr_value in obj.attrs.items():
                print(f"  - Attribute: {attr_name} = {attr_value}")
    
    with h5py.File(h5_file, 'r') as f:
        print("H5 File Structure:")
        f.visititems(print_attrs)

def count_samples(h5_file):
    """
    Count the number of samples in each split (train, val, test).
    
    Args:
        h5_file (str): Path to the H5 file
    """
    with h5py.File(h5_file, 'r') as f:
        splits = ['train', 'val', 'test']
        
        for split in splits:
            if split not in f:
                print(f"{split} split not found in the H5 file.")
                continue
                
            num_cities = len(f[split])
            total_samples = 0
            
            for city in f[split]:
                city_group = f[split][city]
                # Count rainfall scenarios (each is a separate sample)
                for rainfall_type in city_group:
                    if rainfall_type != 'dem':  # Skip the DEM dataset
                        total_samples += 1
            
            print(f"{split.capitalize()} set: {num_cities} cities, {total_samples} samples")

def visualize_sample(h5_file, split, city_id, rainfall_type):
    """
    Visualize a sample from the H5 file (DEM, max flood depth, and their overlay).
    
    Args:
        h5_file (str): Path to the H5 file
        split (str): Data split ('train', 'val', 'test')
        city_id (str): City ID
        rainfall_type (str): Type of rainfall ('100-yr', '50-yr', '25-yr', '10-yr')
    """
    with h5py.File(h5_file, 'r') as f:
        # Check if the requested data exists
        if split not in f or city_id not in f[split] or rainfall_type not in f[split][city_id]:
            print(f"Data not found: split={split}, city_id={city_id}, rainfall_type={rainfall_type}")
            return
        
        # Get the data
        dem = f[split][city_id]['dem'][:]
        max_depth = f[split][city_id][rainfall_type]['max_flood_depth'][:]
        rainfall_value = f[split][city_id][rainfall_type].attrs['rainfall_value']
        
        # Create a figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot DEM
        dem_plot = axs[0].imshow(dem, cmap='terrain')
        axs[0].set_title(f'DEM - {city_id}')
        plt.colorbar(dem_plot, ax=axs[0], label='Elevation (m)')
        
        # Plot max flood depth
        max_depth_plot = axs[1].imshow(max_depth, cmap='Blues')
        axs[1].set_title(f'Max Flood Depth - {rainfall_type} ({rainfall_value} mm)')
        plt.colorbar(max_depth_plot, ax=axs[1], label='Depth (m)')
        
        # Plot overlay (DEM with flood depth)
        # Create a masked array for flood depth (mask where depth is zero)
        masked_depth = np.ma.masked_where(max_depth <= 0.01, max_depth)
        
        # Plot DEM first
        axs[2].imshow(dem, cmap='terrain')
        
        # Add flood depth overlay
        flood_overlay = axs[2].imshow(masked_depth, cmap='Blues', alpha=0.7, 
                                    norm=Normalize(vmin=0, vmax=np.max(max_depth)))
        axs[2].set_title('DEM + Flood Depth Overlay')
        plt.colorbar(flood_overlay, ax=axs[2], label='Depth (m)')
        
        plt.tight_layout()
        
        # Create output directory if it doesn't exist
        output_dir = 'visualizations'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the figure
        output_file = os.path.join(output_dir, f'{split}_{city_id}_{rainfall_type}.png')
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        print(f"Visualization saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Explore and visualize the flood prediction dataset')
    parser.add_argument('--h5_file', type=str, default='flood_data.h5', help='Path to the H5 file')
    parser.add_argument('--explore', action='store_true', help='Explore the structure of the H5 file')
    parser.add_argument('--count', action='store_true', help='Count samples in each split')
    parser.add_argument('--visualize', action='store_true', help='Visualize samples')
    parser.add_argument('--split', type=str, default='test', help='Data split to visualize')
    parser.add_argument('--city', type=str, default=None, help='City ID to visualize')
    parser.add_argument('--rainfall', type=str, default='100-yr', help='Rainfall type to visualize')
    
    args = parser.parse_args()
    
    # If no specific action is requested, show all information
    if not (args.explore or args.count or args.visualize):
        args.explore = True
        args.count = True
        args.visualize = True
    
    if args.explore:
        explore_h5_structure(args.h5_file)
        print()
    
    if args.count:
        count_samples(args.h5_file)
        print()
    
    if args.visualize:
        with h5py.File(args.h5_file, 'r') as f:
            # If no city is specified, visualize the first city in the split
            if args.city is None:
                if args.split in f and len(f[args.split]) > 0:
                    args.city = list(f[args.split].keys())[0]
                    print(f"No city specified, using first city: {args.city}")
                else:
                    print(f"No cities found in {args.split} split.")
                    return
            
            visualize_sample(args.h5_file, args.split, args.city, args.rainfall)

if __name__ == "__main__":
    main() 
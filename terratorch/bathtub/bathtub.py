import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import argparse
import time
import rasterio
from rasterio.transform import Affine
import os
from matplotlib.colors import ListedColormap
from sklearn.metrics import jaccard_score
import warnings

# Suppress numpy warnings temporarily
warnings.filterwarnings('ignore', category=RuntimeWarning)

def simple_bathtub_with_rainfall_robust(dem, rainfall, buffer=10):
    """
    More robust bathtub model implementation with additional error checking
    and handling of numerical edge cases.
    
    Parameters:
    -----------
    dem : numpy.ndarray
        Digital Elevation Model (DEM) as a 2D array
    rainfall : numpy.ndarray
        2D rainfall field with the same shape as the DEM
    buffer : int
        Buffer size to apply from the edges
        
    Returns:
    --------
    water_depth : numpy.ndarray
        Water depth at each cell after redistribution
    """
    # Apply buffer to avoid boundary issues
    if buffer > 0:
        print(f"Applying buffer of {buffer} cells to avoid boundary issues")
        
        # Get buffer dimensions
        rows, cols = dem.shape
        if buffer*2 >= rows or buffer*2 >= cols:
            print("Warning: Buffer size too large for DEM dimensions. Reducing buffer.")
            buffer = min(rows, cols) // 10  # Use 10% of smallest dimension
            print(f"Reduced buffer to {buffer} cells")
        
        # Apply buffer
        dem_buffered = dem[buffer:-buffer, buffer:-buffer]
        rainfall_buffered = rainfall[buffer:-buffer, buffer:-buffer]
    else:
        dem_buffered = dem.copy()
        rainfall_buffered = rainfall.copy()
    
    # Check that dimensions match
    if dem_buffered.shape != rainfall_buffered.shape:
        raise ValueError("DEM and rainfall must have the same shape")
    
    # Make sure we're working with floating point numbers
    dem_buffered = dem_buffered.astype(np.float64)
    rainfall_buffered = rainfall_buffered.astype(np.float64)
    
    # Replace any potential NaN or inf values in DEM and rainfall
    if np.any(~np.isfinite(dem_buffered)):
        print(f"Warning: Found {np.sum(~np.isfinite(dem_buffered))} non-finite values in DEM. Replacing with nearby values.")
        dem_buffered = np.nan_to_num(dem_buffered, nan=np.nanmean(dem_buffered), 
                               posinf=np.nanmax(dem_buffered), neginf=np.nanmin(dem_buffered))
    
    if np.any(~np.isfinite(rainfall_buffered)):
        print(f"Warning: Found {np.sum(~np.isfinite(rainfall_buffered))} non-finite values in rainfall. Replacing with zeros.")
        rainfall_buffered = np.nan_to_num(rainfall_buffered, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Calculate total water volume
    total_water_volume = np.sum(rainfall_buffered)
    print(f"Total rainfall volume: {total_water_volume:.2f} units")
    
    # Check if we have any water
    if total_water_volume <= 0:
        print("No water to distribute. Returning zero depth.")
        # Initialize full-sized output array with zeros
        water_depth = np.zeros_like(dem)
        if buffer > 0:
            # Only the buffered region will have been processed
            water_depth[buffer:-buffer, buffer:-buffer] = 0
        return water_depth
    
    # Label contiguous regions (watersheds)
    watersheds, n_watersheds = label(np.ones_like(dem_buffered))
    print(f"Identified {n_watersheds} independent watersheds")
    
    # Initialize water depth for buffered region
    water_depth_buffered = np.zeros_like(dem_buffered, dtype=np.float64)
    
    # Process each watershed independently
    for region in range(1, n_watersheds + 1):
        # Get cells in this watershed
        mask = watersheds == region
        
        # Get total rainfall for this region
        region_rainfall = rainfall_buffered[mask]
        region_volume = np.sum(region_rainfall)
        
        if region_volume <= 0:
            continue
        
        # Get elevations in this region
        region_dem = dem_buffered[mask]
        
        # Make sure we have valid data in this region
        if np.all(~np.isfinite(region_dem)):
            print(f"Warning: Region {region} has no valid elevation data. Skipping.")
            continue
        
        # Sort elevations
        sorted_elevations = np.sort(region_dem[np.isfinite(region_dem)])
        
        if len(sorted_elevations) == 0:
            print(f"Warning: Region {region} has no valid elevation data after filtering. Skipping.")
            continue
        
        # Binary search for water level
        min_level = sorted_elevations[0]
        max_level = sorted_elevations[-1] + region_volume  # Max possible water level
        
        # Tolerance for binary search
        tolerance = 0.0001
        max_iterations = 50
        
        for iteration in range(max_iterations):
            # Try the middle level
            water_level = (min_level + max_level) / 2
            
            # Calculate water depths at this level (with safety checking)
            depths = np.zeros_like(region_dem)
            valid_cells = np.isfinite(region_dem)
            depths[valid_cells] = np.maximum(0, water_level - region_dem[valid_cells])
            
            # Calculate volume at this level
            volume_at_level = np.sum(depths)
            
            # Check if we're within tolerance
            if abs(volume_at_level - region_volume) < tolerance:
                break
            
            # Adjust bounds for next iteration
            if volume_at_level > region_volume:
                max_level = water_level
            else:
                min_level = water_level
        
        # Apply the final water depth to this region (with safety check)
        final_depth = np.zeros_like(mask, dtype=np.float64)
        valid_cells = mask & np.isfinite(dem_buffered)
        final_depth[valid_cells] = np.maximum(0, water_level - dem_buffered[valid_cells])
        
        # Add to overall depth map
        water_depth_buffered += final_depth
    
    # Final check for any remaining NaN or inf values
    if np.any(~np.isfinite(water_depth_buffered)):
        print(f"Warning: Found {np.sum(~np.isfinite(water_depth_buffered))} non-finite values in result. Replacing with zeros.")
        water_depth_buffered = np.nan_to_num(water_depth_buffered, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Initialize full-sized output array with zeros
    water_depth = np.zeros_like(dem)
    if buffer > 0:
        # Only the buffered region will have been processed
        water_depth[buffer:-buffer, buffer:-buffer] = water_depth_buffered
    else:
        water_depth = water_depth_buffered
        
    return water_depth

def identify_flood_zones(water_depth, min_depth=0.1):
    """
    Identify distinct flood zones from water depth map.
    
    Parameters:
    -----------
    water_depth : numpy.ndarray
        Water depth at each cell
    min_depth : float
        Minimum depth to consider as flooded
        
    Returns:
    --------
    labeled_zones : numpy.ndarray
        Array with labeled flood zones
    n_zones : int
        Number of distinct flood zones
    """
    # Handle any NaN or inf values
    water_depth = np.nan_to_num(water_depth, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create a binary map of flooded areas
    flooded = (water_depth >= min_depth).astype(int)
    
    # Label connected components (flood zones)
    labeled_zones, n_zones = label(flooded)
    
    return labeled_zones, n_zones

def categorize_depths(depth_array):
    """
    Categorize water depths into five categories:
    0: <0.1m (no significant flooding)
    1: 0.1-0.2m (minor flooding)
    2: 0.2-0.5m (moderate flooding)
    3: 0.5-1.0m (major flooding)
    4: ≥1.0m (severe flooding)
    
    Parameters:
    -----------
    depth_array : numpy.ndarray
        Water depth array in meters
        
    Returns:
    --------
    categories : numpy.ndarray
        Categorized depths (0-4)
    """
    # Handle any NaN or inf values
    depth_array = np.nan_to_num(depth_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    categories = np.zeros_like(depth_array, dtype=np.int64)
    categories = np.where(depth_array >= 0.1, 1, categories)
    categories = np.where(depth_array >= 0.2, 2, categories)
    categories = np.where(depth_array >= 0.5, 3, categories)
    categories = np.where(depth_array >= 1.0, 4, categories)
    return categories

def calculate_jaccard_index(ground_truth, prediction, valid_mask=None):
    """
    Calculate Jaccard index (intersection over union) for each category.
    
    Parameters:
    -----------
    ground_truth : numpy.ndarray
        Ground truth categorized depths
    prediction : numpy.ndarray
        Predicted categorized depths
    valid_mask : numpy.ndarray, optional
        Boolean mask of valid data (True where valid)
        
    Returns:
    --------
    jaccard_scores : dict
        Dictionary with Jaccard scores for each category
    """
    if valid_mask is None:
        valid_mask = np.ones_like(ground_truth, dtype=bool)
    
    # Flatten arrays and apply mask
    gt_flat = ground_truth[valid_mask].flatten()
    pred_flat = prediction[valid_mask].flatten()
    
    # Calculate Jaccard index for each category
    jaccard_scores = {}
    
    # Overall binary flooding (any flooding vs no flooding)
    binary_gt = gt_flat > 0
    binary_pred = pred_flat > 0
    
    # Avoid division by zero
    if np.sum(binary_gt | binary_pred) > 0:
        jaccard_scores['binary'] = np.sum(binary_gt & binary_pred) / np.sum(binary_gt | binary_pred)
    else:
        jaccard_scores['binary'] = 1.0 if np.sum(binary_gt) == 0 and np.sum(binary_pred) == 0 else 0.0
    
    # For each flooding category
    category_names = ['No flooding (<0.1m)', 'Minor (0.1-0.2m)', 'Moderate (0.2-0.5m)', 
                      'Major (0.5-1.0m)', 'Severe (≥1.0m)']
    
    for category in range(5):
        gt_category = gt_flat == category
        pred_category = pred_flat == category
        
        # If either array has no instances of this category, set IoU to 0
        if not np.any(gt_category) and not np.any(pred_category):
            jaccard_scores[category_names[category]] = 1.0  # Perfect agreement on absence
        elif not np.any(gt_category) or not np.any(pred_category):
            jaccard_scores[category_names[category]] = 0.0  # No overlap
        else:
            # Calculate Jaccard index (intersection over union)
            intersection = np.sum(gt_category & pred_category)
            union = np.sum(gt_category | pred_category)
            
            if union > 0:
                jaccard_scores[category_names[category]] = intersection / union
            else:
                jaccard_scores[category_names[category]] = 0.0
    
    # Calculate weighted average Jaccard score
    weights = [np.sum(gt_flat == i) for i in range(5)]
    total_weight = sum(weights)
    
    if total_weight > 0:
        weighted_avg = sum(jaccard_scores[category_names[i]] * weights[i] for i in range(5)) / total_weight
    else:
        weighted_avg = 0.0
        
    jaccard_scores['Weighted Average'] = weighted_avg
    
    return jaccard_scores

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Bathtub model for flood inundation with rainfall')
    parser.add_argument('dem_file', help='DEM file in GeoTIFF format')
    parser.add_argument('rainfall', type=float, help='Rainfall amount in mm')
    parser.add_argument('--ground-truth', '-g', help='Ground truth water depth file in GeoTIFF format')
    parser.add_argument('--output', '-o', default='flood_depth.tif', help='Output flood depth file name')
    parser.add_argument('--plot', '-p', action='store_true', help='Show plot of results')
    parser.add_argument('--min-depth', '-m', type=float, default=0.1, help='Minimum depth (mm) to consider as flooded')
    parser.add_argument('--buffer', '-b', type=int, default=10, help='Buffer size to apply from the edges')
    
    args = parser.parse_args()
    
    # Convert rainfall from mm to m for calculations
    rainfall_m = args.rainfall / 1000.0
    
    print(f"Processing DEM: {args.dem_file}")
    print(f"Rainfall amount: {args.rainfall} mm ({rainfall_m} m)")
    print(f"Using buffer size: {args.buffer} cells")
    
    # Read DEM from GeoTIFF
    with rasterio.open(args.dem_file) as src:
        dem = src.read(1)
        metadata = src.meta
        
        # Check for and handle nodata values
        nodata = src.nodata
        if nodata is not None:
            # Create a valid data mask (True where data is valid)
            valid_mask = dem != nodata
            
            # Replace nodata with a high value (will not be flooded)
            max_valid = np.nanmax(dem[valid_mask])
            dem[~valid_mask] = max_valid + 1000  # Much higher than any valid elevation
            
            print(f"Handled nodata values (value: {nodata})")
        else:
            valid_mask = np.ones_like(dem, dtype=bool)
        
        # Check DEM size
        print(f"DEM dimensions: {dem.shape[0]} rows × {dem.shape[1]} columns")
        
        # Create uniform rainfall field (same shape as DEM)
        rainfall = np.ones_like(dem) * rainfall_m
        
        # Apply the rainfall only to valid DEM cells
        rainfall[~valid_mask] = 0
        
        # Start timer
        start_time = time.time()
        
        # Run the bathtub model with buffer
        try:
            water_depth = simple_bathtub_with_rainfall_robust(dem, rainfall, buffer=args.buffer)
        except Exception as e:
            print(f"Error in bathtub model: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        # End timer
        elapsed_time = time.time() - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds")
        
        # Convert water depth back to mm for output
        water_depth_mm = water_depth * 1000
        
        # Identify flood zones
        labeled_zones, n_zones = identify_flood_zones(water_depth, min_depth=args.min_depth / 1000)
        
        # Print summary statistics
        print("\nResults Summary:")
        print(f"Total rainfall volume: {np.sum(rainfall):.4f} m³")
        print(f"Final water volume: {np.sum(water_depth):.4f} m³")
        print(f"Maximum water depth: {np.nanmax(water_depth) * 1000:.2f} mm")
        print(f"Flooded area: {np.sum(water_depth > args.min_depth / 1000)} cells")
        print(f"Number of distinct flood zones: {n_zones}")
        
        # Categorize flood depths
        depth_categories = categorize_depths(water_depth)
        
        # Ground truth comparison if provided
        ground_truth_data = None
        ground_truth_categories = None
        
        if args.ground_truth:
            print(f"\nComparing with ground truth: {args.ground_truth}")
            try:
                with rasterio.open(args.ground_truth) as gt_src:
                    # Read ground truth data
                    ground_truth_data = gt_src.read(1)
                    
                    # Apply the same buffer to ground truth
                    if args.buffer > 0:
                        print(f"Applying {args.buffer} cell buffer to ground truth data")
                        # Create a full zero array for the output
                        ground_truth_buffered = ground_truth_data[args.buffer:-args.buffer, args.buffer:-args.buffer]
                        ground_truth_valid = np.ones_like(ground_truth_buffered, dtype=bool)
                        
                        # Check if ground truth is in mm or m
                        gt_max = np.nanmax(ground_truth_buffered[ground_truth_valid])
                        
                        if gt_max > 100:
                            print("Ground truth appears to be in mm. Converting to meters.")
                            ground_truth_buffered = ground_truth_buffered / 1000  # Convert mm to m
                        
                        # Create a full-sized array but with zeros outside buffer
                        ground_truth_data_full = np.zeros_like(ground_truth_data)
                        ground_truth_data_full[args.buffer:-args.buffer, args.buffer:-args.buffer] = ground_truth_buffered
                        ground_truth_data = ground_truth_data_full
                    else:
                        # Check if ground truth is in mm or m
                        gt_max = np.nanmax(ground_truth_data)
                        
                        if gt_max > 100:
                            print("Ground truth appears to be in mm. Converting to meters.")
                            ground_truth_data = ground_truth_data / 1000  # Convert mm to m
                    
                    # Create valid data mask for ground truth
                    gt_nodata = gt_src.nodata
                    if gt_nodata is not None:
                        gt_valid_mask = ground_truth_data != gt_nodata
                    else:
                        gt_valid_mask = np.isfinite(ground_truth_data)
                    
                    combined_mask = valid_mask & gt_valid_mask
                    
                    # Replace any NaN with zeros
                    ground_truth_data = np.nan_to_num(ground_truth_data, nan=0.0)
                    
                    # Categorize ground truth depths
                    ground_truth_categories = categorize_depths(ground_truth_data)
                    
                    # Calculate Jaccard index
                    jaccard_scores = calculate_jaccard_index(
                        ground_truth_categories, 
                        depth_categories,
                        combined_mask
                    )
                    
                    # Print Jaccard scores
                    print("\nJaccard Index (Intersection over Union):")
                    print(f"Binary flooding: {jaccard_scores['binary']:.4f}")
                    
                    category_names = ['No flooding (<0.1m)', 'Nuisance (0.1-0.2m)', 'Minor (0.2-0.5m)', 
                                      'Moderate (0.5-1.0m)', 'major (≥1.0m)']
                    
                    for category in category_names:
                        print(f"{category}: {jaccard_scores[category]:.4f}")
                    
                    print(f"Weighted Average: {jaccard_scores['Weighted Average']:.4f}")
                    
            except Exception as e:
                print(f"Error processing ground truth: {str(e)}")
                import traceback
                traceback.print_exc()
                ground_truth_data = None
        
        # Save results to GeoTIFF
        output_meta = metadata.copy()
        output_meta.update(
            dtype=rasterio.float32,
            count=1,
            compress='lzw',
            nodata=-9999
        )
        
        # Set nodata values in the output
        water_depth_mm[~valid_mask] = -9999
        
        # Save the water depth raster
        output_file = args.output
        with rasterio.open(output_file, 'w', **output_meta) as dst:
            dst.write(water_depth_mm.astype(rasterio.float32), 1)
        
        print(f"Saved flood depth to: {output_file}")
        
        # # Save flood zones as a separate GeoTIFF
        # zones_file = os.path.splitext(args.output)[0] + '_zones.tif'
        # zones_meta = metadata.copy()
        # zones_meta.update(
        #     dtype=rasterio.int16,
        #     count=1,
        #     compress='lzw',
        #     nodata=-9999
        # )
        
        # # Set nodata values in the output
        # labeled_zones[~valid_mask] = -9999
        
        # with rasterio.open(zones_file, 'w', **zones_meta) as dst:
        #     dst.write(labeled_zones.astype(rasterio.int16), 1)
        
        # print(f"Saved flood zones to: {zones_file}")
        
        # # Save categorized depths
        # categories_file = os.path.splitext(args.output)[0] + '_categories.tif'
        # categories_meta = metadata.copy()
        # categories_meta.update(
        #     dtype=rasterio.int8,
        #     count=1,
        #     compress='lzw',
        #     nodata=-9999
        # )
        
        # # Set nodata values in the output
        # depth_categories_out = depth_categories.copy()
        # depth_categories_out[~valid_mask] = -9999
        
        # with rasterio.open(categories_file, 'w', **categories_meta) as dst:
        #     dst.write(depth_categories_out.astype(rasterio.int8), 1)
        
        # print(f"Saved categorized depths to: {categories_file}")
        
        # Show plot if requested
        if args.plot:
            # Define color maps for flood categories
            colors = ['#FFFFFF', '#ADD8E6', '#6495ED', '#0000FF', '#00008B']
            cmap = ListedColormap(colors)
            
            if ground_truth_data is not None:
                # Create a figure with both ground truth and model results
                plt.figure(figsize=(15, 12))
                
                # Plot DEM
                plt.subplot(231)
                plt.imshow(dem, cmap='terrain')
                plt.colorbar(label='Elevation (m)')
                plt.title('Digital Elevation Model')
                
                # Plot ground truth water depth
                plt.subplot(232)
                plt.imshow(ground_truth_data, cmap='Blues')
                plt.colorbar(label='Water Depth (m)')
                plt.title('Ground Truth Water Depth')
                
                # Plot model water depth
                plt.subplot(233)
                plt.imshow(water_depth, cmap='Blues')
                plt.colorbar(label='Water Depth (m)')
                plt.title('Bathtub Model Water Depth')
                
                # Plot ground truth categories
                plt.subplot(234)
                plt.imshow(ground_truth_categories, cmap=cmap, vmin=0, vmax=4)
                plt.colorbar(label='Flood Category', ticks=[0, 1, 2, 3, 4])
                plt.title('Ground Truth Flood Categories')
                
                # Plot model categories
                plt.subplot(235)
                plt.imshow(depth_categories, cmap=cmap, vmin=0, vmax=4)
                plt.colorbar(label='Flood Category', ticks=[0, 1, 2, 3, 4])
                plt.title('Bathtub Model Flood Categories')
                
                # Plot difference in categories
                category_diff = depth_categories - ground_truth_categories
                diff_cmap = plt.cm.RdBu_r
                plt.subplot(236)
                plt.imshow(category_diff, cmap=diff_cmap, vmin=-4, vmax=4)
                plt.colorbar(label='Category Difference\n(Model - Ground Truth)')
                plt.title('Difference in Flood Categories')
                
                # Add a text box with Jaccard scores
                plt.figtext(0.5, 0.01, 
                            f"Binary Jaccard: {jaccard_scores['binary']:.4f} | "
                            f"Weighted Avg: {jaccard_scores['Weighted Average']:.4f}",
                            ha='center', fontsize=12, 
                            bbox=dict(facecolor='white', alpha=0.8))
                
            else:
                # Create a figure with only model results
                plt.figure(figsize=(15, 10))
                
                # Plot DEM
                plt.subplot(221)
                plt.imshow(dem, cmap='terrain')
                plt.colorbar(label='Elevation (m)')
                plt.title('Digital Elevation Model')
                
                # Plot rainfall
                plt.subplot(222)
                plt.imshow(rainfall, cmap='Blues')
                plt.colorbar(label='Rainfall (m)')
                plt.title(f'Rainfall Distribution ({args.rainfall} mm)')
                
                # Plot water depth
                plt.subplot(223)
                plt.imshow(water_depth * 1000, cmap='Blues')
                plt.colorbar(label='Water Depth (mm)')
                plt.title('Flood Inundation Depth')
                
                # Plot flood categories
                plt.subplot(224)
                plt.imshow(depth_categories, cmap=cmap, vmin=0, vmax=4)
                plt.colorbar(label='Flood Category', ticks=[0, 1, 2, 3, 4])
                plt.title('Flood Categories')
            
            plt.tight_layout()
            
            # Add legend for the flood categories
            category_labels = ['None (<0.1m)', 'Minor (0.1-0.2m)', 'Moderate (0.2-0.5m)', 
                              'Major (0.5-1.0m)', 'Severe (≥1.0m)']
            plt.figlegend([plt.Rectangle((0,0),1,1, color=c) for c in colors], 
                          category_labels, 
                          loc='lower center', 
                          ncol=5, 
                          bbox_to_anchor=(0.5, 0.02))
            
            plt.subplots_adjust(bottom=0.1)
            
            # Save the plot
            plot_file = os.path.splitext(args.output)[0] + '_plot.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to: {plot_file}")
            
            plt.show()

if __name__ == "__main__":
    main()
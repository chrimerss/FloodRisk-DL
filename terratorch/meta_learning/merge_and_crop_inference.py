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

# =====================
# CONFIGURATION SECTION
# =====================
# Set your data directory (where dem_data_* folders are)
DATA_DIR = "/home/users/li1995/global_flood/FloodRisk-DL/data/download_US_DEM"  # Change to your base directory if needed
# Set your county shapefile path
COUNTY_SHAPEFILE = "/home/users/li1995/global_flood/FloodRisk-DL/data/download_US_DEM/national_map_downloader/cb_2018_us_county_20m.shp"
# Set your output directory for merged/cropped results
OUTPUT_DIR = "merged_results"


def merge_and_crop_county(county_dir, shapefile, output_dir):
    """Merge inference results for a county and crop to county boundary."""
    tif_files = sorted(glob.glob(os.path.join(county_dir, "inference_*.tif")))
    if not tif_files:
        print(f"No inference GeoTIFFs found in {county_dir}")
        return

    print(f"Processing {len(tif_files)} inference files in {county_dir}")

    # Extract county name and state from directory name first
    county_name = os.path.basename(county_dir)
    print(f"County directory name: {county_name}")
    
    # Parse county name - handle formats like "dem_data_Baltimore_MD" or "dem_data_Harris_TX"
    parts = county_name.split('_')
    if len(parts) >= 3:
        county_part = parts[2]  # "Baltimore"
        state_part = parts[3] if len(parts) > 3 else None  # "MD"
    else:
        county_part = county_name.split('_')[-1]
        state_part = None
    
    print(f"Looking for county: {county_part}, state: {state_part}")

    # Read county shapefile and find matching county first
    gdf = gpd.read_file(shapefile)
    print(f"Shapefile CRS: {gdf.crs}")
    
    # Find matching county
    county_row = None
    
    # Try different matching strategies
    if state_part:
        # Try exact match with state
        county_row = gdf[(gdf['NAME'].str.contains(county_part, case=False, na=False)) & 
                        (gdf['STATEFP'].str.contains(state_part, case=False, na=False) if 'STATEFP' in gdf.columns else False)]
        if county_row.empty and 'STUSPS' in gdf.columns:
            county_row = gdf[(gdf['NAME'].str.contains(county_part, case=False, na=False)) & 
                            (gdf['STUSPS'] == state_part)]
    
    # Fallback to name-only match
    if county_row is None or county_row.empty:
        county_row = gdf[gdf['NAME'].str.contains(county_part, case=False, na=False)]
    
    if county_row.empty:
        print(f"County boundary not found for {county_part}")
        print(f"Available counties: {gdf['NAME'].head(10).tolist()}")
        return
    
    if len(county_row) > 1:
        print(f"Multiple counties found: {county_row['NAME'].tolist()}")
        county_row = county_row.iloc[0:1]  # Take the first one
    
    county_geom = county_row.geometry.iloc[0]
    county_name_found = county_row['NAME'].iloc[0]
    print(f"Found county: {county_name_found}")

    # Merge all rasters with memory optimization
    print("Merging rasters...")
    src_files_to_mosaic = []
    target_crs = None
    
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
        
        # Determine target CRS (use the most common one, or first one)
        target_crs = max(crs_info.keys(), key=lambda x: len(crs_info[x]))
        print(f"Using target CRS: {target_crs}")
        
        # Second pass: open files and reproject if needed
        temp_files = []  # Keep track of temporary reprojected files
        
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
        
        # Use method='first' and limit memory usage
        mosaic, out_trans = merge(src_files_to_mosaic, method='first')
        out_meta = src_files_to_mosaic[0].meta.copy()
        
        # Get CRS from the raster (should be target_crs)
        raster_crs = target_crs
        print(f"Raster CRS: {raster_crs}")
        
        # Get raster bounds
        raster_bounds = rasterio.transform.array_bounds(mosaic.shape[1], mosaic.shape[2], out_trans)
        print(f"Raster bounds: {raster_bounds}")
        
    finally:
        # Close all opened files immediately to free memory
        for src in src_files_to_mosaic:
            src.close()
        src_files_to_mosaic.clear()
        
        # Clean up temporary reprojected files
        for temp_fp in temp_files:
            if os.path.exists(temp_fp):
                os.remove(temp_fp)
                print(f"Cleaned up temporary file: {temp_fp}")
        
        gc.collect()  # Force garbage collection

    # Reproject county geometry to match raster CRS if needed
    original_county_geom = county_geom
    if gdf.crs != raster_crs:
        print(f"Reprojecting county geometry from {gdf.crs} to {raster_crs}")
        county_geom = gpd.GeoSeries([county_geom], crs=gdf.crs).to_crs(raster_crs).iloc[0]
        print(f"Original county bounds: {original_county_geom.bounds}")
        print(f"Reprojected county bounds: {county_geom.bounds}")
    else:
        print(f"County geometry and raster already in same CRS: {raster_crs}")
    
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
        "compress": "lzw"
    })

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save merged raster to temporary file first, then crop
    temp_raster_path = os.path.join(output_dir, f"temp_merged_{county_part}.tif")
    try:
        print("Saving temporary merged raster...")
        with rasterio.open(temp_raster_path, "w", **out_meta) as temp_dst:
            temp_dst.write(mosaic)
        
        # Clear mosaic from memory
        del mosaic
        gc.collect()
        
        print("Cropping to county boundary...")
        # Crop (mask) the merged raster to the county boundary
        with rasterio.open(temp_raster_path) as src:
            print(f"Source raster CRS: {src.crs}")
            print(f"Source raster bounds: {src.bounds}")
            print(f"Source raster transform: {src.transform}")
            
            try:
                out_image, out_transform = mask(
                    src,
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
                    src,
                    [county_geom],
                    crop=False,
                    nodata=255
                )
                print(f"Masked without cropping")
                print(f"Output raster shape: {out_image.shape}")
                print(f"Output transform: {out_transform}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_raster_path):
            os.remove(temp_raster_path)

    # Update metadata for output
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # Save the cropped raster
    out_tif = os.path.join(output_dir, f"{county_name}_merged_cropped.tif")
    print("Saving final cropped raster...")
    with rasterio.open(out_tif, "w", **out_meta) as dest:
        dest.write(out_image)

    # Create and save visualization with reduced memory usage
    print("Creating visualization...")
    out_png = os.path.join(output_dir, f"{county_name}_merged_cropped.png")
    
    # Get the correct bounds from the cropped raster transform BEFORE any downsampling
    original_height, original_width = out_image.shape[1], out_image.shape[2]
    left, bottom, right, top = rasterio.transform.array_bounds(
        original_height, original_width, out_transform
    )
    
    print(f"Original raster size: {original_height}x{original_width}")
    print(f"Geographic extent: left={left}, right={right}, bottom={bottom}, top={top}")
    print(f"County geometry bounds: {county_geom.bounds}")
    
    # Downsample for visualization if image is very large
    vis_data = out_image[0].copy()
    max_dim = 2048  # Maximum dimension for visualization
    downsample_factor = 1
    
    if vis_data.shape[0] > max_dim or vis_data.shape[1] > max_dim:
        # Calculate downsampling factor
        downsample_factor = max(vis_data.shape[0] // max_dim, vis_data.shape[1] // max_dim, 1)
        if downsample_factor > 1:
            print(f"Downsampling visualization by factor {downsample_factor}")
            vis_data = vis_data[::downsample_factor, ::downsample_factor]
            print(f"Downsampled size: {vis_data.shape}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create a custom colormap for flood categories
    colors = ['blue', 'lightblue', 'yellow', 'orange', 'red', 'white']  # 0-4 + nodata
    cmap = mcolors.ListedColormap(colors[:-1])
    cmap.set_bad(colors[-1])  # nodata color
    
    # Mask nodata values
    plot_data = vis_data.astype(float)
    plot_data[plot_data == 255] = float('nan')
    
    # Use the ORIGINAL geographic extent regardless of downsampling
    # The extent should always be the full geographic bounds of the cropped raster
    print(f"Using extent for plot: left={left}, right={right}, bottom={bottom}, top={top}")
    
    # Ensure county geometry is in the same CRS as the output raster
    county_geom_plot = county_geom
    
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
        if hasattr(county_geom_plot, 'exterior'):
            # Single polygon
            x, y = county_geom_plot.exterior.xy
            ax.plot(x, y, color='red', linewidth=3, label='County Boundary')
            print(f"Plotted single polygon boundary with {len(x)} points")
            print(f"County boundary X range: {min(x)} to {max(x)}")
            print(f"County boundary Y range: {min(y)} to {max(y)}")
        elif hasattr(county_geom_plot, 'geoms'):
            # MultiPolygon
            for i, geom in enumerate(county_geom_plot.geoms):
                if hasattr(geom, 'exterior'):
                    x, y = geom.exterior.xy
                    ax.plot(x, y, color='red', linewidth=3)
                    print(f"Plotted polygon {i} with {len(x)} points")
            ax.plot([], [], color='red', linewidth=3, label='County Boundary')  # For legend
        else:
            print(f"Unknown geometry type: {type(county_geom_plot)}")
    except Exception as e:
        print(f"Error plotting county boundary: {e}")
    
    # Set the axis limits to the county bounds to ensure proper zoom
    ax.set_xlim(county_bounds[0], county_bounds[2])
    ax.set_ylim(county_bounds[1], county_bounds[3])
    
    ax.set_title(f"{county_name_found} - Merged & Cropped Flood Inference", fontsize=14)
    ax.set_xlabel('Easting (m)')
    ax.set_ylabel('Northing (m)')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Flood Category (0-4)', shrink=0.8)
    
    # Add legend for county boundary
    ax.legend(loc='upper right')
    
    # Set aspect ratio to equal for proper geographic display
    ax.set_aspect('equal')
    
    plt.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Clear variables to free memory
    del out_image, vis_data, plot_data
    gc.collect()
    
    print(f"Saved merged and cropped raster to {out_tif}")
    print(f"Saved plot to {out_png}")
    print("-" * 50)


if __name__ == "__main__":
    print(f"Looking for county directories in: {DATA_DIR}")
    county_dirs = sorted(glob.glob(os.path.join(DATA_DIR, "dem_data_*")))
    print(f"Found {len(county_dirs)} county directories")
    
    for county_dir in county_dirs:
        try:
            merge_and_crop_county(county_dir, COUNTY_SHAPEFILE, OUTPUT_DIR)
        except Exception as e:
            print(f"Error processing {county_dir}: {e}")
            import traceback
            traceback.print_exc()
            continue 
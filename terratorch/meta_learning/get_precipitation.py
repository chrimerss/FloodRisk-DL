import sys
import rasterio
import pandas as pd
import pfdf.data.noaa.atlas14 as atlas14
import os
import tempfile

def get_dem_center_coords(dem_file):
    """Extract center coordinates from DEM file and transform to EPSG:4326"""
    from pyproj import Transformer
    
    with rasterio.open(dem_file) as src:
        bounds = src.bounds
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        # Get the CRS of the DEM file
        dem_crs = src.crs
        
        # Transform to EPSG:4326 (WGS84 lat/lon)
        if dem_crs != 'EPSG:4326':
            transformer = Transformer.from_crs(dem_crs, 'EPSG:4326', always_xy=True)
            center_lon, center_lat = transformer.transform(center_x, center_y)
        else:
            center_lon, center_lat = center_x, center_y

            
        return center_lat, center_lon

def get_500yr_1hr_rainfall(lat, lon):
    """Get 1-hour, 500-year precipitation from NOAA Atlas 14"""
    try:
        # Create temporary directory for the download
        # with tempfile.TemporaryDirectory() as temp_dir:
            # Download precipitation data
        csv_path = atlas14.download(
            lat, lon, 
            statistic='mean',
            overwrite=True,
            data='depth',  # precipitation depth in mm
            series='pds',  # partial duration series
            units='metric',
            timeout=30
        )
        
        # Read the raw file to parse the custom NOAA format
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        # Find the header line with return periods
        header_line = None
        data_start_idx = None
        
        for i, line in enumerate(lines):
            if line.startswith('by duration for ARI (years):'):
                header_line = line.strip()
                data_start_idx = i + 1
                break
        
        if header_line is None or data_start_idx is None:
            print("Could not find precipitation data header")
            return 90.0
        
        # Parse the header to find the 500-year column index
        header_parts = [x.strip() for x in header_line.split(',')]
        try:
            # Find index of '500' in the header (should be 9th position: 1,2,5,10,25,50,100,200,500,1000)
            return_periods = header_parts[1:]  # Skip "by duration for ARI (years):"
            col_500_idx = return_periods.index('500') + 1  # +1 because first column is duration
        except ValueError:
            print("Could not find 500-year return period in header")
            return 90.0
        
        # Find the 60-min (1-hour) row
        for i in range(data_start_idx, len(lines)):
            line = lines[i].strip()
            if line.startswith('60-min:'):
                # Parse the data line
                data_parts = [x.strip() for x in line.split(',')]
                if len(data_parts) > col_500_idx:
                    try:
                        rainfall_value = float(data_parts[col_500_idx])
                        # print(f"Found 1-hour, 500-year precipitation: {rainfall_value}mm")
                        return rainfall_value
                    except ValueError:
                        print(f"Could not parse rainfall value: {data_parts[col_500_idx]}")
                        return 90.0
                else:
                    print(f"Not enough columns in data row: {len(data_parts)} vs expected {col_500_idx + 1}")
                    return 90.0
            
        print("Could not find 60-min row in precipitation data")
        return 90.0
            
    except Exception as e:
        print(f"Error getting precipitation data: {e}")
        return 90.0  # Fallback value


if __name__ == "__main__":
    dem_file = sys.argv[1]
    lat, lon = get_dem_center_coords(dem_file)
    rainfall = get_500yr_1hr_rainfall(lat, lon)
    print(f"{rainfall:.1f}")

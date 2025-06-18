import sys
import geopandas as gpd
import pandas as pd
import pfdf.data.noaa.atlas14 as atlas14
import os
import tempfile
import xarray as xr

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

def parse_zarr_filename(zarr_file):
    """Parse Zarr filename to extract county GEOID"""
    # Expected format: County_Name_ST_geoid_12345_cropped_dem.zarr
    basename = os.path.basename(zarr_file)
    if '_geoid_' in basename and basename.endswith('_cropped_dem.zarr'):
        parts = basename.split('_geoid_')
        if len(parts) == 2:
            geoid = parts[1].replace('_cropped_dem.zarr', '')
            return geoid
    return None

def get_county_centroid_from_geoid(geoid, shapefile_path):
    """Get county centroid coordinates from GEOID using shapefile"""
    try:
        # print(f"Looking up county with GEOID: {geoid}")
        
        # Load county shapefile
        counties_gdf = gpd.read_file(shapefile_path)
        
        # Add state abbreviations
        state_fips_to_abbrev = get_state_fips_mapping()
        counties_gdf['STATE_ABBREV'] = counties_gdf['STATEFP'].map(state_fips_to_abbrev)
        
        # Find county by GEOID
        county_row = counties_gdf[counties_gdf['GEOID'] == geoid]
        
        if county_row.empty:
            # print(f"County with GEOID {geoid} not found in shapefile")
            return None, None, None, None
        
        # Get county info
        county_name = county_row.iloc[0]['NAME']
        state_abbrev = county_row.iloc[0]['STATE_ABBREV']
        county_geom = county_row.iloc[0]['geometry']
        
        # print(f"Found county: {county_name}, {state_abbrev}")
        
        # Calculate centroid and transform to EPSG:4326 if needed
        centroid = county_geom.centroid
        
        if counties_gdf.crs != 'EPSG:4326':
            # Transform to WGS84
            centroid_gdf = gpd.GeoDataFrame([1], geometry=[centroid], crs=counties_gdf.crs)
            centroid_gdf = centroid_gdf.to_crs('EPSG:4326')
            centroid = centroid_gdf.geometry.iloc[0]
        
        lat, lon = centroid.y, centroid.x
        # print(f"County centroid: {lat:.6f}, {lon:.6f}")
        
        return lat, lon, county_name, state_abbrev
        
    except Exception as e:
        # print(f"Error getting county centroid: {e}")
        return None, None, None, None

def get_zarr_info(zarr_file):
    """Get basic info from Zarr file"""
    try:
        da = xr.open_zarr(zarr_file)
        county_name = da.attrs.get('county', 'Unknown')
        state_name = da.attrs.get('state', 'Unknown')
        geoid = da.attrs.get('geoid', 'Unknown')
        # print(f"Zarr file info - County: {county_name}, State: {state_name}, GEOID: {geoid}")
        return county_name, state_name, geoid
    except Exception as e:
        # print(f"Error reading Zarr file: {e}")
        return None, None, None

def get_500yr_1hr_rainfall(lat, lon):
    """Get 1-hour, 500-year precipitation from NOAA Atlas 14"""
    try:
        # Create temporary directory for the download
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download precipitation data
            csv_path = atlas14.download(
                lat, lon, 
                parent=temp_dir,
                statistic='mean',
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
                # print("Could not find precipitation data header")
                return 90.0
            
            # Parse the header to find the 500-year column index
            header_parts = [x.strip() for x in header_line.split(',')]
            try:
                # Find index of '500' in the header (should be 9th position: 1,2,5,10,25,50,100,200,500,1000)
                return_periods = header_parts[1:]  # Skip "by duration for ARI (years):"
                col_500_idx = return_periods.index('500') + 1  # +1 because first column is duration
            except ValueError:
                # print("Could not find 500-year return period in header")
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
                            # print(f"Could not parse rainfall value: {data_parts[col_500_idx]}")
                            return 90.0
                    else:
                        # print(f"Not enough columns in data row: {len(data_parts)} vs expected {col_500_idx + 1}")
                        return 90.0
            
            # print("Could not find 60-min row in precipitation data")
            return 90.0
            
    except Exception as e:
        # print(f"Error getting precipitation data: {e}")
        return 90.0  # Fallback value

if __name__ == "__main__":
    zarr_file = sys.argv[1]
    shapefile_path = sys.argv[2]
    
    # Try to get GEOID from Zarr file attributes first
    county_name, state_name, geoid = get_zarr_info(zarr_file)
    
    if geoid and geoid != 'Unknown':
        # Use GEOID from Zarr attributes
        lat, lon, county_name_shp, state_abbrev = get_county_centroid_from_geoid(geoid, shapefile_path)
    else:
        # Fallback: try to parse GEOID from filename
        geoid = parse_zarr_filename(zarr_file)
        if geoid:
            lat, lon, county_name_shp, state_abbrev = get_county_centroid_from_geoid(geoid, shapefile_path)
        else:
            # print("Could not determine county GEOID from Zarr file")
            print("90.0")  # Fallback value
            sys.exit(0)
    
    if lat is not None and lon is not None:
        rainfall = get_500yr_1hr_rainfall(lat, lon)
        print(f"{rainfall:.1f}")
    else:
        print("90.0")  # Fallback value

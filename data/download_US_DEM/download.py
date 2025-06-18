#!/usr/bin/env python3
"""
Example County Data Download (Using Shapefile Bounds)
=====================================================

Example usage of the National Map data downloader for specific counties using
direct shapefile geometry bounds instead of name searching.
This script shows how to download DEM data for various counties by directly
accessing their geometry bounds from US Census shapefile data.

Author: Generated for FloodRisk-DL project
"""

import geopandas as gpd
import pandas as pd
from download_national_map_data import download_county_data, print_available_datasets, download_and_cache_county_data

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

def load_county_shapefile():
    """Load the US county shapefile and return GeoDataFrame with readable state names"""
    print("Loading US county shapefile...")
    
    # Download and cache county data if needed
    shapefile_path = download_and_cache_county_data()
    
    # Load the county shapefile
    counties_gdf = gpd.read_file(shapefile_path)
    
    # Add readable state abbreviations
    state_fips_to_abbrev = get_state_fips_mapping()
    counties_gdf['STATE_ABBREV'] = counties_gdf['STATEFP'].map(state_fips_to_abbrev)
    
    print(f"Loaded {len(counties_gdf)} counties from shapefile")
    print(f"Shapefile CRS: {counties_gdf.crs}")
    
    return counties_gdf

def download_counties_by_geoid(geoids):
    """
    Download DEM data for counties using their GEOID codes
    
    Args:
        geoids: List of county GEOID codes (5-digit strings)
    """
    print("Available datasets:")
    print_available_datasets()
    
    print(f"\nDownloading DEM data for {len(geoids)} counties using GEOID codes")
    print("=" * 70)
    
    # Load county shapefile
    counties_gdf = load_county_shapefile()
    
    for geoid in geoids:
        # Find county by GEOID
        county_row = counties_gdf[counties_gdf['GEOID'] == geoid]
        
        if county_row.empty:
            print(f"\n❌ County with GEOID {geoid} not found in shapefile")
            continue
            
        county_info = county_row.iloc[0]
        county_name = county_info['NAME']
        state_abbrev = county_info['STATE_ABBREV']
        
        # Get geometry bounds directly from shapefile
        geometry = county_info.geometry
        bounds = geometry.bounds  # (minx, miny, maxx, maxy)
        
        full_name = f"{county_name} County, {state_abbrev}"
        print(f"\n🗺️  Processing: {full_name} (GEOID: {geoid})")
        print(f"   📍 Bounds: {bounds}")
        
        try:
            # Download 1-meter DEM data using bounds directly from shapefile
            downloaded_files = download_county_data(
                bbox=bounds,  # Use bounds directly from shapefile geometry
                dataset='DEM_1m',  # 1-meter Digital Elevation Model
                output_folder=f'dem_data_{county_name.replace(" ", "_")}_{state_abbrev}_geoid_{geoid}',
                merge_files=True,  # Merge multiple tiles into single file
                force_download=False  # Ask user before large downloads
            )
            
            if downloaded_files:
                print(f"   ✅ Successfully downloaded {len(downloaded_files)} files")
                print(f"   📁 Saved to: dem_data_{county_name.replace(' ', '_')}_{state_abbrev}_geoid_{geoid}")
            else:
                print(f"   ❌ No data available for {full_name}")
                
        except Exception as e:
            print(f"   ❌ Error downloading {full_name}: {str(e)}")
        
        print("-" * 70)

def download_counties_by_state(state_fips, max_counties=5):
    """
    Download DEM data for counties in a specific state using direct shapefile bounds
    
    Args:
        state_fips: State FIPS code (2-digit string)
        max_counties: Maximum number of counties to download (default: 5)
    """
    print("Available datasets:")
    print_available_datasets()
    
    state_fips_to_abbrev = get_state_fips_mapping()
    state_abbrev = state_fips_to_abbrev.get(state_fips, state_fips)
    
    print(f"\nDownloading DEM data for counties in {state_abbrev} (FIPS: {state_fips})")
    print("=" * 70)
    
    # Load county shapefile
    counties_gdf = load_county_shapefile()
    
    # Filter counties by state FIPS code
    state_counties = counties_gdf[counties_gdf['STATEFP'] == state_fips].head(max_counties)
    
    if state_counties.empty:
        print(f"❌ No counties found for state FIPS code: {state_fips}")
        return
    
    print(f"Found {len(state_counties)} counties in {state_abbrev} (showing first {max_counties})")
    
    for idx, county_row in state_counties.iterrows():
        county_name = county_row['NAME']
        geoid = county_row['GEOID']
        state_abbrev = county_row['STATE_ABBREV']
        
        # Get geometry bounds directly from shapefile
        geometry = county_row.geometry
        bounds = geometry.bounds  # (minx, miny, maxx, maxy)
        
        full_name = f"{county_name} County, {state_abbrev}"
        print(f"\n🗺️  Processing: {full_name} (GEOID: {geoid})")
        print(f"   📍 Bounds: {bounds}")
        
        try:
            # Download 1-meter DEM data using bounds directly from shapefile
            downloaded_files = download_county_data(
                bbox=bounds,  # Use bounds directly from shapefile geometry
                dataset='DEM_1m',  # 1-meter Digital Elevation Model
                output_folder=f'dem_data_{county_name.replace(" ", "_")}_{state_abbrev}_geoid_{geoid}',
                merge_files=True,  # Merge multiple tiles into single file
                force_download=False  # Ask user before large downloads
            )
            
            if downloaded_files:
                print(f"   ✅ Successfully downloaded {len(downloaded_files)} files")
                print(f"   📁 Saved to: dem_data_{county_name.replace(' ', '_')}_{state_abbrev}_geoid_{geoid}")
            else:
                print(f"   ❌ No data available for {full_name}")
                
        except Exception as e:
            print(f"   ❌ Error downloading {full_name}: {str(e)}")
        
        print("-" * 70)

def download_sample_counties_by_shapefile():
    """Download data for sample counties using direct shapefile geometry bounds"""
    
    # Define counties by their GEOID codes (more reliable than name searching)
    # These are some major counties across different states
    sample_geoids = [
        # "06075",  # San Francisco County, CA
        "48201",  # Harris County, TX (Houston)
        # "25086",  # Miami-Dade County, FL  
        # "17031",  # Cook County, IL (Chicago)
        # "53033",  # King County, WA (Seattle)
        # "06037",  # Los Angeles County, CA
        # "06059",  # Orange County, CA
        # "04013",  # Maricopa County, AZ (Phoenix)
        # "01073",  # Jefferson County, AL (Birmingham)
        # "12011"   # Broward County, FL
    ]
    
    download_counties_by_geoid(sample_geoids)

def download_all_counties_in_state():
    """Download data for all counties in a specific state using shapefile bounds"""
    
    print("Download All Counties in State")
    print("=" * 40)
    
    # Example: Download all counties in Delaware (small state for testing)
    # Delaware FIPS code: 10
    state_fips = "10"  # Delaware
    max_counties = 3   # Delaware only has 3 counties
    
    download_counties_by_state(state_fips, max_counties)

def browse_counties_in_shapefile():
    """Browse and display counties available in the shapefile"""
    
    print("Browse Counties in Shapefile")
    print("=" * 35)
    
    # Load county shapefile
    counties_gdf = load_county_shapefile()
    
    # Show summary statistics
    print(f"\nShapefile Summary:")
    print(f"Total counties: {len(counties_gdf)}")
    print(f"States represented: {counties_gdf['STATE_ABBREV'].nunique()}")
    print(f"CRS: {counties_gdf.crs}")
    
    # Show states with county counts
    print(f"\nCounties per state:")
    state_counts = counties_gdf['STATE_ABBREV'].value_counts().sort_index()
    for state, count in state_counts.head(10).items():
        print(f"  {state}: {count} counties")
    print("  ... (showing first 10 states)")
    
    # Show sample counties from California
    print(f"\nSample counties from California:")
    ca_counties = counties_gdf[counties_gdf['STATE_ABBREV'] == 'CA'].head(10)
    print(f"{'County Name':<20} {'GEOID':<8} {'Bounds (minx, miny, maxx, maxy)'}")
    print("-" * 70)
    for idx, row in ca_counties.iterrows():
        bounds = row.geometry.bounds
        bounds_str = f"({bounds[0]:.3f}, {bounds[1]:.3f}, {bounds[2]:.3f}, {bounds[3]:.3f})"
        print(f"{row['NAME']:<20} {row['GEOID']:<8} {bounds_str}")

def download_custom_geoids():
    """Download data for custom county GEOIDs specified by user"""
    
    print("Custom County GEOID Download")
    print("=" * 35)
    
    # Example: Download specific counties by GEOID
    # You can replace these with any county GEOIDs you want
    custom_geoids = [
        "06001",  # Alameda County, CA (Oakland/Berkeley)
        "36061",  # New York County, NY (Manhattan)
        "06073"   # San Diego County, CA
    ]
    
    print(f"Downloading data for custom county GEOIDs: {custom_geoids}")
    download_counties_by_geoid(custom_geoids)

def main():
    """Main function with options"""
    
    print("National Map County Data Downloader - Shapefile Bounds Method")
    print("=" * 65)
    
    choice = input("""
Choose an option:
1. Download sample counties (using predefined GEOID codes from shapefile)
2. Download all counties in a state (using shapefile bounds)
3. Download custom county GEOIDs (using shapefile bounds)
4. Browse counties available in shapefile
5. Exit

Enter choice (1-5): """).strip()
    
    if choice == "1":
        download_sample_counties_by_shapefile()
    elif choice == "2":
        download_all_counties_in_state()
    elif choice == "3":
        download_custom_geoids()
    elif choice == "4":
        browse_counties_in_shapefile()
    elif choice == "5":
        print("Exiting...")
    else:
        print("Invalid choice. Please run again and select 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main() 
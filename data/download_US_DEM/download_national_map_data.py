#!/usr/bin/env python3
"""
National Map Data Downloader
============================

This script uses the dem_getter.py module to download elevation and other geospatial data 
from The National Map based on county geometry bounding boxes.

Usage:
    python download_national_map_data.py

Author: Generated for FloodRisk-DL project
"""

import os
import sys
import zipfile
import tempfile
import urllib.request
from typing import List, Tuple, Optional, Union
from pathlib import Path
import geopandas as gpd
import pandas as pd
from dem_getter import (
    get_aws_paths, 
    batch_download, 
    DATASETS_DICT,
    merge_warp_dems
)

# US Census Bureau county shapefile URL
CENSUS_COUNTY_URL = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_20m.zip"
COUNTY_CACHE_DIR = os.path.expanduser("~/.cache/national_map_downloader")

def print_available_datasets():
    """Print all available datasets from The National Map"""
    print("Available datasets:")
    print("-" * 50)
    for short_name, full_name in DATASETS_DICT.items():
        print(f"{short_name:12} : {full_name}")
    print("-" * 50)

def validate_bounding_box(bbox: Tuple[float, float, float, float]) -> bool:
    """
    Validate bounding box coordinates
    
    Args:
        bbox: Tuple of (xMin, yMin, xMax, yMax)
        
    Returns:
        bool: True if valid, False otherwise
    """
    xMin, yMin, xMax, yMax = bbox
    
    # Check if coordinates are reasonable for US bounds
    if not (-180 <= xMin < xMax <= -60):  # Longitude bounds for US
        print(f"Warning: Longitude values seem outside US bounds: {xMin}, {xMax}")
        
    if not (20 <= yMin < yMax <= 70):  # Latitude bounds for US
        print(f"Warning: Latitude values seem outside US bounds: {yMin}, {yMax}")
        
    return xMin < xMax and yMin < yMax

def download_and_cache_county_data():
    """Download and cache US county shapefile data from Census Bureau"""
    
    cache_dir = Path(COUNTY_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    shapefile_path = cache_dir / "cb_2018_us_county_20m.shp"
    
    # Check if already cached
    if shapefile_path.exists():
        print("Using cached county data...")
        return str(shapefile_path)
    
    print("Downloading US county boundaries from Census Bureau...")
    print(f"Source: {CENSUS_COUNTY_URL}")
    
    # Download the zip file
    zip_path = cache_dir / "county_data.zip"
    try:
        urllib.request.urlretrieve(CENSUS_COUNTY_URL, zip_path)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        
        # Clean up zip file
        zip_path.unlink()
        
        if shapefile_path.exists():
            print(f"County data cached to: {cache_dir}")
            return str(shapefile_path)
        else:
            raise FileNotFoundError("Shapefile not found in downloaded data")
            
    except Exception as e:
        print(f"Error downloading county data: {e}")
        raise

def get_county_bbox_from_name(county_name: str, state_name: str) -> Tuple[float, float, float, float]:
    """
    Get bounding box for a county by name and state
    
    Args:
        county_name: Name of the county (with or without "County" suffix)
        state_name: Name or abbreviation of the state
        
    Returns:
        Tuple of (xMin, yMin, xMax, yMax) in WGS84 coordinates
        
    Raises:
        ValueError: If county is not found
    """
    
    # Download and cache county data if needed
    shapefile_path = download_and_cache_county_data()
    
    # Load the county shapefile
    print("Loading county boundaries...")
    counties_gdf = gpd.read_file(shapefile_path)
    
    # Normalize inputs
    county_normalized = county_name.strip().title()
    state_normalized = state_name.strip().upper()
    
    # Handle "County" suffix - add if not present
    if not county_normalized.endswith(" County"):
        county_normalized += " County"
    
    # Create state FIPS code to abbreviation mapping (based on US Census FIPS codes)
    state_fips_to_abbrev = {
        '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE',
        '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA',
        '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN',
        '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM',
        '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
        '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA',
        '54': 'WV', '55': 'WI', '56': 'WY', '72': 'PR'
    }
    
    # Create state abbreviation to full name mapping
    state_abbreviations = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
        'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
        'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
        'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
        'DC': 'District of Columbia', 'PR': 'Puerto Rico'
    }
    
    # Create reverse mapping from abbreviation to FIPS code
    abbrev_to_fips = {v: k for k, v in state_fips_to_abbrev.items()}
    
    # Determine target state FIPS code
    target_state_fips = None
    
    if len(state_normalized) == 2 and state_normalized in state_abbreviations:
        # It's a state abbreviation like 'CA'
        target_state_fips = abbrev_to_fips.get(state_normalized)
        state_display_name = f"{state_abbreviations[state_normalized]} ({state_normalized})"
    else:
        # It's a full state name like 'California', find the abbreviation
        state_abbrev = None
        for abbrev, full_name in state_abbreviations.items():
            if full_name.upper() == state_normalized.upper():
                state_abbrev = abbrev
                target_state_fips = abbrev_to_fips.get(abbrev)
                break
        
        if target_state_fips:
            state_display_name = f"{state_normalized} ({state_abbrev})"
        else:
            # Try partial matching for state name
            for abbrev, full_name in state_abbreviations.items():
                if state_normalized.upper() in full_name.upper():
                    target_state_fips = abbrev_to_fips.get(abbrev)
                    state_display_name = f"{full_name} ({abbrev})"
                    break
    
    if not target_state_fips:
        raise ValueError(f"Could not find state '{state_name}'. Please use a valid state name or abbreviation.")
    
    # Search for the county
    print(f"Searching for: {county_normalized} in {state_display_name}")
    print(f"Target state FIPS code: {target_state_fips}")
    
    # Search by county name and state FIPS code
    county_match = counties_gdf[
        (counties_gdf['NAME'] == county_normalized.replace(" County", "")) &
        (counties_gdf['STATEFP'] == target_state_fips)
    ]
    
    # If no exact match, try fuzzy matching
    if county_match.empty:
        # Remove "County" for search and try partial matches within the same state
        county_base = county_normalized.replace(" County", "").replace("County", "").strip()
        
        # Try different matching strategies within the target state
        possible_matches_in_state = counties_gdf[
            (counties_gdf['NAME'].str.contains(county_base, case=False, na=False)) &
            (counties_gdf['STATEFP'] == target_state_fips)
        ]
        
        if not possible_matches_in_state.empty:
            print(f"Possible county matches in {state_display_name}:")
            for idx, row in possible_matches_in_state.head(5).iterrows():
                state_abbrev = state_fips_to_abbrev.get(row['STATEFP'], row['STATEFP'])
                print(f"  - {row['NAME']} County, {state_abbrev} (GEOID: {row['GEOID']})")
        else:
            # Also show matches across all states for reference
            all_matches = counties_gdf[
                counties_gdf['NAME'].str.contains(county_base, case=False, na=False)
            ]
            if not all_matches.empty:
                print(f"No matches in {state_display_name}, but found these in other states:")
                for idx, row in all_matches.head(5).iterrows():
                    state_abbrev = state_fips_to_abbrev.get(row['STATEFP'], row['STATEFP'])
                    print(f"  - {row['NAME']} County, {state_abbrev} (GEOID: {row['GEOID']})")
        
        raise ValueError(f"County '{county_name}' not found in state '{state_name}'. "
                        f"Please check spelling or try the exact county name from the suggestions above.")
    
    if len(county_match) > 1:
        print(f"Multiple counties found matching '{county_name}' in {state_display_name}:")
        for idx, row in county_match.iterrows():
            state_abbrev = state_fips_to_abbrev.get(row['STATEFP'], row['STATEFP'])
            print(f"  - {row['NAME']} County, {state_abbrev} (GEOID: {row['GEOID']})")
        # Use the first match
        county_match = county_match.iloc[:1]
        print(f"Using first match: {county_match.iloc[0]['NAME']} County")
    
    # Get the geometry and calculate bounding box
    county_geom = county_match.iloc[0].geometry
    bounds = county_geom.bounds  # (minx, miny, maxx, maxy)
    
    county_row = county_match.iloc[0]
    county_full_name = county_row['NAME']
    state_abbrev = state_fips_to_abbrev.get(county_row['STATEFP'], county_row['STATEFP'])
    
    print(f"✅ Found: {county_full_name} County, {state_abbrev}")
    print(f"   GEOID: {county_row['GEOID']}")
    print(f"   Bounding box: {bounds}")
    
    return bounds  # (xMin, yMin, xMax, yMax)

def search_counties(search_term: str, state_filter: str = None) -> pd.DataFrame:
    """
    Search for counties by name
    
    Args:
        search_term: Partial county name to search for
        state_filter: Optional state abbreviation or name to filter results
        
    Returns:
        DataFrame with matching counties and readable state names
    """
    shapefile_path = download_and_cache_county_data()
    counties_gdf = gpd.read_file(shapefile_path)
    
    # State FIPS to abbreviation mapping
    state_fips_to_abbrev = {
        '01': 'AL', '02': 'AK', '04': 'AZ', '05': 'AR', '06': 'CA', '08': 'CO', '09': 'CT', '10': 'DE',
        '11': 'DC', '12': 'FL', '13': 'GA', '15': 'HI', '16': 'ID', '17': 'IL', '18': 'IN', '19': 'IA',
        '20': 'KS', '21': 'KY', '22': 'LA', '23': 'ME', '24': 'MD', '25': 'MA', '26': 'MI', '27': 'MN',
        '28': 'MS', '29': 'MO', '30': 'MT', '31': 'NE', '32': 'NV', '33': 'NH', '34': 'NJ', '35': 'NM',
        '36': 'NY', '37': 'NC', '38': 'ND', '39': 'OH', '40': 'OK', '41': 'OR', '42': 'PA', '44': 'RI',
        '45': 'SC', '46': 'SD', '47': 'TN', '48': 'TX', '49': 'UT', '50': 'VT', '51': 'VA', '53': 'WA',
        '54': 'WV', '55': 'WI', '56': 'WY', '72': 'PR'
    }
    
    # Search for counties matching the term
    matches = counties_gdf[
        counties_gdf['NAME'].str.contains(search_term, case=False, na=False)
    ]
    
    # Filter by state if provided
    if state_filter:
        state_normalized = state_filter.strip().upper()
        
        # Convert state name/abbreviation to FIPS code
        target_fips = None
        if len(state_normalized) == 2:
            # It's likely an abbreviation
            for fips, abbrev in state_fips_to_abbrev.items():
                if abbrev == state_normalized:
                    target_fips = fips
                    break
        else:
            # It's likely a full state name - simplified matching
            state_abbreviations = {
                'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR', 'CALIFORNIA': 'CA',
                'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE', 'FLORIDA': 'FL', 'GEORGIA': 'GA',
                'HAWAII': 'HI', 'IDAHO': 'ID', 'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA',
                'KANSAS': 'KS', 'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
                'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS', 'MISSOURI': 'MO',
                'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV', 'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ',
                'NEW MEXICO': 'NM', 'NEW YORK': 'NY', 'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH',
                'OKLAHOMA': 'OK', 'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
                'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT', 'VERMONT': 'VT',
                'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV', 'WISCONSIN': 'WI', 'WYOMING': 'WY',
                'DISTRICT OF COLUMBIA': 'DC', 'PUERTO RICO': 'PR'
            }
            
            state_abbrev = state_abbreviations.get(state_normalized)
            if state_abbrev:
                for fips, abbrev in state_fips_to_abbrev.items():
                    if abbrev == state_abbrev:
                        target_fips = fips
                        break
        
        if target_fips:
            matches = matches[matches['STATEFP'] == target_fips]
    
    # Add readable state abbreviations
    matches = matches.copy()
    matches['STATE'] = matches['STATEFP'].map(state_fips_to_abbrev)
    
    # Select and reorder columns for display
    result = matches[['NAME', 'STATE', 'STATEFP', 'GEOID']].head(20)
    
    return result

def download_county_data(
    bbox: Union[Tuple[float, float, float, float], None] = None,
    county_name: str = None,
    state_name: str = None,
    dataset: str = 'DEM_1m',
    output_folder: str = 'national_map_data',
    data_type: str = '',
    input_epsg: int = 4326,
    exclude_redundant: bool = True,
    force_download: bool = False,
    merge_files: bool = False
) -> List[str]:
    """
    Download National Map data for a county
    
    Args:
        bbox: Tuple of (xMin, yMin, xMax, yMax) in decimal degrees or specified CRS (optional)
        county_name: Name of the county (optional, used with state_name)
        state_name: Name or abbreviation of the state (optional, used with county_name)
        dataset: Dataset to download (default: 'DEM_1m')
        output_folder: Folder to save downloaded data
        data_type: Specific data format (leave empty for default)
        input_epsg: EPSG code for input coordinates (default: 4326 for WGS84)
        exclude_redundant: Remove duplicate/overlapping datasets
        force_download: Skip download size confirmation
        merge_files: Merge downloaded rasters into single file
        
    Returns:
        List of downloaded file paths
        
    Note:
        Either provide bbox OR (county_name AND state_name). County/state method
        automatically fetches boundaries from US Census Bureau data.
    """
    
    # Determine how to get the bounding box
    if bbox is not None:
        # Use provided bounding box
        print(f"Using provided bounding box: {bbox}")
        if not validate_bounding_box(bbox):
            raise ValueError("Invalid bounding box coordinates")
        xMin, yMin, xMax, yMax = bbox
        area_name = f"Custom Area {bbox}"
        
    elif county_name and state_name:
        # Get bounding box from county/state names
        print(f"Looking up county: {county_name}, {state_name}")
        try:
            bbox = get_county_bbox_from_name(county_name, state_name)
            xMin, yMin, xMax, yMax = bbox
            area_name = f"{county_name} County, {state_name}"
            print(f"Using county bounding box: {bbox}")
        except Exception as e:
            print(f"Error finding county: {e}")
            raise ValueError(f"Could not find county '{county_name}' in state '{state_name}'")
            
    else:
        raise ValueError("Must provide either 'bbox' or both 'county_name' and 'state_name'")
    
    print(f"Downloading {dataset} data for: {area_name}")
    print(f"Output folder: {output_folder}")
    print(f"Input CRS: EPSG:{input_epsg}")
    
    try:
        # Get AWS download paths
        print("\nQuerying The National Map API...")
        aws_paths = get_aws_paths(
            dataset=dataset,
            xMin=xMin,
            yMin=yMin, 
            xMax=xMax,
            yMax=yMax,
            filePath=None,  # Don't save paths to file
            dataType=data_type,
            inputEPSG=input_epsg,
            doExcludeRedundantData=exclude_redundant
        )
        
        if not aws_paths:
            print("No data found for the specified bounding box and dataset.")
            return []
            
        print(f"Found {len(aws_paths)} datasets to download")
        
        # Download the data
        print(f"\nDownloading data to folder: {output_folder}")
        downloaded_files = batch_download(
            dlList=aws_paths,
            folderName=output_folder,
            doForceDownload=force_download
        )
        
        print(f"\nSuccessfully downloaded {len(downloaded_files)} files")
        
        # Optionally merge raster files
        if merge_files and downloaded_files:
            print("\nMerging downloaded raster files...")
            
            # Filter for raster files (common extensions)
            raster_extensions = ['.tif', '.tiff', '.img', '.bil', '.adf']
            raster_files = [
                f for f in downloaded_files 
                if any(f.lower().endswith(ext) for ext in raster_extensions)
            ]
            
            if len(raster_files) > 1:
                merged_filename = os.path.join(output_folder, f"merged_{dataset}.tif")
                print(f"Merging {len(raster_files)} raster files into: {merged_filename}")
                
                merge_warp_dems(
                    inFileNames=raster_files,
                    outFileName=merged_filename,
                    outExtent=[[xMin, xMax], [yMin, yMax]],
                    outEPSG=input_epsg
                )
                
                downloaded_files.append(merged_filename)
                print(f"Merged file saved as: {merged_filename}")
            else:
                print("Only one raster file found, skipping merge.")
        
        return downloaded_files
        
    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        raise

def get_user_input():
    """Get user input for download parameters"""
    
    print("National Map Data Downloader")
    print("=" * 40)
    
    # Show available datasets
    print_available_datasets()
    
    # Get dataset choice
    dataset = input(f"\nEnter dataset (default: DEM_1m): ").strip()
    if not dataset:
        dataset = 'DEM_1m'
    
    # Validate dataset choice
    if dataset not in DATASETS_DICT:
        print(f"Invalid dataset. Using default: DEM_1m")
        dataset = 'DEM_1m'
    
    # Choose input method
    print(f"\nChoose input method:")
    print("1. County and State names (recommended)")
    print("2. Manual bounding box coordinates")
    print("3. Search for counties")
    
    choice = input("Enter choice (1-3, default: 1): ").strip()
    if not choice:
        choice = "1"
    
    bbox = None
    county_name = None
    state_name = None
    
    if choice == "1":
        # Get county and state
        print(f"\nEnter county and state information:")
        county_name = input("County name (e.g., 'San Francisco' or 'Harris'): ").strip()
        state_name = input("State name or abbreviation (e.g., 'CA' or 'California'): ").strip()
        
        if not county_name or not state_name:
            print("Both county and state are required.")
            return None
            
    elif choice == "2":
        # Get bounding box coordinates
        print(f"\nEnter bounding box coordinates (in decimal degrees, WGS84):")
        print("Format: xMin yMin xMax yMax (space-separated)")
        print("Example for San Francisco County: -122.515 37.708 -122.357 37.833")
        
        bbox_input = input("Bounding box: ").strip()
        try:
            coords = [float(x) for x in bbox_input.split()]
            if len(coords) != 4:
                raise ValueError("Must provide exactly 4 coordinates")
            bbox = tuple(coords)
        except ValueError as e:
            print(f"Invalid coordinates: {e}")
            return None
            
    elif choice == "3":
        # Search for counties
        search_term = input("Enter partial county name to search: ").strip()
        state_filter = input("Enter state to filter by (optional, e.g., 'CA' or 'California'): ").strip()
        
        if search_term:
            print("Searching for counties...")
            try:
                if state_filter:
                    matches = search_counties(search_term, state_filter)
                else:
                    matches = search_counties(search_term)
                    
                if not matches.empty:
                    print("\nMatching counties:")
                    print(f"{'County Name':<20} {'State':<5} {'GEOID':<12} {'State FIPS':<10}")
                    print("-" * 50)
                    for idx, row in matches.iterrows():
                        print(f"{row['NAME']:<20} {row['STATE']:<5} {row['GEOID']:<12} {row['STATEFP']:<10}")
                else:
                    print("No matching counties found.")
                print("\nPlease run the script again and use option 1 with the exact county name and state.")
                return None
            except Exception as e:
                print(f"Error searching counties: {e}")
                return None
        else:
            print("No search term provided.")
            return None
    else:
        print("Invalid choice.")
        return None
    
    # Get output folder
    output_folder = input(f"\nOutput folder (default: national_map_data): ").strip()
    if not output_folder:
        output_folder = 'national_map_data'
    
    # Ask about merging files
    merge_choice = input(f"\nMerge downloaded raster files? (y/N): ").strip().lower()
    merge_files = merge_choice in ['y', 'yes']
    
    return {
        'bbox': bbox,
        'county_name': county_name,
        'state_name': state_name,
        'dataset': dataset,
        'output_folder': output_folder,
        'merge_files': merge_files
    }

def main():
    """Main function for interactive usage"""
    
    # Example usage with county names (commented out)
    # Uncomment and modify for programmatic usage
    """
    # Example 1: Using county and state names
    downloaded_files = download_county_data(
        county_name='San Francisco',
        state_name='CA',
        dataset='DEM_1m',
        output_folder='sf_dem_data',
        merge_files=True
    )
    
    # Example 2: Using bounding box (legacy method)
    sf_bbox = (-122.515, 37.708, -122.357, 37.833)
    downloaded_files = download_county_data(
        bbox=sf_bbox,
        dataset='DEM_1m',
        output_folder='sf_dem_data',
        merge_files=True
    )
    
    print(f"Downloaded files: {downloaded_files}")
    """
    
    # Interactive mode
    try:
        params = get_user_input()
        if params is None:
            return
        
        downloaded_files = download_county_data(**params)
        
        print(f"\nDownload complete!")
        print(f"Files saved to: {params['output_folder']}")
        print(f"Number of files: {len(downloaded_files)}")
        
        if downloaded_files:
            print("\nDownloaded files:")
            for file_path in downloaded_files:
                print(f"  - {file_path}")
                
    except KeyboardInterrupt:
        print("\nDownload cancelled by user.")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
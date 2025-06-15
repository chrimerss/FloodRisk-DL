#!/usr/bin/env python3
"""
Example County Data Download
============================

Example usage of the National Map data downloader for specific counties.
This script shows how to download DEM data for various counties programmatically.

Author: Generated for FloodRisk-DL project
"""

from download_national_map_data import download_county_data, print_available_datasets

def download_counties():
    """Download data for several example counties using county/state names"""
    
    # Define counties by name and state (much simpler!)
    counties = [
        # ("San Francisco", "CA"),
        # ("Harris", "TX"), 
        # ("Miami-Dade", "FL"),
        # ("Philadelphia", "PA"),
        # ("Hillsborough", "FL"),
        # ("Wake", "NC"),
        # ("Greenville", "SC"),
        ("Baltimore", "MD"),
        # ("East Baton Rouge", "LA"),
        # ("Queens", "NY")
    ]
    
    print("Available datasets:")
    print_available_datasets()
    
    print("\nExample County Downloads (using county/state names)")
    print("=" * 60)
    
    for county_name, state_name in counties:
        full_name = f"{county_name} County, {state_name}"
        print(f"\n🗺️  Processing: {full_name}")
        
        try:
            # Download 1-meter DEM data using county/state names
            downloaded_files = download_county_data(
                county_name=county_name,
                state_name=state_name,
                dataset='DEM_1m',  # 1-meter Digital Elevation Model
                output_folder=f'dem_data_{county_name.replace(" ", "_")}_{state_name}',
                merge_files=True,  # Merge multiple tiles into single file
                force_download=False  # Ask user before large downloads
            )
            
            if downloaded_files:
                print(f"   ✅ Successfully downloaded {len(downloaded_files)} files")
                print(f"   📁 Saved to: dem_data_{county_name.replace(' ', '_')}_{state_name}")
            else:
                print(f"   ❌ No data available for {full_name}")
                
        except Exception as e:
            print(f"   ❌ Error downloading {full_name}: {str(e)}")
        
        print("-" * 60)

def download_custom_county():
    """Download data for a custom county using county/state names"""
    
    print("Custom County Download")
    print("=" * 30)
    
    # Example: Download data for a specific county
    # Replace these with your desired county and state
    custom_county = "Alameda"  # County name
    custom_state = "CA"       # State abbreviation or full name
    
    print(f"Downloading data for: {custom_county} County, {custom_state}")
    
    # You can choose different datasets:
    datasets_to_try = ['DEM_1m', 'NED_1-3as', 'DEM_5m']
    
    for dataset in datasets_to_try:
        print(f"\nTrying dataset: {dataset}")
        
        try:
            downloaded_files = download_county_data(
                county_name=custom_county,
                state_name=custom_state,
                dataset=dataset,
                output_folder=f'{custom_county.lower()}_{custom_state}_{dataset}',
                merge_files=True,
                force_download=True  # Skip confirmation for example
            )
            
            if downloaded_files:
                print(f"✅ Downloaded {len(downloaded_files)} files for {dataset}")
                break  # Stop after first successful download
            else:
                print(f"❌ No {dataset} data available for {custom_county} County, {custom_state}")
                
        except Exception as e:
            print(f"❌ Error with {dataset}: {str(e)}")

def download_by_bounding_box():
    """Download data using manual bounding box (legacy method)"""
    
    print("Manual Bounding Box Download")
    print("=" * 35)
    
    # Example: Custom bounding box for a specific area
    # Replace these coordinates with your area's bounding box
    custom_bbox = (
        -122.0,  # xMin (longitude)
        37.0,    # yMin (latitude)  
        -121.5,  # xMax (longitude)
        37.5     # yMax (latitude)
    )
    
    print(f"Using bounding box: {custom_bbox}")
    
    try:
        downloaded_files = download_county_data(
            bbox=custom_bbox,
            dataset='DEM_1m',
            output_folder='custom_bbox_area',
            merge_files=True,
            force_download=True  # Skip confirmation for example
        )
        
        if downloaded_files:
            print(f"✅ Downloaded {len(downloaded_files)} files")
        else:
            print(f"❌ No data available for this bounding box")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def search_counties_demo():
    """Demonstrate the county search functionality"""
    
    print("County Search Demo")
    print("=" * 25)
    
    # Demo searches
    searches = [
        ("Jefferson", None),    # Counties named Jefferson in any state
        ("Washington", None),   # Counties named Washington in any state  
        ("San", "CA"),         # Counties starting with "San" in California
        ("King", "WA"),        # King county in Washington
        ("Harris", "TX")       # Harris county in Texas
    ]
    
    for search_term, state_filter in searches:
        print(f"\n🔍 Searching for '{search_term}'" + 
              (f" in {state_filter}" if state_filter else " (all states)"))
        print("-" * 40)
        
        try:
            if state_filter:
                results = search_counties(search_term, state_filter)
            else:
                results = search_counties(search_term)
            
            if not results.empty:
                print(f"{'County Name':<20} {'State':<5} {'GEOID':<12}")
                print("-" * 40)
                for idx, row in results.head(5).iterrows():  # Show first 5 results
                    print(f"{row['NAME']:<20} {row['STATE']:<5} {row['GEOID']:<12}")
                if len(results) > 5:
                    print(f"... and {len(results) - 5} more")
            else:
                print("No matching counties found.")
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main function with options"""
    
    print("National Map County Data Downloader Examples")
    print("=" * 50)
    
    choice = input("""
Choose an option:
1. Download sample counties (predefined examples using county/state names)
2. Download custom county (using county/state names)
3. Download using manual bounding box (legacy method)
4. Search counties demo (see how county search works)
5. Exit

Enter choice (1-5): """).strip()
    
    if choice == "1":
        download_counties()
    elif choice == "2":
        download_custom_county()
    elif choice == "3":
        download_by_bounding_box()
    elif choice == "4":
        search_counties_demo()
    elif choice == "5":
        print("Exiting...")
    else:
        print("Invalid choice. Please run again and select 1, 2, 3, 4, or 5.")

if __name__ == "__main__":
    main() 
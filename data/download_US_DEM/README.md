# National Map Data Downloader

This package provides Python scripts to download geospatial data from The National Map using simple county and state names. It automatically fetches county boundaries from the [US Census Bureau](https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_20m.zip) and uses the existing `terratorch/dem_getter.py` module to interface with The National Map API.

## ✨ **New Feature: County/State Name Input**

Simply provide county and state names like "San Francisco, CA" or "Harris, TX" instead of manually looking up coordinates!

## Files

- `download_national_map_data.py` - Main downloader script with interactive and programmatic interfaces
- `example_county_download.py` - Example usage script with predefined county bounding boxes
- `terratorch/dem_getter.py` - Core module for interfacing with The National Map API

## Available Datasets

| Short Name | Full Dataset Name |
|------------|-------------------|
| DEM_1m     | Digital Elevation Model (DEM) 1 meter |
| DEM_5m     | Alaska IFSAR 5 meter DEM |
| NED_1-9as  | National Elevation Dataset (NED) 1/9 arc-second |
| NED_1-3as  | National Elevation Dataset (NED) 1/3 arc-second |
| NED_1as    | National Elevation Dataset (NED) 1 arc-second |
| NED_2as    | National Elevation Dataset (NED) Alaska 2 arc-second |
| LPC        | Lidar Point Cloud (LPC) |
| OPR        | Original Product Resolution (OPR) Digital Elevation Model (DEM) |

## Requirements

```bash
pip install numpy requests gdal pyproj geopandas pandas
```

Note: `geopandas` is now required for automatic county boundary fetching from US Census data.

## Usage

### Method 1: Interactive Mode (Recommended)

Run the script and follow the prompts:

```bash
python download_national_map_data.py
```

You'll be prompted to:
1. Choose a dataset type
2. **Choose input method:**
   - **County/State names** (recommended) - Just enter "San Francisco" and "CA"
   - Manual bounding box coordinates (legacy method)
   - Search for counties by partial name
3. Specify output folder
4. Choose whether to merge raster files

### Method 2: Programmatic Usage (County/State Names)

**New simplified approach using county and state names:**

```python
from download_national_map_data import download_county_data

# Download 1-meter DEM data using county/state names
downloaded_files = download_county_data(
    county_name='San Francisco',
    state_name='CA',  # or 'California'
    dataset='DEM_1m',
    output_folder='sf_dem_data',
    merge_files=True,
    force_download=False
)

print(f"Downloaded {len(downloaded_files)} files")
```

**Legacy approach using manual bounding box:**

```python
from download_national_map_data import download_county_data

# Define county bounding box (xMin, yMin, xMax, yMax in WGS84)
county_bbox = (-122.515, 37.708, -122.357, 37.833)  # San Francisco County

# Download 1-meter DEM data
downloaded_files = download_county_data(
    bbox=county_bbox,
    dataset='DEM_1m',
    output_folder='sf_dem_data',
    merge_files=True,
    force_download=False
)

print(f"Downloaded {len(downloaded_files)} files")
```

### Method 3: Using Examples

Run the example script with predefined counties:

```bash
python example_county_download.py
```

## Function Parameters

### `download_county_data()`

**Input Methods (choose one):**
- `county_name` + `state_name`: County and state names (recommended)
- `bbox`: Tuple of (xMin, yMin, xMax, yMax) coordinates (legacy)

**Parameters:**
- `county_name`: Name of county (e.g., 'San Francisco', 'Harris') - optional
- `state_name`: State name or abbreviation (e.g., 'CA', 'California') - optional  
- `bbox`: Tuple of (xMin, yMin, xMax, yMax) coordinates - optional
- `dataset`: Dataset type (default: 'DEM_1m')
- `output_folder`: Folder to save downloads (default: 'national_map_data')
- `data_type`: Specific data format (leave empty for default)
- `input_epsg`: EPSG code for input coordinates (default: 4326 for WGS84)
- `exclude_redundant`: Remove duplicate datasets (default: True)
- `force_download`: Skip download size confirmation (default: False)
- `merge_files`: Merge downloaded rasters into single file (default: False)

**Note:** Either provide `county_name` + `state_name` OR `bbox`. The county/state method automatically fetches boundaries from US Census data.

## County Data Sources

The script automatically downloads county boundary data from the **[US Census Bureau TIGER/Line Shapefiles](https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_county_20m.zip)** when you use county/state names.

### Manual Bounding Box Sources (if needed)

If you prefer to use manual bounding boxes:

1. **Online Tools:**
   - [BoundingBox.io](http://boundingbox.io/) - Interactive map tool
   - [USGS GeoPlatform](https://www.usgs.gov/tools/geoplatform) - Official USGS tool

2. **GIS Software:**
   - QGIS, ArcGIS, or other GIS tools
   - Load county shapefiles and extract bounds

3. **Government Data:**
   - [US Census Bureau TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
   - State and local government GIS portals

## Example Counties 

Here are some example counties you can use with the county/state name method:

```python
# Simple county/state pairs - no coordinates needed!
counties = [
    ("San Francisco", "CA"),
    ("Harris", "TX"), 
    ("Miami-Dade", "FL"),
    ("Cook", "IL"),
    ("King", "WA"),
    ("Los Angeles", "CA"),
    ("Orange", "CA"),
    ("Maricopa", "AZ"),
    ("Jefferson", "AL"),
    ("Broward", "FL")
]

# Usage example
for county_name, state_name in counties:
    download_county_data(
        county_name=county_name,
        state_name=state_name,
        dataset='DEM_1m'
    )
```

### Legacy Bounding Box Examples

If using the manual bounding box method:

```python
county_bboxes = {
    "San Francisco County, CA": (-122.515, 37.708, -122.357, 37.833),
    "Harris County, TX": (-95.823, 29.523, -94.866, 30.110),
    "Miami-Dade County, FL": (-80.868, 25.137, -80.119, 25.979),
    "Cook County, IL": (-88.263, 41.469, -87.525, 42.154),
    "King County, WA": (-122.542, 47.073, -121.063, 47.776)
}
```

## Output

The script will:
1. Query The National Map API for available datasets
2. Display download size and prompt for confirmation (unless `force_download=True`)
3. Download files to the specified output folder
4. Optionally merge multiple raster files into a single file
5. Return a list of downloaded file paths

## Tips

1. **Use County/State Names:** The new method is much easier than manual coordinates
2. **Start Small:** Begin with a small county to test the download process
3. **Check Data Availability:** Not all datasets are available for all areas
4. **Storage Space:** DEM files can be large - check available disk space
5. **Merge Files:** Use `merge_files=True` to combine multiple tiles into one file
6. **State Abbreviations:** Both "CA" and "California" work for state names
7. **County Variations:** Try both "San Francisco" and "San Francisco County"
8. **Search Function:** Use the search option to find county name variations

## Error Handling

The script includes error handling for:
- Invalid bounding box coordinates
- API connection issues
- Missing data for specified areas
- Download failures
- File system errors

## Integration with FloodRisk-DL

This downloader is designed to work with the FloodRisk-DL project by providing high-quality elevation data for flood risk modeling. The downloaded DEM data can be used as input for:

- Hydrodynamic modeling
- Flood inundation mapping
- Terrain analysis
- Machine learning model training

## Support

For issues related to:
- **The National Map API:** Contact USGS support
- **GDAL/Raster processing:** Check GDAL documentation
- **This script:** Review error messages and ensure all dependencies are installed 
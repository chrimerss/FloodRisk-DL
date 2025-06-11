# National Map Data Downloader

This package provides Python scripts to download geospatial data from The National Map using county geometry bounding boxes. It uses the existing `terratorch/dem_getter.py` module to interface with The National Map API.

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
pip install numpy requests gdal pyproj
```

## Usage

### Method 1: Interactive Mode

Run the script and follow the prompts:

```bash
python download_national_map_data.py
```

You'll be prompted to:
1. Choose a dataset type
2. Enter bounding box coordinates (xMin yMin xMax yMax)
3. Specify output folder
4. Choose whether to merge raster files

### Method 2: Programmatic Usage

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

- `bbox`: Tuple of (xMin, yMin, xMax, yMax) coordinates
- `dataset`: Dataset type (default: 'DEM_1m')
- `output_folder`: Folder to save downloads (default: 'national_map_data')
- `data_type`: Specific data format (leave empty for default)
- `input_epsg`: EPSG code for input coordinates (default: 4326 for WGS84)
- `exclude_redundant`: Remove duplicate datasets (default: True)
- `force_download`: Skip download size confirmation (default: False)
- `merge_files`: Merge downloaded rasters into single file (default: False)

## Getting County Bounding Boxes

You can obtain county bounding boxes from various sources:

1. **Online Tools:**
   - [BoundingBox.io](http://boundingbox.io/) - Interactive map tool
   - [USGS GeoPlatform](https://www.usgs.gov/tools/geoplatform) - Official USGS tool

2. **GIS Software:**
   - QGIS, ArcGIS, or other GIS tools
   - Load county shapefiles and extract bounds

3. **Government Data:**
   - [US Census Bureau TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
   - State and local government GIS portals

## Example County Bounding Boxes

Here are some example county bounding boxes (xMin, yMin, xMax, yMax in WGS84):

```python
counties = {
    "San Francisco County, CA": (-122.515, 37.708, -122.357, 37.833),
    "Harris County, TX": (-95.823, 29.523, -94.866, 30.110),
    "Miami-Dade County, FL": (-80.868, 25.137, -80.119, 25.979),
    "Cook County, IL": (-88.263, 41.469, -87.525, 42.154),
    "King County, WA": (-122.542, 47.073, -121.063, 47.776),
    "Los Angeles County, CA": (-118.944, 33.704, -117.646, 34.823),
    "Orange County, CA": (-118.006, 33.347, -117.421, 33.948)
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

1. **Start Small:** Begin with a small county or area to test the download process
2. **Check Data Availability:** Not all datasets are available for all areas
3. **Storage Space:** DEM files can be large - check available disk space
4. **Merge Files:** Use `merge_files=True` to combine multiple tiles into one file
5. **Coordinate System:** Bounding boxes should be in WGS84 (EPSG:4326) decimal degrees

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
import xarray as xr
import rasterio
import numpy as np
import os
from rasterio.windows import Window
from glob import glob
import json
from rasterio.transform import from_origin
from rtree import index  # For the spatial index
from scipy.spatial import cKDTree
from pyproj import CRS, Transformer  # For reprojection
import h5py

BUFFER = 10
rgb_to_class = {
    (255, 0, 0): 1,  # Red - building
    (133, 133, 133): 2,  # Gray - road
    (255, 0, 192): 3,  # Purple - parking lot
    (34, 139, 34): 4,  # Dark green - tree canopy
    (128, 236, 104): 5,  # Light green - grass/shrub
    (0, 0, 255): 6,  # Blue - water
    (255, 193, 37): 7,  # Yellow - agriculture
    (128, 0, 0): 8,  # Dark red - barren
    (255, 255, 255): 9,  # White - others
}

# Convert dictionary to a structured NumPy array for fast lookup
rgb_keys = np.array(list(rgb_to_class.keys()), dtype=np.uint8)
class_values = np.array(list(rgb_to_class.values()), dtype=np.uint8)


def map_rgb_to_class(ds2):
    """Maps an RGB image in xarray format to discrete classes."""
    # Ensure ds2 has correct dimensions
    if 'band' not in ds2.dims or 'y' not in ds2.dims or 'x' not in ds2.dims:
        raise ValueError(f"Expected dimensions ('band', 'y', 'x'), but got {ds2.dims}")

    # Extract RGB bands safely
    r, g, b = ds2.sel(band=1)['band_data'], ds2.sel(band=2)['band_data'], ds2.sel(band=3)['band_data']

    # Stack into (height, width, 3)
    rgb_array = np.stack([r.values, g.values, b.values], axis=-1)

    # Ensure data type compatibility
    rgb_array = rgb_array.astype(np.uint8)  

    # Create an empty array for class labels
    class_array = np.zeros((ds2.sizes['y'], ds2.sizes['x']), dtype=np.uint8)  

    # Vectorized mapping
    for rgb, class_label in rgb_to_class.items():
        mask = np.all(rgb_array == np.array(rgb, dtype=np.uint8), axis=-1)
        class_array[mask] = class_label  

    # Convert to xarray DataArray
    class_map = xr.DataArray(class_array, coords={'y': ds2.y, 'x': ds2.x}, dims=('y', 'x'))

    return class_map

# Apply function to ds2
# ds2_classified = map_rgb_to_class(ds2)

def create_geotiff_index(geotiff_paths, target_crs="EPSG:4326"):  # Default to WGS84
    """Creates a spatial index of GeoTIFF boundaries, reprojecting to target_crs."""
    idx = index.Index()
    geotiff_data = {}

    for i, geotiff_path in enumerate(geotiff_paths):
        with rasterio.open(geotiff_path) as src:
            # Get source CRS
            source_crs = src.crs

            # Create transformer (only once per GeoTIFF)
            transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True) # Always xy=True ensures lon, lat order

            # Get the bounds of the GeoTIFF in source CRS
            left, bottom, right, top = src.bounds

            # Reproject bounds to target CRS
            reprojected_left, reprojected_bottom = transformer.transform(left, bottom)
            reprojected_right, reprojected_top = transformer.transform(right, top)

            # print(f"Reprojected bounds: {reprojected_left}, {reprojected_bottom}, {reprojected_right}, {reprojected_top}")

            # Insert reprojected bounds into the index
            idx.insert(i, (reprojected_left, reprojected_bottom, reprojected_right, reprojected_top))
            geotiff_data[i] = {"path": geotiff_path, "transform": src.transform, "height": src.height, "width": src.width, "source_crs": source_crs}


    return idx, geotiff_data

def get_value_from_geotiffs_indexed(idx, geotiff_data, ref_file):
    """Retrieves value using the spatial index."""
    # Query the index for intersecting GeoTIFFs
    target_crs="EPSG:4326"
    ds= xr.open_dataset(ref_file)['band_data']
    lonmin= ds.x.values.min()
    lonmax= ds.x.values.max()
    latmin= ds.y.values.min()
    latmax= ds.y.values.max()
    # print(geotiff_data[62])
    if len(list(idx.intersection((lonmin, latmin, lonmax, latmax))))>1:
        # return None
        ds2= xr.combine_by_coords([xr.open_dataset(geotiff_data[i]['path']) for i in idx.intersection((lonmin, latmin, lonmax, latmax))]).rio.reproject(ds.rio.crs)
    else:
        ds2= xr.open_dataset(geotiff_data[list(idx.intersection((lonmin, latmin, lonmax, latmax)))[0]]['path']).rio.reproject(ds.rio.crs)
    # print(ds.dims, ds2.dims)
    ds2_classified= map_rgb_to_class(ds2)
    ds2_resampled = ds2_classified.interp(x=ds.x, y=ds.y, method="nearest")
    return ds2_resampled


def make_lulc(input_dir, output_dir, ref_file):
    paths= sorted(glob("../dataset/HOU_LULC/HOU*.tif"))
    idx, geotiff_path = create_geotiff_index(paths)
    ds= get_value_from_geotiffs_indexed(idx, geotiff_path, ref_file)
    ds.squeeze().rio.to_raster(os.path.join(output_dir, f"{input_dir.split('/')[-1]}_LULC.tif"))
    


def crop_geotiff_overlap(input_file, output_dir, crop_size=1024, overlap=100):
    out_name = input_file.split("/")[-1].split(".")[0]
    with rasterio.open(input_file) as src:
        width = src.width
        height = src.height
        count = 1

        for i in range(0, height - crop_size+1, crop_size - overlap):
            for j in range(0, width - crop_size +1, crop_size - overlap):
                window = Window(j, i, crop_size, crop_size)
                transform = src.window_transform(window)
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": window.height,
                    "width": window.width,
                    "transform": transform
                })

                data = src.read(window=window).squeeze().astype(np.float32)

                # Output original data
                output_file = os.path.join(output_dir, f"{out_name}_{count:03d}.npy")
                with rasterio.open(output_file, "w", **out_meta) as dest:
                    dest.write(data[np.newaxis, :, :])
                count += 1

                # # Flip upside down and output
                flipped_ud = np.flipud(data)[np.newaxis, :, :]
                output_file = os.path.join(output_dir, f"{out_name}_{count:03d}.npy")
                with rasterio.open(output_file, "w", **out_meta) as dest:
                    dest.write(flipped_ud)
                count += 1

                # # Flip left to right and output
                flipped_lr = np.fliplr(data)[np.newaxis, :, :]
                output_file = os.path.join(output_dir, f"{out_name}_{count:03d}.npy")
                with rasterio.open(output_file, "w", **out_meta) as dest:
                    dest.write(flipped_lr)
                count += 1

                #both flip upside down and left to right and output
                flipped_up_lr = np.flipud(flipped_lr.squeeze())[np.newaxis, :, :]
                output_file = os.path.join(output_dir, f"{out_name}_{count:03d}.npy")
                with rasterio.open(output_file, "w", **out_meta) as dest:
                    dest.write(flipped_up_lr)
                count += 1
    print(f'{count} of tiles generated')

def crop_geotiff_random(input_flood, dem_input, input_lulc, group, crop_size=512, num_images=400, random_coords=None, min_distance=50):
    """
    Randomly crops an input GeoTIFF into multiple 1024x1024 images with flipping, ensuring coordinates are not too close.
    Save training data to cropped_data with .npy
    
    Parameters:
        input_file (str): Path to the input GeoTIFF file.
        output_dir (str): Path to the output directory.
        crop_size (int): Size of the cropped image (default: 1024).
        num_images (int): Number of random crops to generate if random_coords is None (default: 200).
        random_coords (list or None): List of coordinates [(i, j), ...] for cropping or None to generate randomly.
        min_distance (int): Minimum allowable distance between two crop coordinates (default: 100).

    Return:
        random_coords (list): Randomply generated coordinates to be passed as input so repeatable
    """
    out_name = input_file.split("/")[-1].split(".")[0]
    rainfall= int(input_file.split('_')[2].split('m')[0])

    with rasterio.open(input_file) as src:
        width = src.width
        height = src.height
        print(f'width: {width}, height: {height}')
        count = 1
        
        # Generate random coordinates if not provided
        if random_coords is None:
            random_coords = []
            while len(random_coords) < num_images:
                i = np.random.randint(BUFFER, height - crop_size-BUFFER)
                j = np.random.randint(BUFFER, width - crop_size-BUFFER)
                # Check if the new coordinate is far enough from all existing coordinates
                if all(np.sqrt((i - x) ** 2 + (j - y) ** 2) >= min_distance for x, y in random_coords):
                    random_coords.append((i, j))
        
        for i, j in random_coords:
            window = Window(j, i, crop_size, crop_size)
            transform = src.window_transform(window)
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": window.height,
                "width": window.width,
                "transform": transform
            })
        
            data = src.read(window=window).squeeze().astype(np.float32)
            with rasterio.open(dem_input) as f:
                dem= f.read(window=window).squeeze().astype(np.float32)

            with rasterio.open(input_lulc) as f:
                lulc=f.read(window=window).squeeze().astype(np.float32)
            arr_rain= np.ones(data.shape) * rainfall
            data= np.stack([data, arr_rain, dem, lulc])
            assert data.shape==(4,crop_size,crop_size), f'it has non valid values, {input_file}, its shape is {data.shape}'

            # Output original data
            city_rainfall= '_'.join(out_name.split('_')[:2])
            if city_rainfall not in group.keys():
                subgroup= group.create_group(city_rainfall)
            else:
                subgroup= group[city_rainfall]
            # output_file = os.path.join(output_dir, f"{out_name}_{count:03d}.npy")
            
            subgroup.create_dataset(f"{count:04d}", data=data, compression='gzip')
            # np.save(output_file, data)
            # with rasterio.open(output_file, "w", **out_meta) as dest:
            #     dest.write(data[np.newaxis, :, :])
            count += 1

            # Flip upside down and output
            # flipped_ud = data[:,::-1,:]
            # subgroup.create_dataset(f"{count:04d}", data=flipped_ud, compression='gzip')
            # output_file = os.path.join(output_dir, f"{out_name}_{count:03d}.npy")
            # with rasterio.open(output_file, "w", **out_meta) as dest:
            #     dest.write(flipped_ud)
            # np.save(output_file, flipped_ud)
            # count += 1
            # Flip left to right and output
            # flipped_lr = data[:,:,::-1]
            # subgroup.create_dataset(f"{count:04d}", data=flipped_lr, compression='gzip')
            # output_file = os.path.join(output_dir, f"{out_name}_{count:03d}.npy")
            
            # with rasterio.open(output_file, "w", **out_meta) as dest:
            #     dest.write(flipped_lr)
            # np.save(output_file, flipped_lr)
            # count += 1

            # Both flip upside down and left to right and output
            # flipped_up_lr = data[:,::-1,::-1]
            # output_file = os.path.join(output_dir, f"{out_name}_{count:03d}.npy")
            # subgroup.create_dataset(f"{count:04d}", data= flipped_up_lr, compression='gzip')

            # count += 1
            
    print(f"{count - 1} tiles generated and saved")
    return random_coords  # Return the used coordinates for reproducibility

def crop_maximum(input_dir, output_dir, rainfall_level=110):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # given a input directory, find the maximum flood inundation depth and crop it
    fnames= glob(f"{input_dir}/{input_dir.split('/')[-1].split('_')[0]}_{rainfall_level}*.tif")
    ds= xr.concat([xr.open_dataset(f) for f in fnames], dim='band_value').load()
    max_depth = ds.max(dim='band_value')
    # print(max_depth)
    max_depth.squeeze().rio.to_raster(os.path.join(output_dir, f"{input_dir.split('/')[-1]}_{rainfall_level}_max.tif"))


if __name__ == "__main__":
    with open("HOU_rainfall.json", "r") as f:
        data = json.load(f)
    # train_data= ['HOU002', 'HOU003', 'HOU004', 'HOU005', 'HOU006',
    #              'AUS001','DAL001','OKC001','OKC002','LA001','SF001',
    #              'NYC001','ATL001','ATL002','ORL001','ORL002','MIA001']
    # test_data= ['HOU007', 'AUS002','SF002']
    # val_data= ['DAL002','LA002','NYC002','MIA002']
    train_data = ['HOU002', 'HOU003', 'HOU004', 'HOU005', 'HOU006']
    val_data= ['HOU007']
    test_data= ['HOU007']

    for city in data:
        dataset = city['City ID']
        output_dir = f"../cropped_data/{dataset}"
        if dataset in train_data:
            H5= h5py.File('training_HOU_512_400.h5', 'a')
        elif dataset in test_data:
            H5= h5py.File('testing_HOU_512_400.h5', 'a')
        else:
            H5= h5py.File('validation_HOU_512_400.h5', 'a')
        group= H5
        print(f'####### Processing {dataset} #######')

        first = True
        for rainfall_level in [city['500-yr'], city['100-yr'], city['50-yr'], city['25-yr'], city['10-yr'], city['1-yr']]:
            rainfall_level = rainfall_level.replace(' ', '')
            group_name= dataset + '_' +rainfall_level
            if group_name in group.keys():
                print(group_name, 'already exists! move on...')
                continue
            else:
                input_dir = f'/home/users/li1995/global_flood/UrbanFloods2D/dataset/{dataset}'
                dem_input = os.path.join(input_dir, f"{dataset}_DEM.tif")
                input_file = f'/home/users/li1995/global_flood/UrbanFloods2D/sample/{dataset}_{rainfall_level}_max.tif'
                input_lulc = f'/home/users/li1995/global_flood/UrbanFloods2D/sample/{dataset}_LULC.tif'

                # crop_maximum(input_dir, '/home/users/li1995/global_flood/UrbanFloods2D/sample', rainfall_level)
                os.makedirs(output_dir, exist_ok=True)

                if first:
                    random_coords = crop_geotiff_random(input_file, dem_input, input_lulc, group)
                    first = False
                else:
                    crop_geotiff_random(input_file, dem_input, input_lulc, group, random_coords=random_coords)

        # make_lulc(input_dir, os.path.join('../sample'),input_file)
        # lulc_input= f'../sample/{dataset}_LULC.tif'
        # crop_geotiff_random(lulc_input, output_dir, random_coords=random_coords)
        # crop_lulc(output_dir, coords=random_coords, ref_file= input_file)

        
        # crop_geotiff_random(dem_input, output_dir, random_coords=random_coords)
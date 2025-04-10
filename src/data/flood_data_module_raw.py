import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import torchvision.transforms as transforms
import random
import json
import rasterio
from glob import glob

class RawFloodDataset(Dataset):
    def __init__(self, data_dir, max_flood_dir, split, transform=None, normalize=True, 
                cache_size=100, cities_file=None):
        """
        Dataset for flood prediction using raw TIF files with lazy loading.
        
        Args:
            data_dir (str): Path to the directory containing DEM files
            max_flood_dir (str): Path to the directory containing max flood depth files
            split (str): Data split ('train', 'val', 'test')
            transform: Optional transforms to apply
            normalize (bool): Whether to normalize the data
            stats_file (str): Path to JSON file containing normalization statistics
            cache_size (int): Number of samples to cache in memory
            cities_file (str): Path to the JSON file containing city-rainfall data
        """
        self.data_dir = data_dir
        self.max_flood_dir = max_flood_dir
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.cache_size = cache_size
        self.cache = {}  # LRU cache for samples
        
        # Load city-rainfall data to know which cities & rainfall values to use
        self.cities_file = cities_file or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cities_rainfall.json')
        with open(self.cities_file, 'r') as f:
            self.cities_data = json.load(f)
        
        # Define which cities to use for each split
        train_cities = ['HOU002', 'HOU003', 'HOU004', 'HOU005', 'HOU006',
                       'AUS001', 'DAL001', 'OKC001', 'OKC002', 'LA001', 'SF001',
                       'NYC001', 'ATL001', 'ATL002', 'ORL001', 'ORL002', 'MIA001']
        test_cities = ['HOU007', 'AUS002', 'SF002']
        val_cities = ['DAL002', 'LA002', 'NYC002', 'MIA002']
        
        # Select cities for the current split
        if split == 'train':
            self.cities = train_cities
        elif split == 'val':
            self.cities = val_cities
        elif split == 'test':
            self.cities = test_cities
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'.")
        
        # Prepare the list of samples (dem_file, rainfall_value, max_flood_file)
        self.samples = []
        self._prepare_samples()
        
    
    def _prepare_samples(self):
        """Prepare the list of samples by collecting all available data files."""
        for city_id in self.cities:
            # Get rainfall values for this city
            city_rainfall_data = next((city for city in self.cities_data if city['City ID'] == city_id), None)
            if not city_rainfall_data:
                print(f"Warning: No rainfall data found for {city_id}, skipping.")
                continue
            
            # DEM file path
            dem_file = os.path.join(self.data_dir, city_id, f"{city_id}_DEM.tif")
            if not os.path.exists(dem_file):
                print(f"Warning: DEM file not found for {city_id}: {dem_file}, skipping.")
                continue
            
            # For each rainfall level, check if the max flood depth file exists
            for rainfall_type in ['100-yr', '50-yr', '25-yr', '10-yr']:
                rainfall_value = city_rainfall_data.get(rainfall_type, '').replace(' ', '')
                if not rainfall_value:
                    continue
                
                max_depth_file = os.path.join(self.max_flood_dir, f"{city_id}_{rainfall_value}_max.tif")
                if not os.path.exists(max_depth_file):
                    print(f"Warning: Max depth file not found: {max_depth_file}, skipping.")
                    continue
                
                # Store the sample as a tuple of (dem_file, rainfall_value, max_depth_file)
                self.samples.append((
                    dem_file,
                    float(rainfall_value.replace('mm', '')),
                    max_depth_file,
                    city_id  # Include city_id for normalization purposes
                ))
    
    def __len__(self):
        return len(self.samples)
    
    
    def normalize_data(self, dem, max_depth, rainfall, city_id):
        """Normalize the data using city-specific statistics."""
        
        stats = self.city_stats[city_id]
        
        # Normalize DEM
        dem_normalized = (dem - dem.mean()) / (dem.std() + 1e-8)
        
        # Normalize flood depth
        depth_normalized = torch.log(max_depth+1e-4) / 2.
        
        # Rainfall is uniform across domain, no normalization needed
        rainfall_normalized = rainfall / 1000.
        
        return dem_normalized, depth_normalized, rainfall_normalized

    
    def __getitem__(self, idx):
        """Get a sample from the dataset with lazy loading."""
        # Check if sample is in cache
        if idx in self.cache:
            return self.cache[idx]
        
        # Get sample information
        dem_file, rainfall_value, max_depth_file, city_id = self.samples[idx]
        
        # Read entire DEM and max flood depth files
        with rasterio.open(dem_file) as src:
            dem = src.read(1)
        
        with rasterio.open(max_depth_file) as src:
            max_depth = src.read(1)
        
        # Convert to torch tensors
        dem_tensor = torch.from_numpy(dem).float().unsqueeze(0)
        max_depth_tensor = torch.from_numpy(max_depth).float().unsqueeze(0)
        rainfall_tensor = torch.tensor(rainfall_value).float().unsqueeze(0)
        
        # Normalize if required
        if self.normalize:
            dem_normalized, depth_normalized, rainfall_normalized = self.normalize_data(
                dem_tensor, max_depth_tensor, rainfall_tensor, city_id
            )
            dem_tensor = dem_normalized
            max_depth_tensor = depth_normalized
            rainfall_tensor = rainfall_normalized
        
        # Create input tensor
        rainfall_channel = torch.ones_like(dem_tensor) * rainfall_tensor
        input_tensor = torch.cat([dem_tensor, rainfall_channel], dim=0)
        
        # Create result tuple
        result = (input_tensor, max_depth_tensor)
        
        # Update cache (simple LRU implementation)
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            self.cache.pop(next(iter(self.cache)))
        self.cache[idx] = result
        
        return result

class RawFloodDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir="/home/users/li1995/global_flood/UrbanFloods2D/dataset", 
        max_flood_dir="/home/users/li1995/global_flood/UrbanFloods2D/sample", 
        batch_size=16, 
        num_workers=4, 
        normalize=True, 
        prefetch_factor=2, 
        persistent_workers=True,
        image_size=512
    ):
        """
        PyTorch Lightning data module for flood prediction with raw TIF files.
        
        Args:
            data_dir (str): Path to the directory containing DEM files
            max_flood_dir (str): Path to the directory containing max flood depth files
            batch_size (int): Batch size
            num_workers (int): Number of workers for data loading
            normalize (bool): Whether to normalize the data
            stats_file (str): Path to JSON file for normalization statistics
            prefetch_factor (int): Number of batches to prefetch per worker
            persistent_workers (bool): Keep worker processes alive between dataloader calls
            image_size (int): Size of images to crop to (assuming square images)
        """
        super().__init__()
        self.data_dir = data_dir
        self.max_flood_dir = max_flood_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.image_size = image_size
        
        # Define transforms for data augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        
        # Define transforms for validation and testing (just cropping, no augmentation)
        self.eval_transform = transforms.Compose([
            transforms.RandomCrop(image_size),
        ])
    
    def setup(self, stage=None):
        """Set up the datasets for each stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = RawFloodDataset(
                data_dir=self.data_dir,
                max_flood_dir=self.max_flood_dir,
                split='train',
                transform=self.train_transform,  # Apply augmentation to training data
                normalize=self.normalize,
            )
            
            # Save normalization stats after calculating them on training data
            if self.normalize and not self.stats_file:
                self.stats_file = f'raw_normalization_stats.json'
                self.train_dataset.save_normalization_stats(self.stats_file)
            
            self.val_dataset = RawFloodDataset(
                data_dir=self.data_dir,
                max_flood_dir=self.max_flood_dir,
                split='val',
                transform=self.eval_transform,  # Only crop for validation
                normalize=self.normalize,
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = RawFloodDataset(
                data_dir=self.data_dir,
                max_flood_dir=self.max_flood_dir,
                split='test',
                transform=self.eval_transform,  # Only crop for testing
                normalize=self.normalize,
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=self.persistent_workers
        )
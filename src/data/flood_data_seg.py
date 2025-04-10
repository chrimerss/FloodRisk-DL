import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import torchvision.transforms as transforms
import random
import json

class FloodDataset(Dataset):
    def __init__(self, data_dir, split, image_size=512, num_images=400, transform=None, normalize=True, cache_size=100):
        """
        Dataset for flood prediction using H5 file with lazy loading.
        
        Args:
            h5_file (str): Path to the H5 file
            split (str): Data split ('train', 'val', 'test')
            transform: Optional transforms to apply
            normalize (bool): Whether to normalize the data
            stats_file (str): Path to JSON file containing normalization statistics
            cache_size (int): Number of samples to cache in memory
        """
        
        mapper= {'train':'training', 'test':'testing', 'val':'validation'}
        self.split = mapper[split]
        self.h5_file = os.path.join(data_dir, f'{self.split}_{image_size}_{num_images}.h5')
        self.transform = transform
        self.normalize = normalize
        self.cache_size = cache_size
        self.cache = {}  # LRU cache for samples
        self.num_images= num_images
        
        # Don't open the file here, just check it exists
        if not os.path.exists(self.h5_file):
            raise FileNotFoundError(f"H5 file not found: {h5_file}")
        
        # Temporarily open the file to get metadata
        with h5py.File(self.h5_file, 'r') as h5:
            self.samples = []
            
            # Iterate through all city-rainfall subgroups
            for subgroup_name in h5.keys():
                # Get number of tiles in this subgroup
                num_tiles = num_images
                
                # Add each tile to samples list
                for tile_idx in range(1, num_tiles + 1):
                    self.samples.append({
                        'subgroup_name': subgroup_name,
                        'tile_idx': tile_idx
                    })
        
        # Load normalization stats
        # if normalize:
            ## I decided to use log transformation instead of standardized
            # if stats_file and os.path.exists(stats_file):
            #     self.load_normalization_stats(stats_file)
            # else:
                # self.calculate_normalization_stats()
    
    def __len__(self):
        return len(self.samples)
    
    def calculate_normalization_stats(self):
        """Calculate normalization statistics for each city with minimal memory usage."""
        self.city_stats = {}
        
        # Process smaller batches to reduce memory usage
        for split in ['training', 'validation', 'testing']:
            h5_file= '/'.join(self.h5_file.split('/')[:-1])+'/'+split+'.h5'
            with h5py.File(h5_file, 'r') as h5:
              # Only use training data for stats
                # Get unique city IDs from subgroup names
                city_ids = set(subgroup_name.split('_')[0] for subgroup_name in h5.keys())
                
                for city_id in city_ids:
                    # Initialize online statistics calculation
                    dem_sum = 0
                    dem_sum_sq = 0
                    depth_sum = 0
                    depth_sum_sq = 0
                    dem_count = 0
                    depth_count = 0
                    
                    # Get all subgroups for this city
                    city_subgroups = [name for name in h5.keys() 
                                  if name.startswith(f"{city_id}_")]
                    
                    # Sample a subset for statistics (fewer samples to save memory)
                    sample_indices = np.linspace(0, len(city_subgroups) - 1, 
                                          min(10, len(city_subgroups)), dtype=int)
                    
                    for idx in sample_indices:
                        subgroup_name = city_subgroups[idx]
                        subgroup = h5[subgroup_name]
                        
                        # Sample fewer tiles
                        num_tiles = self.num_images
                        tile_indices = np.linspace(1, num_tiles, min(3, num_tiles), dtype=int)
                        
                        for tile_idx in tile_indices:
                            # Get data in chunks to reduce memory
                            combined_data = subgroup[f'{tile_idx:04d}'][:]
                            
                            # Process DEM data
                            dem_data = combined_data[2]
                            dem_sum += np.sum(dem_data)
                            dem_sum_sq += np.sum(dem_data ** 2)
                            dem_count += dem_data.size
                            
                            # Process depth data
                            depth_data = combined_data[0]
                            depth_sum += np.sum(depth_data)
                            depth_sum_sq += np.sum(depth_data ** 2)
                            depth_count += depth_data.size
                    
                    # Calculate statistics
                    dem_mean = dem_sum / dem_count
                    dem_std = np.sqrt((dem_sum_sq / dem_count) - (dem_mean ** 2))
                    
                    depth_mean = depth_sum / depth_count
                    depth_std = np.sqrt((depth_sum_sq / depth_count) - (depth_mean ** 2))
                    
                    # Store statistics for this city
                    self.city_stats[city_id] = {
                        'dem_mean': float(dem_mean),
                        'dem_std': float(dem_std),
                        'depth_mean': float(depth_mean),
                        'depth_std': float(depth_std)
                    }
    
    
    def normalize_data(self, dem, max_depth, rainfall, city_id):
        """Normalize the data using city-specific statistics."""
        
        # stats = self.city_stats[city_id]
        
        # Normalize DEM
        dem_normalized = (dem - dem.mean()) / (dem.std() + 1e-8)
        
        # Normalize flood depth
        # depth_normalized = max_depth / (stats['depth_std'] + 1e-8)  # Don't subtract mean for depth
        # depth_normalized= torch.log(max_depth + 1e-4) / 2.0
        
        # Rainfall is uniform across domain, no normalization needed
        rainfall_normalized = rainfall / 1000.
        
        return dem_normalized, max_depth, rainfall_normalized
    
    
    def __getitem__(self, idx):
        """Get a sample from the dataset with lazy loading."""
        # Check if sample is in cache
        if idx in self.cache:
            return self.cache[idx]
        
        # Open file only when needed
        with h5py.File(self.h5_file, 'r') as h5:
            sample = self.samples[idx]
            subgroup_name = sample['subgroup_name']
            tile_idx = sample['tile_idx']
            
            # Extract city ID from subgroup name
            city_id = subgroup_name.split('_')[0]
            
            # Get combined data (2, 1024, 1024)
            combined_data = h5[subgroup_name][f'{tile_idx:04d}'][:] 
            
            # Get rainfall value
        rainfall_value = combined_data[1,:,:]
        
        # Split into DEM and flood depth
        dem = combined_data[2,:,:]
        max_depth = combined_data[0]

        #segmentate flood_depth
        categories = np.zeros_like(max_depth, dtype=np.int32)
        categories = np.where(max_depth >= 0.1, 1, categories)
        categories = np.where(max_depth >= 0.2, 2, categories)
        categories = np.where(max_depth >= 0.3, 3, categories)
        categories = np.where(max_depth >= 0.5, 4, categories)
        categories = np.where(max_depth >= 1.0, 5, categories)
        
        # Convert to torch tensors
        dem_tensor = torch.from_numpy(dem).float().unsqueeze(0)
        flood_cat_tensor = torch.from_numpy(categories).float().unsqueeze(0)
        rainfall_tensor = torch.tensor(rainfall_value).float().unsqueeze(0)
        
        # Normalize if required
        if self.normalize:
            dem_normalized, depth_normalized, rainfall_normalized = self.normalize_data(
                dem_tensor, flood_cat_tensor, rainfall_tensor, city_id
            )
            dem_tensor = dem_normalized
            flood_cat_tensor = depth_normalized
            rainfall_tensor = rainfall_normalized
        
        # Create input tensor
        rainfall_channel = torch.ones_like(dem_tensor) * rainfall_tensor
        input_tensor = torch.cat([dem_tensor, rainfall_channel], dim=0)
        
        # Apply transforms if available
        if self.transform:
            input_tensor = self.transform(input_tensor)
        
        # Create result tuple
        result = (input_tensor, flood_cat_tensor)
        
        # Update cache (simple LRU implementation)
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            self.cache.pop(next(iter(self.cache)))
        self.cache[idx] = result
        
        return result

class FloodDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, image_size=512, num_images=400, num_workers=4, normalize=True, stats_file=None, 
                 prefetch_factor=2, persistent_workers=True):
        """
        PyTorch Lightning data module for flood prediction with optimized loading.
        
        Args:
            h5_file (str): Path to the H5 file
            batch_size (int): Batch size
            num_workers (int): Number of workers for data loading
            normalize (bool): Whether to normalize the data
            prefetch_factor (int): Number of batches to prefetch per worker
            persistent_workers (bool): Keep worker processes alive between dataloader calls
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.image_size= image_size
        self.num_images= num_images
        
        # Define transforms
        # self.train_transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.RandomVerticalFlip(),
        # ])
    
    def setup(self, stage=None):
        """Set up the datasets for each stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = FloodDataset(
                data_dir=self.data_dir,
                image_size=self.image_size,
                num_images=self.num_images,                 
                split='train',
                transform=None,
                normalize=self.normalize,
            )
            
            # Save normalization stats after calculating them on training data
            
            self.val_dataset = FloodDataset(
                data_dir=self.data_dir,
                image_size=self.image_size,
                num_images=self.num_images,  
                split='val',
                transform=None,  # No augmentation for validation
                normalize=self.normalize,
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = FloodDataset(
                data_dir=self.data_dir,
                image_size=self.image_size,
                num_images=self.num_images,               
                split='test',
                transform=None,  # No augmentation for testing
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
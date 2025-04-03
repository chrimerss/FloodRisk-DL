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
    def __init__(self, h5_file, split, transform=None, normalize=True, stats_file=None):
        """
        Dataset for flood prediction using H5 file.
        
        Args:
            h5_file (str): Path to the H5 file
            split (str): Data split ('train', 'val', 'test')
            transform: Optional transforms to apply
            normalize (bool): Whether to normalize the data
            stats_file (str): Path to JSON file containing normalization statistics
        """
        self.h5_file = h5_file
        self.split = split
        self.transform = transform
        self.normalize = normalize
        
        # Open the H5 file and keep it open
        self.h5 = h5py.File(h5_file, 'r')
        
        if split not in self.h5:
            self.h5.close()
            raise ValueError(f"Split '{split}' not found in the H5 file")
        
        self.samples = []
        
        # Iterate through all city-rainfall subgroups
        for subgroup_name in self.h5[split]:
            subgroup = self.h5[split][subgroup_name]
            
            # Get number of tiles in this subgroup
            num_tiles = subgroup.attrs['num_tiles']
            
            # Add each tile to samples list
            for tile_idx in range(1, num_tiles + 1):
                self.samples.append({
                    'subgroup_name': subgroup_name,
                    'tile_idx': tile_idx
                })
        
        # Load or calculate normalization stats
        if normalize:
            if stats_file and os.path.exists(stats_file):
                self.load_normalization_stats(stats_file)
            else:
                self.calculate_normalization_stats()
    
    def __del__(self):
        """Clean up by closing the H5 file."""
        if hasattr(self, 'h5'):
            self.h5.close()
    
    def __len__(self):
        return len(self.samples)
    
    def calculate_normalization_stats(self):
        """Calculate normalization statistics for each city."""
        self.city_stats = {}
        for split in ['train','test','val']:
            # Get unique city IDs from subgroup names
            city_ids = set(subgroup_name.split('_')[0] for subgroup_name in self.h5[split].keys())
            
            for city_id in city_ids:
                dem_values = []
                depth_values = []
                
                # Get all subgroups for this city
                city_subgroups = [name for name in self.h5[split].keys() 
                                if name.startswith(f"{city_id}_")]
                
                # Sample a subset of data for calculating statistics
                sample_indices = np.linspace(0, len(city_subgroups) - 1, 
                                        min(20, len(city_subgroups)), dtype=int)
                
                for idx in sample_indices:
                    subgroup_name = city_subgroups[idx]
                    subgroup = self.h5[split][subgroup_name]
                    
                    # Sample a few tiles from this subgroup
                    num_tiles = subgroup.attrs['num_tiles']
                    tile_indices = np.linspace(1, num_tiles, min(5, num_tiles), dtype=int)
                    
                    for tile_idx in tile_indices:
                        # Get combined data
                        combined_data = subgroup[f'tile_{tile_idx:06d}'][:]
                        
                        # Split into DEM and flood depth
                        dem_values.append(combined_data[0].flatten())
                        depth_values.append(combined_data[1].flatten())
                
                # Concatenate and calculate statistics
                dem_values = np.concatenate(dem_values)
                depth_values = np.concatenate(depth_values)
                
                # Store statistics for this city
                self.city_stats[city_id] = {
                    'dem_mean': float(dem_values.mean()),
                    'dem_std': float(dem_values.std()),
                    'depth_mean': float(depth_values.mean()),
                    'depth_std': float(depth_values.std())
                }
    
    def save_normalization_stats(self, stats_file):
        """Save normalization statistics to a JSON file."""
        with open(stats_file, 'w') as f:
            json.dump(self.city_stats, f, indent=4)
    
    def load_normalization_stats(self, stats_file):
        """Load normalization statistics from a JSON file."""
        with open(stats_file, 'r') as f:
            self.city_stats = json.load(f)
    
    def normalize_data(self, dem, max_depth, rainfall, city_id):
        """Normalize the data using city-specific statistics."""
        
        stats = self.city_stats[city_id]
        
        # Normalize DEM
        dem_normalized = (dem - stats['dem_mean']) / (stats['dem_std'] + 1e-8)
        
        # Normalize flood depth
        depth_normalized = max_depth / (stats['depth_std'] + 1e-8)  # Don't subtract mean for depth
        
        # Rainfall is uniform across domain, no normalization needed
        rainfall_normalized = rainfall
        
        return dem_normalized, depth_normalized, rainfall_normalized
    
    def denormalize_data(self, dem_normalized, depth_normalized, rainfall_normalized, city_id):
        """Denormalize the data using city-specific statistics."""
        stats = self.city_stats[city_id]
        
        # Denormalize DEM
        dem = dem_normalized * (stats['dem_std'] + 1e-8) + stats['dem_mean']
        
        # Denormalize flood depth
        depth = depth_normalized * (stats['depth_std'] + 1e-8)
        
        # Rainfall is uniform across domain, no denormalization needed
        rainfall = rainfall_normalized
        
        return dem, depth, rainfall
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        subgroup_name = sample['subgroup_name']
        tile_idx = sample['tile_idx']
        
        # Extract city ID from subgroup name
        city_id = subgroup_name.split('_')[0]
        
        # Get combined data (2, 1024, 1024)
        combined_data = self.h5[self.split][subgroup_name][f'tile_{tile_idx:06d}'][:]
        
        # Get rainfall value from subgroup attributes
        rainfall_value = self.h5[self.split][subgroup_name].attrs['rainfall_value']
        
        # Get tile position info
        tile_info = self.h5[self.split][subgroup_name][f'tile_info_{tile_idx:06d}'][:]
        
        # Split into DEM and flood depth
        dem = combined_data[0]
        max_depth = combined_data[1]
        
        # Convert to torch tensors
        dem_tensor = torch.from_numpy(dem).float().unsqueeze(0)  # Add channel dimension
        max_depth_tensor = torch.from_numpy(max_depth).float().unsqueeze(0)  # Add channel dimension
        rainfall_tensor = torch.tensor(rainfall_value).float()
        
        # Normalize if required
        if self.normalize:
            dem_normalized, depth_normalized, rainfall_normalized = self.normalize_data(
                dem_tensor, max_depth_tensor, rainfall_tensor, city_id
            )
            dem_tensor = dem_normalized
            max_depth_tensor = depth_normalized
            rainfall_tensor = rainfall_normalized
        
        # Create input tensor (2, 1024, 1024)
        rainfall_channel = torch.ones_like(dem_tensor) * rainfall_tensor
        input_tensor = torch.cat([dem_tensor, rainfall_channel], dim=0)
        
        # Apply transforms if available
        if self.transform:
            input_tensor = self.transform(input_tensor)
        
        # Return the input, target, and metadata
        return input_tensor, max_depth_tensor

class FloodDataModule(pl.LightningDataModule):
    def __init__(self, h5_file, batch_size=16, num_workers=4, normalize=True, stats_file=None):
        """
        PyTorch Lightning data module for flood prediction.
        
        Args:
            h5_file (str): Path to the H5 file
            batch_size (int): Batch size
            num_workers (int): Number of workers for data loading
            normalize (bool): Whether to normalize the data
            stats_file (str): Path to JSON file containing normalization statistics
        """
        super().__init__()
        self.h5_file = h5_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.stats_file = stats_file
        
        # Define transforms
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Add more augmentations if needed
        ])
    
    def setup(self, stage=None):
        """Set up the datasets for each stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = FloodDataset(
                h5_file=self.h5_file,
                split='train',
                transform=self.train_transform,
                normalize=self.normalize,
                stats_file=self.stats_file
            )
            
            # Save normalization stats after calculating them on training data
            if self.normalize and not self.stats_file:
                self.stats_file = 'normalization_stats.json'
                self.train_dataset.save_normalization_stats(self.stats_file)
            
            self.val_dataset = FloodDataset(
                h5_file=self.h5_file,
                split='val',
                transform=None,  # No augmentation for validation
                normalize=self.normalize,
                stats_file=self.stats_file
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = FloodDataset(
                h5_file=self.h5_file,
                split='test',
                transform=None,  # No augmentation for testing
                normalize=self.normalize,
                stats_file=self.stats_file
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        ) 
import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pathlib import Path
import torchvision.transforms as transforms

class FloodDataset(Dataset):
    def __init__(self, h5_file, split, transform=None, normalize=True):
        """
        Dataset for flood prediction using H5 file.
        
        Args:
            h5_file (str): Path to the H5 file
            split (str): Data split ('train', 'val', 'test')
            transform: Optional transforms to apply
            normalize (bool): Whether to normalize the data
        """
        self.h5_file = h5_file
        self.split = split
        self.transform = transform
        self.normalize = normalize
        
        # Open the H5 file to count samples and build an index
        with h5py.File(h5_file, 'r') as f:
            if split not in f:
                raise ValueError(f"Split '{split}' not found in the H5 file")
            
            self.samples = []
            
            # Build an index of all samples
            for city_id in f[split]:
                city_group = f[split][city_id]
                
                # Skip if DEM data is not available
                if 'dem' not in city_group:
                    continue
                
                # Process each rainfall scenario
                for rainfall_type in city_group:
                    if rainfall_type == 'dem':  # Skip the DEM dataset itself
                        continue
                    
                    # Add to samples list
                    self.samples.append({
                        'city_id': city_id,
                        'rainfall_type': rainfall_type
                    })
        
        # Calculate normalization stats for DEM and max flood depth if needed
        if normalize:
            self.calculate_normalization_stats()
    
    def __len__(self):
        return len(self.samples)
    
    def calculate_normalization_stats(self):
        """Calculate normalization statistics for the dataset."""
        dem_values = []
        depth_values = []
        rainfall_values = []
        
        # Sample a subset of data for calculating statistics
        sample_indices = np.linspace(0, len(self.samples) - 1, min(100, len(self.samples)), dtype=int)
        
        with h5py.File(self.h5_file, 'r') as f:
            for idx in sample_indices:
                sample = self.samples[idx]
                city_id = sample['city_id']
                rainfall_type = sample['rainfall_type']
                
                # Get DEM data
                dem_data = f[self.split][city_id]['dem'][:]
                dem_values.append(dem_data.flatten())
                
                # Get max flood depth data
                max_depth = f[self.split][city_id][rainfall_type]['max_flood_depth'][:]
                depth_values.append(max_depth.flatten())
                
                # Get rainfall value
                rainfall_value = f[self.split][city_id][rainfall_type].attrs['rainfall_value']
                rainfall_values.append(rainfall_value)
        
        # Concatenate and calculate statistics
        dem_values = np.concatenate(dem_values)
        depth_values = np.concatenate(depth_values)
        rainfall_values = np.array(rainfall_values)
        
        # Calculate mean and std for normalization
        self.dem_mean = dem_values.mean()
        self.dem_std = dem_values.std()
        self.depth_mean = depth_values.mean()
        self.depth_std = depth_values.std()
        self.rainfall_mean = rainfall_values.mean()
        self.rainfall_std = rainfall_values.std()
    
    def normalize_data(self, dem, max_depth, rainfall):
        """Normalize the data using calculated statistics."""
        # Normalize DEM
        dem_normalized = (dem - self.dem_mean) / (self.dem_std + 1e-8)
        
        # Normalize flood depth
        depth_normalized = max_depth / (self.depth_std + 1e-8)  # Don't subtract mean for depth
        
        # Normalize rainfall
        rainfall_normalized = (rainfall - self.rainfall_mean) / (self.rainfall_std + 1e-8)
        
        return dem_normalized, depth_normalized, rainfall_normalized
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.samples[idx]
        city_id = sample['city_id']
        rainfall_type = sample['rainfall_type']
        
        with h5py.File(self.h5_file, 'r') as f:
            # Get DEM data
            dem = f[self.split][city_id]['dem'][:]
            
            # Get max flood depth data
            max_depth = f[self.split][city_id][rainfall_type]['max_flood_depth'][:]
            
            # Get rainfall value
            rainfall_value = f[self.split][city_id][rainfall_type].attrs['rainfall_value']
        
        # Convert to torch tensors
        dem_tensor = torch.from_numpy(dem).float().unsqueeze(0)  # Add channel dimension
        max_depth_tensor = torch.from_numpy(max_depth).float().unsqueeze(0)  # Add channel dimension
        rainfall_tensor = torch.tensor(rainfall_value).float()
        
        # Normalize if required
        if self.normalize:
            dem_normalized, depth_normalized, rainfall_normalized = self.normalize_data(
                dem_tensor, max_depth_tensor, rainfall_tensor
            )
            dem_tensor = dem_normalized
            max_depth_tensor = depth_normalized
            rainfall_tensor = rainfall_normalized
        
        # Apply transforms if available
        if self.transform:
            # Create a 3-channel input by repeating DEM and adding rainfall as a constant channel
            rainfall_channel = torch.ones_like(dem_tensor) * rainfall_tensor
            combined_input = torch.cat([dem_tensor, rainfall_channel], dim=0)
            
            if self.transform:
                combined_input = self.transform(combined_input)
        else:
            # Create a 2-channel input (DEM and rainfall)
            rainfall_channel = torch.ones_like(dem_tensor) * rainfall_tensor
            combined_input = torch.cat([dem_tensor, rainfall_channel], dim=0)
        
        # Return the input and target
        return combined_input, max_depth_tensor

class FloodDataModule(pl.LightningDataModule):
    def __init__(self, h5_file, batch_size=16, num_workers=4, normalize=True):
        """
        PyTorch Lightning data module for flood prediction.
        
        Args:
            h5_file (str): Path to the H5 file
            batch_size (int): Batch size
            num_workers (int): Number of workers for data loading
            normalize (bool): Whether to normalize the data
        """
        super().__init__()
        self.h5_file = h5_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        
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
                normalize=self.normalize
            )
            
            self.val_dataset = FloodDataset(
                h5_file=self.h5_file,
                split='val',
                transform=None,  # No augmentation for validation
                normalize=self.normalize
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = FloodDataset(
                h5_file=self.h5_file,
                split='test',
                transform=None,  # No augmentation for testing
                normalize=self.normalize
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
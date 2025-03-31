import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import rasterio
from sklearn.model_selection import train_test_split

class FloodDataset(Dataset):
    def __init__(self, data_paths, transform=None):
        """
        Placeholder dataset class for flood prediction
        
        Args:
            data_paths (list): List of tuples containing (rainfall_path, dem_path, label)
            transform: Optional transforms to apply
        """
        self.data_paths = data_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        rainfall_path, dem_path, label = self.data_paths[idx]
        
        # Placeholder: In a real implementation, you would load the data from paths
        # For now, we'll just create random tensors
        rainfall = torch.rand(1, 224, 224)  # Channel, Height, Width
        dem = torch.rand(1, 224, 224)
        
        # Stack input features
        data = torch.cat([rainfall, dem], dim=0)
        
        if self.transform:
            data = self.transform(data)
            
        return data, label

class FloodDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_dir = config.data.data_dir
        self.batch_size = config.data.batch_size
        self.num_workers = config.data.num_workers
        
    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU."""
        # Placeholder for data download/preparation
        pass
        
    def setup(self, stage=None):
        """
        Load and split dataset for training, validation, and testing.
        
        Args:
            stage (str): 'fit' or 'test'
        """
        if stage == 'fit' or stage is None:
            # Placeholder dataset creation
            # In a real implementation, you would scan the data directory for rainfall and DEM files
            # and create proper datasets
            
            # For simplicity, create dummy data paths
            n_samples = 1000
            all_data_paths = [
                (f"rainfall_{i}.tif", f"dem_{i}.tif", np.random.randint(0, 2))  # Binary labels
                for i in range(n_samples)
            ]
            
            # Split data
            train_val_paths, test_paths = train_test_split(
                all_data_paths,
                test_size=self.config.data.test_ratio,
                random_state=42
            )
            
            train_paths, val_paths = train_test_split(
                train_val_paths,
                test_size=self.config.data.val_ratio / (1 - self.config.data.test_ratio),
                random_state=42
            )
            
            # Create datasets
            self.train_dataset = FloodDataset(train_paths)
            self.val_dataset = FloodDataset(val_paths)
            self.test_dataset = FloodDataset(test_paths)
        
        if stage == 'test' or stage is None:
            # If not already initialized in 'fit'
            if not hasattr(self, 'test_dataset'):
                # Create dummy test data
                test_paths = [
                    (f"rainfall_test_{i}.tif", f"dem_test_{i}.tif", np.random.randint(0, 2))
                    for i in range(200)
                ]
                self.test_dataset = FloodDataset(test_paths)
    
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
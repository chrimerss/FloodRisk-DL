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
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from torchgeo.datasets import NonGeoDataset
# Make sure TerraTorch imports are correctly defined
try:
    from terratorch.datasets.generic_pixel_wise_dataset import GenericNonGeoSegmentationDataset
except ImportError:
    print("Warning: Could not import GenericNonGeoSegmentationDataset from terratorch")

import albumentations as A
from albumentations.pytorch import ToTensorV2


class H5FloodSegmentationDataset(NonGeoDataset):
    """
    Dataset for flood segmentation using H5 files, compatible with TerraTorch.
    """
    
    def __init__(
        self,
        data_dir: Path,
        split: str,
        num_classes: int = 6,
        image_size: int = 512,
        num_images: int = 400,
        transform: Optional[A.Compose] = None,
        normalize: bool = True,
        cache_size: int = 100,
        rgb_indices: List[int] = None
    ) -> None:
        """
        Args:
            data_dir (Path): Path to directory containing H5 files
            split (str): Data split ('train', 'val', 'test')
            num_classes (int): Number of segmentation classes
            image_size (int): Image size
            num_images (int): Number of images per city-rainfall pair
            transform (A.Compose, optional): Albumentations transforms to apply
            normalize (bool): Whether to normalize the data
            cache_size (int): Number of samples to cache in memory
            rgb_indices (List[int]): Indices for RGB visualization (for TerraTorch compatibility)
        """
        super().__init__()
        
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
            
        mapper = {'train': 'training', 'val': 'validation', 'test': 'testing'}
        self.split_name = mapper[split]
        self.h5_file = os.path.join(data_dir, f'{self.split_name}_HOU_{image_size}_{num_images}.h5')
        self.transform = transform
        self.normalize = normalize
        self.cache_size = cache_size
        self.cache = {}  # LRU cache for samples
        self.num_images = num_images
        self.num_classes = num_classes
        self.rgb_indices = [0, 1, 11] if rgb_indices is None else rgb_indices  # DEM, rainfall, LULC
        
        # Don't open the file here, just check it exists
        if not os.path.exists(self.h5_file):
            raise FileNotFoundError(f"H5 file not found: {self.h5_file}")
        
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
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset with lazy loading in TerraTorch format."""
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
            
            # Get combined data
            combined_data = h5[subgroup_name][f'{tile_idx:04d}'][:]
            
            # Split into components
            max_depth = combined_data[0]
            rainfall_value = combined_data[1,:,:]
            dem = combined_data[2,:,:]
            lulc = combined_data[3,:,:]
            
            # Segmentate flood_depth into classes
            categories = np.zeros_like(max_depth, dtype=np.int32)
            categories = np.where(max_depth >= 0.1, 1, categories)
            categories = np.where(max_depth >= 0.2, 2, categories)
            categories = np.where(max_depth >= 0.3, 3, categories)
            categories = np.where(max_depth >= 0.5, 4, categories)
            categories = np.where(max_depth >= 1.0, 5, categories)
            
            # Convert to torch tensors
            dem_tensor = torch.from_numpy(dem).float().unsqueeze(0)
            rainfall_tensor = torch.from_numpy(rainfall_value).float().unsqueeze(0)
            
            # One-hot encode LULC
            lulc_tensor = torch.from_numpy(lulc).float()
            lulc_one_hot_tensor = F.one_hot(lulc_tensor.long(), num_classes=10)
            lulc_one_hot_tensor = lulc_one_hot_tensor.permute(2, 0, 1).float()
            
            # Normalize if required
            if self.normalize:
                dem_normalized = (dem_tensor - dem_tensor.mean()) / (dem_tensor.std() + 1e-8)
                dem_tensor = dem_normalized
                rainfall_tensor = rainfall_tensor / 1000.
            
            # Create input tensor (12 channels total)
            input_tensor = torch.cat([dem_tensor, rainfall_tensor, lulc_one_hot_tensor], dim=0)
            
            # Move channels from first dimension to last (TerraTorch expects channels last)
            image = input_tensor.permute(1, 2, 0).numpy()
            mask = categories
            
            # Create result dictionary in TerraTorch format
            result = {
                "image": image,  # Shape: (H, W, C)
                "mask": mask,    # Shape: (H, W)
                "filename": f"{subgroup_name}_{tile_idx:04d}"
            }
            
            # Apply transforms if available (Albumentations expects channels last)
            if self.transform:
                result = self.transform(**result)
            
            # Update cache (simple LRU implementation)
            if len(self.cache) >= self.cache_size:
                # Remove oldest item
                self.cache.pop(next(iter(self.cache)))
            self.cache[idx] = result
            
            return result
    
    def plot(self, sample: Dict[str, torch.Tensor], suptitle: str = None, show_axes: bool = False):
        """
        Plot a sample from the dataset - implements TerraTorch's plot functionality.
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
        
        # Get sample data
        image = sample["image"]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
            
        # Use RGB indices for visualization
        if len(image.shape) == 3:  # If channels last
            image = image[..., self.rgb_indices]
        else:  # If channels first
            image = image.take(self.rgb_indices, axis=0)
            image = np.transpose(image, (1, 2, 0))
            
        # Normalize for visualization
        image = (image - image.min(axis=(0, 1))) * (1 / (image.max(axis=(0, 1)) + 1e-8))
        image = np.clip(image, 0, 1)

        label_mask = sample["mask"]
        if isinstance(label_mask, torch.Tensor):
            label_mask = label_mask.numpy()

        showing_predictions = "prediction" in sample
        if showing_predictions:
            prediction_mask = sample["prediction"]
            if isinstance(prediction_mask, torch.Tensor):
                prediction_mask = prediction_mask.numpy()

        # Create plot
        num_images = 5 if showing_predictions else 4
        fig, ax = plt.subplots(1, num_images, figsize=(12, 10), layout="compressed")
        axes_visibility = "on" if show_axes else "off"

        # for legend
        ax[0].axis("off")

        norm = mpl.colors.Normalize(vmin=0, vmax=self.num_classes - 1)
        ax[1].axis(axes_visibility)
        ax[1].title.set_text("Image")
        ax[1].imshow(image)

        ax[2].axis(axes_visibility)
        ax[2].title.set_text("Ground Truth Mask")
        ax[2].imshow(label_mask, cmap="jet", norm=norm)

        ax[3].axis(axes_visibility)
        ax[3].title.set_text("GT Mask on Image")
        ax[3].imshow(image)
        ax[3].imshow(label_mask, cmap="jet", alpha=0.3, norm=norm)

        if showing_predictions:
            ax[4].axis(axes_visibility)
            ax[4].title.set_text("Predicted Mask")
            ax[4].imshow(prediction_mask, cmap="jet", norm=norm)

        # Add legend
        cmap = plt.get_cmap("jet")
        class_names = ["No Flood", "≥0.1m", "≥0.2m", "≥0.3m", "≥0.5m", "≥1.0m"]
        legend_data = []
        
        for i in range(self.num_classes):
            name = class_names[i] if i < len(class_names) else str(i)
            data = [i, cmap(norm(i)), name]
            legend_data.append(data)
            
        handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
        labels = [n for k, c, n in legend_data]
        ax[0].legend(handles, labels, loc="center")
        
        if suptitle is not None:
            plt.suptitle(suptitle)
            
        return fig


class FloodSegmentationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for flood segmentation compatible with TerraTorch.
    """
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 16,
        image_size: int = 512,
        num_images: int = 400,
        num_workers: int = 4,
        normalize: bool = True,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        num_classes: int = 6
    ):
        """
        Args:
            data_dir (str): Path to directory containing H5 files
            batch_size (int): Batch size
            image_size (int): Size of images
            num_images (int): Number of images per city-rainfall pair
            num_workers (int): Number of workers for data loading
            normalize (bool): Whether to normalize the data
            prefetch_factor (int): Number of batches to prefetch per worker
            persistent_workers (bool): Keep worker processes alive between dataloader calls
            num_classes (int): Number of segmentation classes
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.normalize = normalize
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.image_size = image_size
        self.num_images = num_images
        self.num_classes = num_classes
        
        # Define transforms (use Albumentations for TerraTorch compatibility)
        self.train_transform = None
        
        self.val_transform = None
    
    def setup(self, stage=None):
        """Set up the datasets for each stage."""
        if stage == 'fit' or stage is None:
            self.train_dataset = H5FloodSegmentationDataset(
                data_dir=self.data_dir,
                split='train',
                num_classes=self.num_classes,
                image_size=self.image_size,
                num_images=self.num_images,
                transform=self.train_transform,
                normalize=self.normalize,
            )
            
            self.val_dataset = H5FloodSegmentationDataset(
                data_dir=self.data_dir,
                split='val',
                num_classes=self.num_classes,
                image_size=self.image_size,
                num_images=self.num_images,
                transform=self.val_transform,
                normalize=self.normalize,
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = H5FloodSegmentationDataset(
                data_dir=self.data_dir,
                split='test',
                num_classes=self.num_classes,
                image_size=self.image_size,
                num_images=self.num_images,
                transform=self.val_transform,
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
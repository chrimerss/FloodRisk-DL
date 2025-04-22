# Import and expose classes from data.py
from .data import H5FloodSegmentationDataset, FloodSegmentationDataModule

# Make these classes available when importing the package
__all__ = ['H5FloodSegmentationDataset', 'FloodSegmentationDataModule']
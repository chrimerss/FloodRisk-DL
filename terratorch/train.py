from data import FloodSegmentationDataModule
from lightning.pytorch.loggers import WandbLogger
import os
import torch
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
import h5py
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from terratorch.tasks import SemanticSegmentationTask
from task_class import model_args_tiny, model_args_100, model_args_300, model_args_600, model_args_unet


def main():
    """Run the flood segmentation task with Prithvi EO v2 300 backbone."""
    
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Data paths and parameters
    data_dir = "/home/users/li1995/global_flood/FloodRisk-DL/src/data_preparation"  # Update this to your actual path
    output_dir = "output/all-unet-res152"
    batch_size = 8
    max_epochs = 100
    num_workers = 8
    
    # Set up data module
    data_module = FloodSegmentationDataModule(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=True,
        num_classes=6
    )
    
    # Set up model configuration with Prithvi backbone
    model_args = model_args_unet

    ## CNN backbone
    

    # scheduler_args = dict(warmup_epochs=5, max_epochs=50, eta_min=1e-8)
    scheduler_args= dict(mode='min', factor=0.1, patience=10)
    
    # Set up class information
    class_names = ["No Flood", "Nuisance Flood (0.1-0.2m]", "Minor Flood (0.2-0.4m]", "Moderate Flood (0.4-1.0m]", "Major Flood (>1.0m]"]
    class_weights = [0.5, 1.0, 1.0, 1.0, 1.5]  # Adjust if needed
    
    # Create the segmentation task
    # model = SemanticSegmentationTask(
    #     model_args=model_args,
    #     model_factory="EncoderDecoderFactory",
    #     loss="ce",  # Cross-entropy loss
    #     optimizer="AdamW",
    #     optimizer_hparams={"weight_decay": 0.05},
    #     class_names=class_names,
    #     class_weights=class_weights,
    #     scheduler='ReduceLROnPlateau',
    #     scheduler_hparams=scheduler_args,
    #     lr=1e-4,
    #     ignore_index=-1,
    #     freeze_backbone=False,
    #     plot_on_val=1
    # )

    model = SemanticSegmentationTask(
            model_args=model_args,
            model_factory="SMPModelFactory",
            loss="ce",
            lr=1e-4,
            ignore_index=-1,
            optimizer="AdamW",
            optimizer_hparams={"weight_decay": 0.05},
            freeze_backbone=False,
            class_names=class_names,
            class_weights=class_weights
        )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        monitor="val/Multiclass_Jaccard_Index",
        mode="max",
        filename="best-{epoch}",
        save_top_k=1
    )

    early_stopping = EarlyStopping(
        monitor="val/Multiclass_Jaccard_Index",
        patience=20,
        mode="max",
    )    
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    
    # Set up logger
    logger = WandbLogger(
        project='UrbanFloods2D-Segmentation',
        name='all-unet-res152',
        log_model=True
    )
    
    # Set up trainer
    trainer = pl.Trainer(
        accelerator="auto",  # Use GPU if available
        devices="auto",
        strategy='ddp_find_unused_parameters_true',
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        logger=logger,
        accumulate_grad_batches=8,
        log_every_n_steps=20,
        precision="16-mixed"  # Use mixed precision for faster training
    )
    
    # Train the model
    # model= FloodSegmentationModule(task)
    trainer.fit(model, datamodule=data_module)
    
    # Test the model using the best checkpoint
    trainer.test(datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()

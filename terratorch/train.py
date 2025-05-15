from data import FloodSegmentationDataModule, FloodCategory, FLOOD_COLORS
from lightning.pytorch.loggers import WandbLogger
import os
import torch
import numpy as np
import lightning.pytorch as pl
from pathlib import Path
import h5py
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure

import albumentations as A
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
from terratorch.tasks import SemanticSegmentationTask
from task_class import model_args_tiny, model_args_100, model_args_300, model_args_600, model_args_unet


class FloodImageLogger(Callback):
    """Callback to log flood images during validation."""
    
    def __init__(self, num_samples=4):
        super().__init__()
        self.num_samples = num_samples
        import wandb
        self.wandb= wandb
        
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Log images for a batch during validation."""
        # Only log the first few batches to avoid flooding the logs
        if batch_idx >= self.num_samples:
            return

        # Skip if outputs is None (happens during sanity check)
        if outputs is None:
            return
                        
            
        # Get images and predictions
        images = batch["image"]  # Shape: (B, C, H, W)
        masks = batch["mask"]    # Shape: (B, H, W)
        
        # print(images.shape, masks.shape, outputs)
        # Get predictions from outputs
        logits = outputs["output"]
        preds = torch.argmax(logits, dim=1)  # Shape: (B, H, W)
        
        # Only log the first image in the batch
        image = images[0].cpu()
        mask = masks[0].cpu()
        pred = preds[0].cpu()
        
        # Create a figure with a custom colormap based on the specified colors
        flood_colors = [FLOOD_COLORS[FloodCategory(i)] for i in range(len(FloodCategory))]
        cmap = plt.matplotlib.colors.ListedColormap(flood_colors)
        
        fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        
        # Plot DEM
        axes[0].set_title("DEM")
        dem_img = axes[0].imshow(image[0].numpy(), cmap='terrain')
        fig.colorbar(dem_img, ax=axes[0])
        
        # Plot Slope
        axes[1].set_title("Slope")
        slope_img = axes[1].imshow(image[1].numpy())
        fig.colorbar(slope_img, ax=axes[1])
        
        # Plot Ground Truth
        axes[2].set_title("Ground Truth")
        norm = mpl.colors.Normalize(vmin=0, vmax=len(FloodCategory)-1)
        gt_img = axes[2].imshow(mask.numpy(), cmap=cmap, norm=norm)
        gt_cbar = fig.colorbar(gt_img, ax=axes[2], ticks=range(len(FloodCategory)))
        gt_cbar.ax.set_yticklabels([cat.name for cat in FloodCategory])
        
        # Plot Prediction
        axes[3].set_title("Prediction")
        pred_img = axes[3].imshow(pred.numpy(), cmap=cmap, norm=norm)
        pred_cbar = fig.colorbar(pred_img, ax=axes[3], ticks=range(len(FloodCategory)))
        pred_cbar.ax.set_yticklabels([cat.name for cat in FloodCategory])
        
        # Log to logger
        trainer.logger.experiment.log({"validation_samples": [self.wandb.Image(fig, caption=f"Batch {batch_idx}")]})
        
        # Close the figure to avoid memory leaks
        plt.close(fig)


def main():
    """Run the flood segmentation task with Prithvi EO v2 300 backbone."""
    
    # Set random seed for reproducibility
    pl.seed_everything(42)
    
    # Data paths and parameters
    data_dir = "/home/users/li1995/global_flood/FloodRisk-DL/src/data_preparation"  # Update this to your actual path
    output_dir = "output/all-600-dice"
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
    model_args = model_args_600

    ## CNN backbone
    

    # scheduler_args = dict(warmup_epochs=5, max_epochs=50, eta_min=1e-8)
    scheduler_args= dict(mode='min', factor=0.1, patience=10)
    
    # Set up class information
    class_names = ["No Flood", "Nuisance Flood (0.1-0.2m]", "Minor Flood (0.2-0.4m]", "Moderate Flood (0.4-1.0m]", "Major Flood (>1.0m]"]
    class_weights = [1, 2, 4, 6, 10]  # Adjust if needed
    
    # Create the segmentation task
    model = SemanticSegmentationTask(
        model_args=model_args,
        model_factory="EncoderDecoderFactory",
        loss="ce",  # Cross-entropy loss
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": 0.05},
        class_names=class_names,
        class_weights=class_weights,
        scheduler='ReduceLROnPlateau',
        scheduler_hparams=scheduler_args,
        lr=1e-4,
        ignore_index=-1,
        freeze_backbone=False,
        plot_on_val=1)

    # model = SemanticSegmentationTask(
    #         model_args=model_args,
    #         model_factory="SMPModelFactory",
    #         loss="ce",
    #         lr=1e-4,
    #         ignore_index=-1,
    #         optimizer="AdamW",
    #         optimizer_hparams={"weight_decay": 0.05},
    #         freeze_backbone=False,
    #         class_names=class_names,
    #         class_weights=class_weights
    #     )
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{output_dir}/checkpoints",
        # monitor="val/Multiclass_Jaccard_Index",
        monitor="val/multiclassaccuracy_Major Flood (>1.0m]",
        mode="max",
        filename="best-{epoch}",
        save_top_k=1
    )

    early_stopping = EarlyStopping(
        # monitor="val/Multiclass_Jaccard_Index",
        monitor="val/multiclassaccuracy_Major Flood (>1.0m]",
        patience=20,
        mode="max",
    )    
    
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    

    
    # Set up logger
    logger = WandbLogger(
        project='UrbanFloods2D-Segmentation',
        name='all-600-dice',
        log_model=True
    )
    
    # Add image logging callback
    image_logger = FloodImageLogger(num_samples=4)
    
    
    # Set up trainer
    trainer = pl.Trainer(
        accelerator="auto",  # Use GPU if available
        devices="auto",
        strategy='ddp_find_unused_parameters_true',
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping, image_logger],
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

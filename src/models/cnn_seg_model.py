import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, DeepLabV3_ResNet101_Weights, DeepLabV3_ResNet50_Weights
from .lr_scheduler import LinearWarmupCosineAnnealingLR
from enum import Enum

class FloodCategory(Enum):
    """Enumeration of flood categories based on max flood depth."""
    NO_FLOOD = 0      # max_flood_depth < 0.1
    NUISANCE = 1      # 0.1 <= max_flood_depth < 0.2
    MINOR = 2         # 0.2 <= max_flood_depth < 0.3
    MEDIUM = 3        # 0.3 <= max_flood_depth < 0.5
    MAJOR = 4         # 0.5 <= max_flood_depth < 1.0
    EXTREME = 5       # max_flood_depth >= 1.0

FLOOD_COLORS = {
    FloodCategory.NO_FLOOD: '#FFFFFF',   # White
    FloodCategory.NUISANCE: '#C6DBEF',   # Light blue
    FloodCategory.MINOR: '#9ECAE1',      # Medium light blue
    FloodCategory.MEDIUM: '#6BAED6',     # Medium blue
    FloodCategory.MAJOR: '#3182BD',      # Dark blue
    FloodCategory.EXTREME: '#08519C'     # Very dark blue
}

class FloodSegmentationModule(nn.Module):
    """
    Flood segmentation model using pretrained DeepLabV3 from torchvision.
    Uses transfer learning and adapts the model for flood depth prediction.
    """
    def __init__(self, backbone="resnet50", in_channels=2, num_classes=6, pretrained=True):
        """
        Initialize the segmentation model.
        
        Args:
            backbone (str): Backbone to use - "resnet50" or "resnet101"
            in_channels (int): Number of input channels (DEM and rainfall)
            num_classes (int): Number of output classes (flood depth categories)
            pretrained (bool): Whether to use pretrained weights
        """
        super().__init__()
        
        # Select backbone and load pretrained weights if requested
        if backbone == "resnet50":
            weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            self.model = deeplabv3_resnet50(weights=weights)
        else:
            weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
            self.model = deeplabv3_resnet101(weights=weights)
        
        # Adapt the first layer for 2-channel input (DEM and rainfall)
        # Preserve pretrained weights for RGB channels and initialize the new channel
        if in_channels != 3:
            original_conv = self.model.backbone.conv1
            new_conv = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size, 
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias
            )
            
            # Initialize weights using pretrained weights
            if pretrained:
                # For 2 channels, reuse the first two channels from the pretrained model
                new_conv.weight.data[:, :2, :, :] = original_conv.weight.data[:, :2, :, :]
                
                # For additional channels, initialize with random values
                if in_channels > 3:
                    new_conv.weight.data[:, 3:, :, :] = torch.randn(
                        original_conv.weight.data[:, 0, :, :].shape[0], 
                        in_channels - 3, 
                        original_conv.weight.data.shape[2], 
                        original_conv.weight.data.shape[3]
                    ) * 0.01
            
            # Replace the model's first convolution layer
            self.model.backbone.conv1 = new_conv
        
        # Replace the classifier to output the desired number of classes
        self.model.classifier = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Output segmentation [B, num_classes, H, W]
        """
        # The DeepLabV3 model returns a dict with 'out' key for the output
        result = self.model(x)
        return result['out']


class FloodSegmentationModel(pl.LightningModule):
    """PyTorch Lightning module for training and evaluating the flood segmentation model."""
    def __init__(self, config):
        """
        Initialize the Lightning module.
        
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Model parameters
        backbone = config.model.get('backbone', "resnet50")
        in_channels = config.model.get('in_channels', 2)  # Default: DEM and rainfall channels
        num_classes = 6  # Number of flood categories (NO_FLOOD, NUISANCE, MINOR, MEDIUM, MAJOR, EXTREME)
        pretrained = config.model.get('pretrained', True)
        
        # Initialize model
        self.model = FloodSegmentationModule(
            backbone=backbone,
            in_channels=in_channels,
            num_classes=num_classes,
            pretrained=pretrained
        )
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()

        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        
        
        # Calculate classification loss
        class_loss = self.ce_loss(logits, targets)
        
        # Convert predicted categories back to depth for regression metrics
        _, predicted_categories = torch.max(logits, dim=1)
        
        
        # Log metrics
        self.log('train_loss', class_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return class_loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        
        # Calculate classification loss
        class_loss = self.ce_loss(logits, targets)
        
        # Convert predicted categories back to depth for regression metrics
        _, predicted_categories = torch.max(logits, dim=1)

        self.log('val_loss', class_loss, on_step=True, on_epoch=True, prog_bar=True)
        # Visualize predictions for first batch
        if batch_idx == 0:
            self._log_predictions(inputs, targets, predicted_categories)
            
        return class_loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        logits = self(inputs)
        
        # Calculate classification loss
        class_loss = self.ce_loss(logits, targets)
        
        # Convert predicted categories back to depth for regression metrics
        _, predicted_categories = torch.max(logits, dim=1)

        # Log metrics
        self.log('test_loss', class_loss, on_step=False, on_epoch=True)
        
        return class_loss
    
    def _log_predictions(self, inputs, targets, predicted_categories, num_samples=4):
        """Log visualizations of model predictions to wandb."""
        # Only log a limited number of samples
        batch_size = min(num_samples, inputs.size(0))
        
        
        
        # Create color maps for visualization
        flood_colors = [FLOOD_COLORS[FloodCategory(i)] for i in range(6)]
        cmap = plt.matplotlib.colors.ListedColormap(flood_colors)
        
        for i in range(batch_size):
            # Get the data for this sample
            dem = inputs[i, 0].detach().cpu().numpy()  # DEM
            rainfall = inputs[i, 1].detach().cpu().numpy()  # Rainfall
            rainfall_value = np.mean(rainfall)  # Estimate rainfall value
            
            target_cat = targets[i, 0].detach().cpu().numpy()

            pred_cat = predicted_categories[i].detach().cpu().numpy()

            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(12, 12))
            
            # Plot DEM
            im0 = axs[0, 0].imshow(dem, cmap='terrain')
            axs[0, 0].set_title(f'DEM (Elevation)')
            fig.colorbar(im0, ax=axs[0, 0])
            
            # Plot rainfall
            axs[0, 1].text(0.5, 0.5, f'Rainfall: {rainfall_value:.2f}', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axs[0, 1].transAxes, fontsize=20)
            axs[0, 1].axis('off')
            
            # Plot target depth as continuous value
            vmax = max(np.max(target_depth), np.max(pred_depth)) 
            diff = pred_cat - target_cat
            im2 = axs[0, 2].imshow(diff, cmap='coolwarm', vmin=-5, vmax=5)
            axs[0, 2].set_title('Category Error (Predicted - Target)')
            fig.colorbar(im2, ax=axs[0, 2])
            
            # Plot target depth as categories
            im3 = axs[1, 0].imshow(target_cat, cmap=cmap, vmin=0, vmax=5)
            axs[1, 0].set_title('Target Categories')
            cbar = fig.colorbar(im3, ax=axs[1, 0], ticks=range(6))
            cbar.ax.set_yticklabels([cat.name for cat in FloodCategory])
            
            # Plot predicted depth as categories
            im4 = axs[1, 1].imshow(pred_cat, cmap=cmap, vmin=0, vmax=5)
            axs[1, 1].set_title('Predicted Categories')
            cbar = fig.colorbar(im4, ax=axs[1, 1], ticks=range(6))
            cbar.ax.set_yticklabels([cat.name for cat in FloodCategory])
            
            
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({f"segmentation_sample_{i}": wandb.Image(fig)})
            plt.close(fig)
            

    
    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Create learning rate scheduler
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.config.training.warmup_epochs,
            max_epochs=self.config.training.get('maxepochs', 50),
            warmup_start_lr=self.config.training.get('warmuplr', 1e-6),
            eta_min=self.config.training.get('etamin', 1e-6)
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "epoch",
                "name": "learning_rate"
            }
        }
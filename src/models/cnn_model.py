import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
import matplotlib.pyplot as plt
import numpy as np
from .lr_scheduler import LinearWarmupCosineAnnealingLR

class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and ReLU activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout_rate=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class DownsampleBlock(nn.Module):
    """Downsample block with max pooling followed by convolutional blocks."""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels, dropout_rate=dropout_rate)
        self.conv2 = ConvBlock(out_channels, out_channels, dropout_rate=dropout_rate)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)

        return x

class UpsampleBlock(nn.Module):
    """Upsample block with transposed convolution and skip connections."""
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(in_channels, out_channels, dropout_rate=dropout_rate)
        self.conv2 = ConvBlock(out_channels, out_channels, dropout_rate=dropout_rate)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, skip):
        x = self.upconv(x)
        # Handle cases where dimensions don't match due to padding
        if x.shape != skip.shape:
            # Center crop skip to match x if needed
            diff_y = skip.size()[2] - x.size()[2]
            diff_x = skip.size()[3] - x.size()[3]
            skip = skip[:, :, 
                         diff_y // 2:(diff_y // 2 + x.size()[2]), 
                         diff_x // 2:(diff_x // 2 + x.size()[3])]
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        
        return x

class FloodCNN(nn.Module):
    """CNN model with U-Net like architecture for flood prediction."""
    def __init__(self, in_channels=2, out_channels=1, filters=[64, 128, 256, 512], dropout_rate=0.2):
        super().__init__()
        self.filters = filters
        
        # Initial convolution
        self.init_conv1 = ConvBlock(in_channels, filters[0], dropout_rate=0)
        self.init_conv2 = ConvBlock(filters[0], filters[0], dropout_rate=0)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList([
            DownsampleBlock(filters[i], filters[i+1], dropout_rate=dropout_rate) 
            for i in range(len(filters)-1)
        ])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBlock(filters[-1], filters[-1]*2, dropout_rate=dropout_rate),
            ConvBlock(filters[-1]*2, filters[-1]*2, dropout_rate=dropout_rate),
            nn.ConvTranspose2d(filters[-1]*2, filters[-1], kernel_size=2, stride=2)
        )
        
        # Upsampling path
        self.up_blocks = nn.ModuleList([
            UpsampleBlock(filters[i+1], filters[i], dropout_rate=dropout_rate)
            for i in range(len(filters)-2, -1, -1)
        ])
        
        # Final convolution
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Initial convolution
        x = self.init_conv1(x)
        x = self.init_conv2(x)
        
        # Save skip connections
        skips = [x]
        
        # Downsampling path
        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Upsampling path with skip connections
        skips = skips[::-1]  # Reverse to use in upsampling
        for i, up_block in enumerate(self.up_blocks):
            skip = skips[i+1]
            x = up_block(x, skip)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x

class FloodCNNModel(pl.LightningModule):
    """PyTorch Lightning module for training and evaluating the FloodCNN model."""
    def __init__(self, config):
        super().__init__()
        
        # Save hyperparameters to be logged by MLflow
        self.save_hyperparameters(config)
        self.config = config
        
        # Model parameters
        in_channels = config.model.get('in_chans', 2)  # Default: DEM and rainfall channels
        out_channels = 1  # Flood depth prediction
        filters = config.model.get('filters', [64, 128, 256, 512])
        dropout_rate = config.model.get('dropout_rate', 0.2)
        
        # Initialize model
        self.model = FloodCNN(
            in_channels=in_channels,
            out_channels=out_channels,
            filters=filters,
            dropout_rate=dropout_rate
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        
        # Metrics
        self.train_mse = MeanSquaredError()
        self.train_mae = MeanAbsoluteError()
        self.train_r2 = R2Score()
        
        self.val_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        
        self.test_mse = MeanSquaredError()
        self.test_mae = MeanAbsoluteError()
        self.test_r2 = R2Score()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.loss_fn(predictions, targets)
        
        # Update metrics
        self.train_mse(predictions, targets)
        self.train_mae(predictions, targets)
        self.train_r2(predictions.view(-1), targets.view(-1))
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', self.train_mse, on_step=False, on_epoch=True)
        self.log('train_mae', self.train_mae, on_step=False, on_epoch=True)
        self.log('train_r2', self.train_r2, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.loss_fn(predictions, targets)
        
        # Update metrics
        self.val_mse(predictions, targets)
        self.val_mae(predictions, targets)
        self.val_r2(predictions.view(-1), targets.view(-1))
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_mse', self.val_mse, on_step=False, on_epoch=True)
        self.log('val_mae', self.val_mae, on_step=False, on_epoch=True)
        self.log('val_r2', self.val_r2, on_step=False, on_epoch=True)
        
        # Visualize predictions for first batch
        if batch_idx == 0:
            self._log_predictions(inputs, targets, predictions)
            
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs)
        loss = self.loss_fn(predictions, targets)
        
        # Update metrics
        self.test_mse(predictions, targets)
        self.test_mae(predictions, targets)
        self.test_r2(predictions.view(-1), targets.view(-1))
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_mse', self.test_mse, on_step=False, on_epoch=True)
        self.log('test_mae', self.test_mae, on_step=False, on_epoch=True)
        self.log('test_r2', self.test_r2, on_step=False, on_epoch=True)
        
        return loss
    
    def _log_predictions(self, inputs, targets, predictions, num_samples=4):
        """Log visualizations of model predictions to wandb."""
        # Only log a limited number of samples
        n = min(num_samples, inputs.shape[0])
        
        for i in range(n):
            dem = inputs[i, 0].detach().cpu().numpy()
            rainfall = inputs[i, 1].detach().mean().cpu().numpy()
            target = targets[i, 0].detach().cpu().numpy()
            # denormalize target
            target= np.exp((target+1e-4) *2)
            pred = predictions[i, 0].detach().cpu().numpy()
            pred= np.exp((pred+1e-4) *2)
            
            # Create figure with subplots
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot DEM
            im0 = axs[0,0].imshow(dem, cmap='terrain')
            axs[0,0].set_title(f'DEM (Rainfall {rainfall}mm)')
            fig.colorbar(im0, ax=axs[0,0])

            # Plot histogram of predicted flood depths
            axs[0, 1].hist(pred.flatten(), bins=50, color='steelblue', edgecolor='black')
            axs[0, 1].set_title('Histogram of Predicted Depths')
            axs[0, 1].set_xlabel('Depth (m)')
            axs[0, 1].set_ylabel('Frequency')
            
            vmax= target.max()

            # Plot target depth
            im1 = axs[1, 0].imshow(target, cmap='Blues', vmin=0, vmax=vmax)
            axs[1, 0].set_title('Target Max Flood Depth')
            fig.colorbar(im1, ax=axs[1,0])
            
            # Plot predicted depth
            im2 = axs[1, 1].imshow(pred, cmap='Blues', vmin=0, vmax=vmax)
            axs[1, 1].set_title('Predicted Max Flood Depth')
            fig.colorbar(im2, ax=axs[1, 1])
            
            plt.tight_layout()
            
            # Log to wandb
            wandb.log({f"sample_{i}": wandb.Image(fig)})
            plt.close(fig)
    
    def configure_optimizers(self):
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Create learning rate scheduler
        if self.config.training.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=self.config.training.max_epochs,
                    eta_min=1e-6
                )
        elif self.config.training.lr_scheduler == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=True
                )
        elif self.config.training.lr_scheduler == "LinearWarmupCosineAnnealingLR":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=self.config.training.warmup_epochs,
                max_epochs=self.config.training.maxepochs,
                warmup_start_lr=self.config.training.warmuplr,
                eta_min=self.config.training.etamin,
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss",
            },
        }
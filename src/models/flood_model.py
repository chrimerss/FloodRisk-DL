import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import wandb
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
import matplotlib.pyplot as plt
import numpy as np

class FloodPredictionModel(pl.LightningModule):
    def __init__(self, config):
        """
        Flood prediction model using Swin Transformer.
        
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Use timm's Swin Transformer implementation
        # Note: We're using a regression task (predicting flood depth) instead of classification
        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224', 
            pretrained=True,
            in_chans=config.model.in_channels,  # DEM and rainfall input
            num_classes=0  # Remove classification head
        )
        
        # Add custom regression head
        feature_dim = self.model.num_features
        self.regression_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1024),  # Output size for a 32x32 prediction
        )
        
        # Upsampling layers to get back to the original resolution (1024x1024)
        self.upsampling = nn.Sequential(
            nn.Unflatten(1, (1, 32, 32)),  # Reshape to [B, 1, 32, 32]
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)  # Upsample to 1024x1024
        )
        
        # Metrics for evaluation
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        
    def forward(self, x):
        # Extract features from Swin Transformer
        features = self.model(x)
        
        # Apply regression head
        output = self.regression_head(features)
        
        # Upsample to original resolution
        output = self.upsampling(output)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Calculate MSE loss
        loss = F.mse_loss(y_hat, y)
        
        # Log metrics
        self.train_mse(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_mse', self.train_mse, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Calculate loss
        loss = F.mse_loss(y_hat, y)
        
        # Log metrics
        self.val_mse(y_hat, y)
        self.val_mae(y_hat, y)
        self.val_r2(y_hat, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mse', self.val_mse, prog_bar=True)
        self.log('val_mae', self.val_mae, prog_bar=True)
        self.log('val_r2', self.val_r2, prog_bar=True)
        
        # Log sample predictions to WandB
        if batch_idx == 0:
            self._log_predictions(x, y, y_hat)
            
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Calculate loss
        loss = F.mse_loss(y_hat, y)
        
        # Log metrics
        self.test_mse(y_hat, y)
        
        self.log('test_loss', loss)
        self.log('test_mse', self.test_mse)
        
        return loss
    
    def _log_predictions(self, inputs, targets, predictions, num_samples=4):
        """Log sample predictions to WandB."""
        # Only use a subset of the batch
        batch_size = min(inputs.size(0), num_samples)
        
        for i in range(batch_size):
            # Get the input, target, and prediction for this sample
            dem = inputs[i, 0].detach().cpu().numpy()  # DEM
            rainfall_channel = inputs[i, 1].detach().cpu().numpy()  # Rainfall (uniform value)
            target = targets[i, 0].detach().cpu().numpy()  # Target flood depth
            pred = predictions[i, 0].detach().cpu().numpy()  # Predicted flood depth
            
            # Estimate the rainfall value (it should be uniform across the channel)
            rainfall_value = np.mean(rainfall_channel)
            
            # Create a figure with 4 subplots
            fig, axs = plt.subplots(2, 2, figsize=(16, 12))
            
            # Plot DEM
            im0 = axs[0, 0].imshow(dem, cmap='terrain')
            axs[0, 0].set_title(f'DEM (Elevation)')
            plt.colorbar(im0, ax=axs[0, 0])
            
            # Plot rainfall (just indicate the value)
            axs[0, 1].text(0.5, 0.5, f'Rainfall: {rainfall_value:.2f}', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=axs[0, 1].transAxes, fontsize=20)
            axs[0, 1].axis('off')
            
            # Plot target flood depth
            im2 = axs[1, 0].imshow(target, cmap='Blues')
            axs[1, 0].set_title('Target Max Flood Depth')
            plt.colorbar(im2, ax=axs[1, 0])
            
            # Plot predicted flood depth
            im3 = axs[1, 1].imshow(pred, cmap='Blues')
            axs[1, 1].set_title('Predicted Max Flood Depth')
            plt.colorbar(im3, ax=axs[1, 1])
            
            plt.tight_layout()
            
            # Log to WandB
            wandb.log({f"prediction_sample_{i}": wandb.Image(fig)})
            plt.close(fig)
            
            # Plot the difference as well
            diff = pred - target
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(diff, cmap='RdBu', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
            ax.set_title('Prediction Error (Predicted - Target)')
            plt.colorbar(im, ax=ax)
            
            wandb.log({f"prediction_error_{i}": wandb.Image(fig)})
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
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=self.config.training.max_epochs,
                    eta_min=1e-6
                ),
                'interval': 'epoch',
                'name': 'lr'
            }
        else:
            scheduler = {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                    verbose=True
                ),
                'interval': 'epoch',
                'monitor': 'val_loss',
                'name': 'lr'
            }
        
        return [optimizer], [scheduler] 
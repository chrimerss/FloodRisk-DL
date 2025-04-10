from diffusers import UNet2DModel, DDPMScheduler
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
import matplotlib.pyplot as plt
import numpy as np
from .lr_scheduler import LinearWarmupCosineAnnealingLR



class FloodDiffusionModel(pl.LightningModule):
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
        self.model = model = UNet2DModel(
                                        sample_size=config.data.image_size,  # the target image resolution
                                        in_channels=self.config.model.in_channels,  # the number of input channels, 3 for RGB images
                                        out_channels=1,  # the number of output channels
                                        layers_per_block=config.model.layers_per_block,  # how many ResNet layers to use per UNet block
                                        block_out_channels=config.model.block_out_channels,  # the number of output channels for each UNet block
                                        down_block_types=(
                                            "DownBlock2D",  # a regular ResNet downsampling block
                                            "DownBlock2D",
                                            "DownBlock2D",
                                            "DownBlock2D",
                                            "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                                            "DownBlock2D",
                                        ),
                                        up_block_types=(
                                            "UpBlock2D",  # a regular ResNet upsampling block
                                            "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                                            "UpBlock2D",
                                            "UpBlock2D",
                                            "UpBlock2D",
                                            "UpBlock2D",
                                        ),
                                    )
        self.noise_scheduler= DDPMScheduler(num_train_timesteps=self.config.model.timestep)
        
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
        
    def forward(self, x_t, t):

        noise_pred = self.model(x_t, t, return_dict=False)[0]

        return noise_pred
        
    def _predict_clean(self, x_t, noise_pred, t):
        """
        Reconstruct the clean image x_0 from x_t and noise_pred at time t.
        Using the standard DDPM formula:
            x_0_hat = (x_t - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
        """
        # alphas_cumprod[t] is alpha_t
        alpha_t = self.noise_scheduler.alphas_cumprod[t].reshape(-1, 1, 1, 1).to(x_t.device)
        # sqrt_alpha_t: shape (B, 1, 1, 1)
        sqrt_alpha_t = alpha_t.sqrt()
        sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt()

        x_0_hat = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        return x_0_hat        
    
    def training_step(self, batch, batch_idx):
        inputs, targets = batch

        condition= inputs[:,0:1,:,:]

        # sum over channel dimension
        bs= inputs.shape[0]
        # 2) Random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bs,), device=inputs.device, dtype=torch.long
        )

        # 3) Sample random noise and get x_t = alpha_t^0.5 * x_0 + ...
        noise = torch.randn_like(targets)
        noisy_images = self.noise_scheduler.add_noise(targets, noise, timesteps)

        x_2chan = torch.cat([noisy_images, condition], dim=1)

        # 4) Model forward => predict noise
        noise_pred = self(x_2chan, timesteps)
        
        # 5) Compute loss
        loss = self.loss_fn(noise_pred, noise)
        # with torch.no_grad():
#    ?         predictions = self._predict_clean(noisy_images, noise_pred, timesteps)        
        # Update metrics
        # self.train_mse(predictions, targets)
        # self.train_mae(predictions, targets)
        # self.train_r2(predictions.view(-1), targets.view(-1))
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, targets = batch

        condition= inputs[:,0:1,:,:]

        # sum over channel dimension
        bs= inputs.shape[0]
        # 2) Random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bs,), device=inputs.device, dtype=torch.long
        )

        # 3) Sample random noise and get x_t = alpha_t^0.5 * x_0 + ...
        noise = torch.randn_like(targets)
        noisy_images = self.noise_scheduler.add_noise(targets, noise, timesteps)

        x_2chan = torch.cat([noisy_images, condition], dim=1)

        # 4) Model forward => predict noise
        noise_pred = self(x_2chan, timesteps)
        
        # 5) Compute loss
        loss = self.loss_fn(noise_pred, noise)

        # Approx denoised image
        with torch.no_grad():
            predictions = self._predict_clean(noisy_images, noise_pred, timesteps)

        # self.val_mse(predictions, targets)
        # self.val_mae(predictions, targets)
        # self.val_r2(predictions.view(-1), targets.view(-1))

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log sample predictions
        if batch_idx == 0:
            self._log_predictions(inputs, targets, predictions)
            
        return loss
    
    def test_step(self, batch, batch_idx):
        inputs, targets = batch

        condition= inputs[:,0:1,:,:]

        # sum over channel dimension
        bs= inputs.shape[0]
        # 2) Random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bs,), device=inputs.device, dtype=torch.long
        )

        # 3) Sample random noise and get x_t = alpha_t^0.5 * x_0 + ...
        noise = torch.randn_like(targets)
        noisy_images = self.noise_scheduler.add_noise(targets, noise, timesteps)

        x_2chan = torch.cat([noisy_images, condition], dim=1)

        # 4) Model forward => predict noise
        noise_pred = self(x_2chan, timesteps)
        
        # 5) Compute loss
        loss = self.loss_fn(noise_pred, noise)
        
        # Update metrics
        # self.test_mse(predictions, targets)
        # self.test_mae(predictions, targets)
        # self.test_r2(predictions.view(-1), targets.view(-1))
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)

        
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
            pred_1d= pred.flatten()
            data_finite = pred_1d[np.isfinite(pred_1d)]
            axs[0, 1].hist(data_finite, bins=50, color='steelblue', edgecolor='black')
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
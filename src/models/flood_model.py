import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
import matplotlib.pyplot as plt
import numpy as np
from .utils import Mlp, DropPath
from .lr_scheduler import LinearWarmupCosineAnnealingLR

# Swin Transformer Block
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True, 
                 drop_path=0.0, attn_drop=0.0, proj_drop=0.0, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # Create LayerNorm with the correct dimension
        # For a tensor of shape [B, H, W, C], we need to normalize over the last dimension C
        self.norm1 = nn.LayerNorm(dim)  # Normalize over the channel dimension
        self.norm2 = nn.LayerNorm(dim)  # Normalize over the channel dimension
        
        # Multi-head self attention
        self.attn = WindowAttention(
            dim, 
            window_size=(window_size, window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop
        )
        
        # MLP block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=nn.GELU, 
            drop=proj_drop
        )
        
        # Drop path (stochastic depth)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        
        # Reshape for layer norm - LayerNorm expects [B, H, W, C] format
        x = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        
        # Apply layer norm along the channel dimension only
        x_norm = self.norm1(x)  # B, H, W, C
        
        # Cyclic shift if needed (for shifted window attention)
        if self.shift_size > 0:
            shifted_x = torch.roll(x_norm, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x_norm
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # (B*num_windows, window_size, window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows)  # (B*num_windows, window_size, window_size, C)
        
        # Merge windows
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # Convert back to channel-first format for residual connection
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        
        # Apply residual connection
        x = shortcut + self.drop_path(x)
        
        # MLP with skip connection
        # First convert to channel-last for layer norm
        x_channel_last = x.permute(0, 2, 3, 1).contiguous()  # B, H, W, C
        x_norm = self.norm2(x_channel_last)
        
        # Convert back to channel-first for MLP
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        
        # Apply MLP and residual connection
        x = x + self.drop_path(self.mlp(x_norm))
        
        return x

def window_partition(x, window_size):
    """
    Partition into non-overlapping windows.
    Args:
        x: (B, H, W, C)
        window_size: window size
    Returns:
        windows: (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition.
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    """Window based multi-head self attention module."""
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Define a parameter for relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        
        # Get pair-wise relative position index for each token in the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B*num_windows, window_size, window_size, C)
        """
        # Handle different input shapes
        if len(x.shape) == 4:
            B_, H, W, C = x.shape
            N = H * W
            x = x.reshape(B_, N, C)
        else:
            B_, N, C = x.shape
            H = W = int(N ** 0.5)
        
        # Generate qkv 
        qkv = self.qkv(x)
        
        # Reshape qkv for multi-head attention
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B_, num_heads, N, C//num_heads
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # Apply scale factor to query
        q = q * self.scale
        
        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1))  # B_, num_heads, N, N

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        
        # Add position bias to attention scores
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply softmax and dropout
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Calculate weighted sum
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        
        # Apply projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        # Reshape back to window format if input was 4D
        if len(x.shape) == 3 and H == W:
            x = x.reshape(B_, H, W, C)
            
        return x

class SwinTransformer(nn.Module):
    def __init__(self, img_size=1024,
                        patch_size=4,
                        in_chans=2,  # DEM and rainfall channels
                        num_classes=0,  # No classification head
                        embed_dim=96,
                        num_layers=4,
                        num_heads=8,
                        window_size=8,
                        mlp_ratio=4.,
                        qkv_bias=True,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        drop_path_rate=0.1):
        super().__init__()
        
        # Initial embedding dimension
        self.embed_dim = embed_dim
        self.in_channels = in_chans
        self.num_layers = num_layers
        self.window_size = window_size
        self.num_heads = num_heads
        
        # Patch embedding with normalized conv - preserve more spatial information
        self.patch_embed = nn.Sequential(
            nn.Conv2d(self.in_channels, self.embed_dim, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(self.embed_dim),
            nn.GELU()
        )

        # Create stages with progressively increasing dimensions
        self.dims = [self.embed_dim * (2**i) for i in range(self.num_layers)]
        
        # Transformer blocks for encoding
        self.encoder_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=self.dims[l],
                num_heads=self.num_heads,
                window_size=self.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=drop_path_rate * (l / (self.num_layers - 1)),
                attn_drop=0,
                proj_drop=0
            )
            for l in range(self.num_layers)
        ])

        # Downsampling layers
        self.downsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.dims[l], self.dims[l+1], kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(self.dims[l+1]),
                nn.GELU()
            )
            for l in range(self.num_layers - 1)
        ])

        # Transformer blocks for decoding with skip connections
        # Use the same dimension for each decoder level as the corresponding encoder level
        self.decoder_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=self.dims[l-1],  # Use the dimension of the target level
                num_heads=self.num_heads,
                window_size=self.window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=drop_path_rate * (l / (self.num_layers - 1)),
                attn_drop=0,
                proj_drop=0
            )
            for l in range(self.num_layers - 1, 0, -1)
        ])

        # Upsampling layers
        self.upsample = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(self.dims[l], self.dims[l-1], kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(self.dims[l-1]),
                nn.GELU()
            )
            for l in range(self.num_layers - 1, 0, -1)
        ])
        
        # The fusion layers will be created dynamically during forward pass
        # to account for potential dimension changes
        
        # Final output layer - gradual upsampling to preserve detail
        self.output_layer = nn.Sequential(
            nn.Conv2d(self.dims[0], self.dims[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dims[0] // 2),
            nn.GELU(),
            nn.Conv2d(self.dims[0] // 2, self.dims[0] // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.dims[0] // 4),
            nn.GELU(),
            nn.ConvTranspose2d(self.dims[0] // 4, 1, kernel_size=4, stride=4),
        )
        
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Encoder path with skip connections
        features = [x]  # Store initial features
        feature_dims = [x.shape[1]]  # Store feature dimensions
        
        # Process through encoder blocks and downsample
        for i, (encoder_block, down) in enumerate(zip(self.encoder_blocks[:-1], self.downsample)):
            x = encoder_block(x)
            features.append(x)  # Store features before downsampling
            feature_dims.append(x.shape[1])
            x = down(x)
        
        # Process at bottleneck
        x = self.encoder_blocks[-1](x)
        
        # Decoder path with skip connections
        for i, (decoder_block, up) in enumerate(zip(self.decoder_blocks, self.upsample)):
            x = up(x)  # Upsample
            skip_features = features[-(i+2)]  # Get corresponding skip connection
            
            # Ensure spatial dimensions match before concatenating
            if x.size(2) != skip_features.size(2) or x.size(3) != skip_features.size(3):
                # Resize skip features to match upsampled features if needed
                skip_features = F.interpolate(skip_features, size=(x.size(2), x.size(3)), 
                                           mode='bilinear', align_corners=False)
            
            # Use a fixed convolution to reduce skip connection channels if needed
            if skip_features.size(1) != x.size(1):
                # Create a 1x1 conv to adjust channels and add it to the module
                skip_conv = nn.Conv2d(skip_features.size(1), x.size(1), kernel_size=1).to(x.device)
                skip_features = skip_conv(skip_features)
                
            # Skip connection with residual addition instead of concatenation
            # This avoids dimension mismatch issues completely
            x = x + skip_features
            
            # Apply transformer block
            x = decoder_block(x)
        
        # Final upsampling and output projection
        output = self.output_layer(x)
        
        return output


class FloodPredictionModel(pl.LightningModule):
    def __init__(self, config):
        """
        Flood prediction model using custom Swin Transformer.
        
        Args:
            config: Configuration object with model parameters
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Custom Swin Transformer with proper depth calculation
        self.model = SwinTransformer(
            img_size=self.config.model.img_size,
            patch_size=self.config.model.patch_size,
            in_chans=self.config.model.in_channels,  # DEM and rainfall channels
            num_classes=0,  # No classification head
            embed_dim=self.config.model.embed_dim,
            num_layers=self.config.model.num_layers,
            num_heads=self.config.model.num_heads,
            window_size=self.config.model.window_size,
            mlp_ratio=self.config.model.mlp_ratio,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1
        )
    
        
        # Metrics for evaluation remain the same
        self.train_mse = MeanSquaredError()
        self.val_mse = MeanSquaredError()
        self.test_mse = MeanSquaredError()
        self.val_mae = MeanAbsoluteError()
        self.val_r2 = R2Score()
        self.criterion= nn.MSELoss()
        
    def forward(self, x):
        # Extract features from Swin Transformer
        output = self.model(x)
        
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Calculate MSE loss
        loss = self.criterion(y_hat, y)
        
        # Log metrics
        self.train_mse(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_mse', self.train_mse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # Calculate loss
        loss = F.mse_loss(y_hat, y)
        
        # Log metrics
        self.val_mse(y_hat, y)
        self.val_mae(y_hat, y)
        # self.val_r2(y_hat, y)
        
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_mse', self.val_mse, prog_bar=True, sync_dist=True)
        self.log('val_mae', self.val_mae, prog_bar=True, sync_dist=True)
        # self.log('val_r2', self.val_r2, prog_bar=True)
        
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
        
        self.log('test_loss', loss, sync_dist=True)
        self.log('test_mse', self.test_mse, sync_dist=True)
        
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

            vmax= target.max()
            
            # Plot target flood depth
            im2 = axs[1, 0].imshow(target, cmap='Blues', vmin=0, vmax=vmax)
            axs[1, 0].set_title('Target Max Flood Depth')
            plt.colorbar(im2, ax=axs[1, 0])
            
            # Plot predicted flood depth
            im3 = axs[1, 1].imshow(pred, cmap='Blues', vmin=0, vmax=vmax)
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
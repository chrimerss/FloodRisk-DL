import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torchmetrics import MeanSquaredError, MeanAbsoluteError, R2Score
import matplotlib.pyplot as plt
import numpy as np

class PatchEmbed(nn.Module):
    """Split image into patches and embed them."""
    def __init__(self, img_size=1024, patch_size=4, in_chans=2, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (B, n_patches, embed_dim)
        return x

class PatchMerging(nn.Module):
    """Merge patches to reduce resolution."""
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        
    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class SwinTransformerBlock(nn.Module):
    """Swin Transformer block with shifted window attention."""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(window_size, window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            qk_scale=None, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        
        # Shift window
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        
        # Reverse window partition
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            
        x = x.view(B, H * W, C)
        x = shortcut + x
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        return x

class WindowAttention(nn.Module):
    """Window based multi-head self attention."""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """MLP as used in Vision Transformer."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """Partition input into windows."""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """Reverse window partition."""
    B = int(windows.shape[0] / ((H / window_size) * (W / window_size)))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x

class SwinTransformer(nn.Module):
    """Custom Swin Transformer implementation."""
    def __init__(self, img_size=1024, patch_size=4, in_chans=2, num_classes=0,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.n_patches
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build layers
        self.layers = nn.ModuleList()
        for i_layer in range(len(depths)):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer)
            self.layers.append(layer)
        
        # Calculate the final feature dimension
        self.num_features = int(embed_dim * 2 ** (len(depths)))
        
        # Update norm layer
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        H = W = int(np.sqrt(self.patch_embed.n_patches))
        
        # Store features and dimensions at each layer
        features = []
        
        # Process through layers
        for i, layer in enumerate(self.layers):
            x, H, W = layer(x, H, W)
            
            if i == len(self.layers) - 1:
                # Last layer - apply normalization
                B, L, C = x.shape
                assert C == self.num_features, f"Feature dim mismatch: {C} vs {self.num_features}"
                x = self.norm(x)
                x = self.head(x)
                
        return x

class BasicLayer(nn.Module):
    """Basic Swin Transformer layer."""
    def __init__(self, dim, depth, num_heads, window_size, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = window_size // 2
        
        # Build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])
        
        # Patch merging layer
        self.downsample = PatchMerging(dim=dim, norm_layer=norm_layer)
        
    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        x = self.downsample(x, H, W)
        return x, H//2, W//2  # Return the new H, W dimensions after downsampling

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
            img_size=1024,
            patch_size=4,
            in_chans=2,  # DEM and rainfall channels
            num_classes=0,  # No classification head
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=8,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1
        )
        
        # Get the correct feature dimension from the model
        feature_dim = self.model.num_features
        print(f"Feature dimension: {feature_dim}")
        
        # Modify regression head to handle the correct dimension
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1024),  # Output size for a 32x32 prediction
        )
        
        # Upsampling layers to get back to the original resolution (1024x1024)
        self.upsampling = nn.Sequential(
            nn.Unflatten(1, (1, 32, 32)),  # Reshape to [B, 1, 32, 32]
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)  # Upsample to 1024x1024
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
        features = self.model(x)
        
        # Global pooling to get a fixed-size representation
        B, L, C = features.shape
        features = features.mean(dim=1)  # [B, C]
        
        # Apply regression head
        output = self.regression_head(features)
        
        # Upsample to original resolution
        output = self.upsampling(output)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # Calculate MSE loss
        loss = self.criterion(y_hat, y)
        
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
        # self.val_r2(y_hat, y)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mse', self.val_mse, prog_bar=True)
        self.log('val_mae', self.val_mae, prog_bar=True)
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
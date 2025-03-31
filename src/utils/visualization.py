import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

def plot_sample_prediction(inputs, target, prediction, sample_idx=0):
    """
    Plot sample prediction with inputs and ground truth
    
    Args:
        inputs (torch.Tensor): Input tensor with shape [B, C, H, W]
        target (torch.Tensor): Target tensor with shape [B]
        prediction (torch.Tensor): Prediction tensor with shape [B]
        sample_idx (int): Sample index to plot
    
    Returns:
        fig: Matplotlib figure
    """
    rainfall = inputs[sample_idx, 0].cpu().numpy()
    dem = inputs[sample_idx, 1].cpu().numpy()
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot rainfall
    im0 = axs[0].imshow(rainfall, cmap='Blues')
    axs[0].set_title('Rainfall')
    plt.colorbar(im0, ax=axs[0])
    
    # Plot DEM
    im1 = axs[1].imshow(dem, cmap='terrain')
    axs[1].set_title('Digital Elevation Model (DEM)')
    plt.colorbar(im1, ax=axs[1])
    
    # Plot comparison
    axs[2].bar(['Target', 'Prediction'], 
             [target[sample_idx].item(), prediction[sample_idx].item()],
             color=['blue', 'orange'])
    axs[2].set_ylim([0, 1])
    axs[2].set_title('Flood Prediction')
    
    plt.tight_layout()
    return fig

def log_to_wandb(trainer, model, dataloader, num_samples=4):
    """
    Log sample predictions to WandB
    
    Args:
        trainer: PyTorch Lightning trainer
        model: Model to make predictions
        dataloader: DataLoader to get samples from
        num_samples: Number of samples to log
    """
    model.eval()
    batch = next(iter(dataloader))
    inputs, targets = batch
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        logits = model(inputs)
        probs = torch.sigmoid(logits.squeeze())
    
    for i in range(min(num_samples, len(inputs))):
        fig = plot_sample_prediction(inputs, targets, probs, sample_idx=i)
        wandb.log({f"sample_{i}": wandb.Image(fig)})
        plt.close(fig)
        
def visualize_feature_maps(model, inputs, layer_name, sample_idx=0):
    """
    Visualize feature maps from a specific layer
    
    Args:
        model: PyTorch model
        inputs: Input tensor
        layer_name: Name of the layer to visualize
        sample_idx: Sample index to visualize
    
    Returns:
        fig: Matplotlib figure with feature maps
    """
    # Placeholder for feature map visualization
    # In a real implementation, you would register hooks to extract feature maps
    
    # For illustration, just create random feature maps
    feature_maps = torch.rand(8, 16, 16)  # Example feature maps [C, H, W]
    
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    axs = axs.flatten()
    
    for i in range(min(len(feature_maps), 8)):
        axs[i].imshow(feature_maps[i].cpu().numpy(), cmap='viridis')
        axs[i].set_title(f"Channel {i}")
        axs[i].axis('off')
    
    plt.tight_layout()
    return fig 
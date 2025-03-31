import os
import argparse
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from models.flood_model import FloodPredictionModel
from pathlib import Path
import rasterio
from matplotlib.colors import Normalize
from tqdm import tqdm

def load_checkpoint(checkpoint_path):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        model: Loaded PyTorch model
    """
    model = FloodPredictionModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.freeze()
    
    return model

def prepare_input(dem_path, rainfall_value, normalize=True, dem_mean=None, dem_std=None, rainfall_mean=None, rainfall_std=None):
    """
    Prepare input for the model from DEM file and rainfall value.
    
    Args:
        dem_path (str): Path to the DEM TIF file
        rainfall_value (float): Rainfall value in mm
        normalize (bool): Whether to normalize the inputs
        dem_mean (float): Mean value for DEM normalization
        dem_std (float): Standard deviation for DEM normalization
        rainfall_mean (float): Mean value for rainfall normalization
        rainfall_std (float): Standard deviation for rainfall normalization
        
    Returns:
        torch.Tensor: Input tensor for the model [1, 2, H, W]
    """
    # Load DEM data
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
    
    # Convert to torch tensors
    dem_tensor = torch.from_numpy(dem).float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    rainfall_tensor = torch.tensor(rainfall_value).float()
    
    # Create rainfall channel (same value at all pixels)
    rainfall_channel = torch.ones_like(dem_tensor) * rainfall_tensor
    
    # Normalize if required
    if normalize:
        if dem_mean is None:
            # Use default values if not provided
            dem_mean = dem_tensor.mean().item()
            dem_std = dem_tensor.std().item() or 1.0
            rainfall_mean = 50.0  # Arbitrary default
            rainfall_std = 25.0   # Arbitrary default
            
        # Normalize DEM
        dem_tensor = (dem_tensor - dem_mean) / (dem_std + 1e-8)
        
        # Normalize rainfall
        rainfall_tensor = (rainfall_tensor - rainfall_mean) / (rainfall_std + 1e-8)
        rainfall_channel = torch.ones_like(dem_tensor) * rainfall_tensor
    
    # Combine DEM and rainfall channels
    input_tensor = torch.cat([dem_tensor, rainfall_channel], dim=1)  # [1, 2, H, W]
    
    return input_tensor

def predict_from_h5(model, h5_file, city_id, rainfall_type, device='cuda'):
    """
    Make a prediction for a city and rainfall type from the H5 file.
    
    Args:
        model: PyTorch model
        h5_file (str): Path to the H5 file
        city_id (str): City ID
        rainfall_type (str): Type of rainfall ('100-yr', '50-yr', '25-yr', '10-yr')
        device (str): Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        tuple: (DEM array, actual max depth array, predicted max depth array, rainfall value)
    """
    with h5py.File(h5_file, 'r') as f:
        # Find the split containing the city
        split = None
        for s in ['train', 'val', 'test']:
            if city_id in f[s]:
                split = s
                break
        
        if split is None:
            raise ValueError(f"City {city_id} not found in any split in the H5 file")
        
        # Get DEM data
        dem = f[split][city_id]['dem'][:]
        
        # Get max flood depth data and rainfall value
        max_depth = f[split][city_id][rainfall_type]['max_flood_depth'][:]
        rainfall_value = f[split][city_id][rainfall_type].attrs['rainfall_value']
    
    # Prepare input
    input_tensor = prepare_input(
        dem_path=None,  # Not used in this case
        rainfall_value=rainfall_value,
        normalize=True,
        dem_mean=dem.mean(),
        dem_std=dem.std(),
        rainfall_mean=50.0,  # Using defaults
        rainfall_std=25.0
    )
    
    # Replace the DEM channel with the actual DEM data
    dem_tensor = torch.from_numpy(dem).float().unsqueeze(0).unsqueeze(0)
    input_tensor[:, 0, :, :] = (dem_tensor - dem.mean()) / (dem.std() + 1e-8)
    
    # Move input to device
    input_tensor = input_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert to numpy
    prediction = output.cpu().numpy().squeeze()
    
    return dem, max_depth, prediction, rainfall_value

def visualize_prediction(dem, actual, prediction, city_id, rainfall_value, output_dir="predictions"):
    """
    Visualize the prediction alongside actual max depth and DEM.
    
    Args:
        dem (numpy.ndarray): DEM array
        actual (numpy.ndarray): Actual max flood depth array
        prediction (numpy.ndarray): Predicted max flood depth array
        city_id (str): City ID
        rainfall_value (float): Rainfall value in mm
        output_dir (str): Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure with 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(18, 12))
    
    # Plot DEM
    im0 = axs[0, 0].imshow(dem, cmap='terrain')
    axs[0, 0].set_title(f'DEM - {city_id}')
    plt.colorbar(im0, ax=axs[0, 0], label='Elevation (m)')
    
    # Plot actual max flood depth
    im1 = axs[0, 1].imshow(actual, cmap='Blues')
    axs[0, 1].set_title(f'Actual Max Flood Depth - Rainfall: {rainfall_value} mm')
    plt.colorbar(im1, ax=axs[0, 1], label='Depth (m)')
    
    # Plot predicted max flood depth
    im2 = axs[1, 0].imshow(prediction, cmap='Blues')
    axs[1, 0].set_title('Predicted Max Flood Depth')
    plt.colorbar(im2, ax=axs[1, 0], label='Depth (m)')
    
    # Plot difference (error)
    diff = prediction - actual
    im3 = axs[1, 1].imshow(diff, cmap='RdBu', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
    axs[1, 1].set_title('Prediction Error (Predicted - Actual)')
    plt.colorbar(im3, ax=axs[1, 1], label='Error (m)')
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_dir, f'{city_id}_{rainfall_value}mm.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Visualization saved to {output_file}")
    
    # Calculate and return metrics
    mae = np.mean(np.abs(diff))
    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    
    return mae, rmse

def main():
    parser = argparse.ArgumentParser(description='Run inference with a trained flood prediction model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--h5_file', type=str, default='flood_data.h5', help='Path to the H5 file')
    parser.add_argument('--city', type=str, default=None, help='City ID to predict for (if None, predict for all test cities)')
    parser.add_argument('--rainfall', type=str, default=None, help='Rainfall type to predict for (if None, predict for all rainfall types)')
    parser.add_argument('--output_dir', type=str, default='predictions', help='Directory to save predictions')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_checkpoint(args.checkpoint)
    model.to(args.device)
    
    # Get cities and rainfall types
    with h5py.File(args.h5_file, 'r') as f:
        if args.city is None:
            # Use all test cities
            cities = list(f['test'].keys())
        else:
            cities = [args.city]
        
        if args.rainfall is None:
            # Use all rainfall types for the first city
            rainfall_types = []
            for split in ['train', 'val', 'test']:
                if cities[0] in f[split]:
                    rainfall_types = [rt for rt in f[split][cities[0]] if rt != 'dem']
                    break
        else:
            rainfall_types = [args.rainfall]
    
    # Create results dataframe
    results = []
    
    # Run inference for each city and rainfall type
    for city_id in cities:
        for rainfall_type in rainfall_types:
            print(f"Predicting for {city_id}, {rainfall_type}...")
            
            try:
                # Make prediction
                dem, actual, prediction, rainfall_value = predict_from_h5(
                    model, args.h5_file, city_id, rainfall_type, args.device
                )
                
                # Visualize prediction
                mae, rmse = visualize_prediction(
                    dem, actual, prediction, city_id, rainfall_value, args.output_dir
                )
                
                # Add to results
                results.append({
                    'city_id': city_id,
                    'rainfall_type': rainfall_type,
                    'rainfall_value': rainfall_value,
                    'mae': mae,
                    'rmse': rmse
                })
                
                print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}")
                
            except Exception as e:
                print(f"Error predicting for {city_id}, {rainfall_type}: {e}")
    
    # Print overall results
    if results:
        print("\nOverall Results:")
        mae_avg = sum(r['mae'] for r in results) / len(results)
        rmse_avg = sum(r['rmse'] for r in results) / len(results)
        print(f"Average MAE: {mae_avg:.4f}")
        print(f"Average RMSE: {rmse_avg:.4f}")

if __name__ == "__main__":
    main() 
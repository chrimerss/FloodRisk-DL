import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
from datetime import datetime
import argparse
from pathlib import Path
import joblib

# Add parent directory to path for imports
sys.path.append('/home/users/li1995/global_flood/FloodRisk-DL/terratorch')

from test import (
    load_model, classify_depths, calc_slope, load_full_dataset, 
    calculate_jaccard_scores, extract_rainfall_levels, 
    run_bathtub_model, RAINFALL_DICT, FLOOD_COLORS, FloodCategory
)

# Import necessary TerraTorch components
from terratorch.tasks import SemanticSegmentationTask

# Import model arguments
from task_class import (
    model_args_res50, model_args_res101, model_args_res152, 
    model_args_tiny, model_args_100, model_args_300, model_args_600
)

# Import model paths
from model_pth import FloodCategory as ModelPaths

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_meta_model(meta_model_path):
    """
    Load the trained meta-model.
    
    Args:
        meta_model_path: Path to the saved meta-model
        
    Returns:
        Loaded meta-model
    """
    print(f"Loading meta-model from: {meta_model_path}")
    try:
        meta_model = joblib.load(meta_model_path)
        print("Meta-model loaded successfully.")
        print(f"Model type: {type(meta_model).__name__}")
        return meta_model
    except Exception as e:
        print(f"Error loading meta-model: {str(e)}")
        return None

def load_base_models():
    """
    Load all base deep learning models.
    
    Returns:
        Dictionary of models {model_name: model}
    """
    models = {}
    # Define model configurations
    model_config = {
        'RES50': model_args_res50,
        'RES101': model_args_res101,
        'RES152': model_args_res152,
        'TINY': model_args_tiny,
        '100M': model_args_100,
        '300M': model_args_300,
        '600M': model_args_600
    }
    
    # Load each model
    for model_name, model_args in model_config.items():
        try:
            model_path = getattr(ModelPaths, model_name).value
            if os.path.exists(model_path):
                print(f"Loading {model_name} model...")
                models[model_name] = load_model(model_path, model_args, DEVICE)
            else:
                print(f"Model path not found for {model_name}: {model_path}")
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
    
    print(f"Loaded {len(models)} base models.")
    return models

def get_model_predictions(domain, rainfall_level, models):
    """
    Get predictions from all base models for a given domain and rainfall level.
    
    Args:
        domain: Domain name (e.g., 'HOU001')
        rainfall_level: Rainfall level (e.g., '100mm')
        models: Dictionary of loaded models
        
    Returns:
        Dictionary with predictions and metadata
    """
    try:
        # Load the dataset
        print(f"Loading domain {domain} with rainfall {rainfall_level}...")
        dem, slope, rainfall, flood_cat, target, dem_mean, dem_std = load_full_dataset(domain, rainfall_level)
        
        height, width = dem.shape
        print(f"Dataset dimensions: {height} x {width}")
        
        # Initialize dictionary of predictions
        predictions = {}
        probs = {}  # For storing class probabilities
        
        # Process each ML model using fixed 512x512 moving window
        window_size = 512  # Fixed window size for processing
        
        for model_name, model in models.items():
            try:
                print(f"  Running {model_name} model with 512x512 moving window...")
                # Initialize output arrays
                pred_cat = np.zeros((height, width), dtype=np.int64)
                probs_out = np.zeros((5, height, width), dtype=np.float32)
                
                # Process image in 512x512 windows with overlap
                for y in range(0, height, window_size):
                    y_end = min(y + window_size, height)
                    
                    for x in range(0, width, window_size):
                        x_end = min(x + window_size, width)
                        
                        # Extract window
                        dem_window = dem[y:y_end, x:x_end]
                        slope_window = slope[y:y_end, x:x_end]
                        rainfall_window = rainfall[y:y_end, x:x_end]
                        
                        # Prepare input tensor
                        model_input = np.stack([dem_window, slope_window, rainfall_window])
                        input_tensor = torch.from_numpy(model_input).unsqueeze(0).to(DEVICE)
                        
                        # Get prediction
                        with torch.no_grad():
                            output = model(input_tensor)
                            pred = output.output.squeeze().detach().cpu().numpy()
                            window_pred_cat = pred.argmax(axis=0)
                            
                            # Store predictions and probabilities in the full array
                            pred_cat[y:y_end, x:x_end] = window_pred_cat
                            
                            # Handle different shapes of probabilities
                            if len(pred.shape) == 3:  # (classes, height, width)
                                probs_out[:, y:y_end, x:x_end] = pred
                            else:  # Handle other formats if needed
                                for c in range(min(5, pred.shape[0])):
                                    probs_out[c, y:y_end, x:x_end] = pred[c]
                        
                        # Clear CUDA cache to free memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                # Store results
                predictions[model_name] = pred_cat
                probs[model_name] = probs_out
                
            except Exception as e:
                print(f"  Error running model {model_name}: {e}")
                import traceback
                traceback.print_exc()
                # Initialize with zeros if model fails
                predictions[model_name] = np.zeros_like(flood_cat, dtype=np.int64)
                probs[model_name] = np.zeros((5, height, width), dtype=np.float32)
        
        # Calculate jaccard scores and F1 scores
        jaccard_scores = {}
        for model_name, pred in predictions.items():
            jaccard_scores[model_name] = calculate_jaccard_scores(flood_cat, pred)
        
        return {
            'dem': dem,
            'slope': slope,
            'rainfall': rainfall,
            'flood_cat': flood_cat,
            'target': target,
            'predictions': predictions,
            'probabilities': probs,
            'jaccard_scores': jaccard_scores,
            'dem_mean': dem_mean,
            'dem_std': dem_std
        }
    
    except Exception as e:
        print(f"Error getting predictions for {domain}, {rainfall_level}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'dem': np.array([]),
            'slope': np.array([]),
            'rainfall': np.array([]),
            'flood_cat': np.array([]),
            'target': np.array([]),
            'predictions': {},
            'probabilities': {},
            'jaccard_scores': {},
            'dem_mean': 0,
            'dem_std': 1
        }

def extract_features(pred_result):
    """
    Extract features for meta-model inference from base model predictions.
    
    Args:
        pred_result: Dictionary with prediction results
        
    Returns:
        Feature array for meta-model input
    """
    # Extract base data and probabilities
    dem = pred_result['dem']
    slope = pred_result['slope']
    rainfall = pred_result['rainfall']
    probs = pred_result['probabilities']
    
    height, width = dem.shape
    
    # We want to extract features for the entire image
    # Features: DEM, slope, rainfall, model probabilities
    num_base_features = 3  # DEM, slope, rainfall
    
    # Get available DL models
    dl_models = list(probs.keys())
    num_model_features = len(dl_models) * 5  # 5 class probabilities per model
    num_features = num_base_features + num_model_features
    
    # Reshape the input data to a 2D array (pixels x features)
    X = np.zeros((height * width, num_features), dtype=np.float32)
    
    # Add base features (first 3 columns)
    X[:, 0] = dem.flatten()
    X[:, 1] = slope.flatten()
    X[:, 2] = rainfall.flatten()
    
    # Add model probabilities (remaining columns)
    feature_idx = num_base_features
    for model_name in dl_models:
        model_probs = probs[model_name]
        for class_idx in range(5):
            X[:, feature_idx] = model_probs[class_idx].flatten()
            feature_idx += 1
    
    return X

def run_meta_model_inference(domain, rainfall_level, meta_model, base_models, output_dir):
    """
    Run inference using the meta-model on a specific domain and rainfall level.
    
    Args:
        domain: Domain name
        rainfall_level: Rainfall level
        meta_model: Loaded meta-model
        base_models: Dictionary of loaded base models
        output_dir: Directory to save results
        
    Returns:
        Dictionary with meta-model evaluation results
    """
    print(f"\nRunning meta-model inference on {domain}, rainfall {rainfall_level}")
    
    # Get base model predictions
    pred_result = get_model_predictions(domain, rainfall_level, base_models)
    
    if pred_result['dem'].size == 0:
        print(f"No valid predictions for {domain}, {rainfall_level}. Skipping.")
        return None
    
    # Extract features for meta-model
    print("Extracting features for meta-model...")
    X = extract_features(pred_result)
    
    # Run meta-model prediction
    print("Running meta-model prediction...")
    height, width = pred_result['dem'].shape
    try:
        start_time = time.time()
        meta_pred = meta_model.predict(X)
        inference_time = time.time() - start_time
        print(f"Meta-model prediction completed in {inference_time:.2f} seconds")
        
        # Reshape prediction to 2D array
        meta_pred = meta_pred.reshape(height, width)
        
        # Calculate meta-model performance metrics
        meta_jaccard = calculate_jaccard_scores(pred_result['flood_cat'], meta_pred)
        
        # Store meta prediction in results
        pred_result['meta_prediction'] = meta_pred
        pred_result['meta_jaccard'] = meta_jaccard
        
        # Create and save visualization
        create_visualization(domain, rainfall_level, pred_result, meta_jaccard, output_dir)
        
        # Print binary jaccard and F1 score results
        print("\nMeta-model performance:")
        print(f"Binary Jaccard: {meta_jaccard['binary']:.4f}")
        print(f"Binary F1 Score: {meta_jaccard['binary_f1']:.4f}")
        
        # Print performance of all base models for comparison
        print("\nBase model performance:")
        for model_name, jaccard in pred_result['jaccard_scores'].items():
            print(f"{model_name} - Binary Jaccard: {jaccard['binary']:.4f}, Binary F1: {jaccard['binary_f1']:.4f}")
        
        return pred_result
    
    except Exception as e:
        print(f"Error in meta-model inference: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_visualization(domain, rainfall_level, pred_result, meta_jaccard, output_dir):
    """
    Create visualization comparing meta-model with base models.
    
    Args:
        domain: Domain name
        rainfall_level: Rainfall level
        pred_result: Dictionary with prediction results
        meta_jaccard: Meta-model Jaccard scores
        output_dir: Directory to save visualizations
    """
    print(f"Creating visualization for {domain}, {rainfall_level}")
    
    try:
        # Create colormap for flood categories
        flood_colors = [FLOOD_COLORS[FloodCategory(i)] for i in range(len(FloodCategory))]
        cmap = ListedColormap(flood_colors)
        
        # Get data
        flood_cat = pred_result['flood_cat']
        meta_pred = pred_result['meta_prediction']
        base_preds = pred_result['predictions']
        
        # Check if the visualization would be too large
        height, width = flood_cat.shape
        total_pixels = height * width
        max_viz_pixels = 2000000  # 2 million pixels max for visualization
        
        # If the image is too large, downsample it
        if total_pixels > max_viz_pixels:
            # Calculate downsample ratio
            downsample_ratio = int(np.ceil(np.sqrt(total_pixels / max_viz_pixels)))
            print(f"Image too large for visualization ({height}x{width}). Downsampling by factor of {downsample_ratio}.")
            
            # Downsample data
            flood_cat = flood_cat[::downsample_ratio, ::downsample_ratio]
            meta_pred = meta_pred[::downsample_ratio, ::downsample_ratio]
            
            # Downsample base predictions
            downsampled_base_preds = {}
            for model_name, pred in base_preds.items():
                downsampled_base_preds[model_name] = pred[::downsample_ratio, ::downsample_ratio]
            base_preds = downsampled_base_preds
        
        # Calculate how many subplots we need
        n_models = len(base_preds)
        total_plots = n_models + 2  # +2 for ground truth and meta-model
        
        # Limit the number of subplots to avoid memory issues
        max_plots = 9  # Max 9 subplots (3x3 grid)
        if total_plots > max_plots:
            print(f"Too many models to visualize ({n_models}). Limiting to {max_plots-2} top models.")
            
            # Find the best performing models based on binary F1 score
            model_scores = [(model, scores['binary_f1']) for model, scores in pred_result['jaccard_scores'].items()]
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Keep only the top models
            top_models = [model for model, _ in model_scores[:max_plots-2]]
            
            # Filter base_preds to only include top models
            base_preds = {model: pred for model, pred in base_preds.items() if model in top_models}
            
            # Update n_models and total_plots
            n_models = len(base_preds)
            total_plots = n_models + 2
        
        # Create a grid of subplots
        ncols = min(3, total_plots)
        nrows = (total_plots + ncols - 1) // ncols
        
        # Create figure with smaller dpi to reduce memory usage
        plt.figure(figsize=(ncols*4, nrows*3), dpi=100)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3), dpi=100)
        axes = axes.flatten() if total_plots > 1 else [axes]
        
        # Hide any unused subplots
        for i in range(total_plots, len(axes)):
            axes[i].axis('off')
        
        # Plot ground truth
        axes[0].set_title("Ground Truth")
        axes[0].imshow(flood_cat, cmap=cmap, vmin=0, vmax=4)
        
        # Plot meta-model prediction
        binary_f1 = meta_jaccard['binary_f1']
        axes[1].set_title(f"Meta-Model - F1: {binary_f1:.3f}")
        axes[1].imshow(meta_pred, cmap=cmap, vmin=0, vmax=4)
        
        # Plot base model predictions
        for i, (model_name, pred) in enumerate(base_preds.items()):
            idx = i + 2  # Offset for ground truth and meta-model
            binary_f1 = pred_result['jaccard_scores'][model_name]['binary_f1']
            axes[idx].set_title(f"{model_name} - F1: {binary_f1:.3f}")
            axes[idx].imshow(pred, cmap=cmap, vmin=0, vmax=4)
        
        # Add overall title
        fig.suptitle(f"Model Comparison: {domain}, Rainfall: {rainfall_level}", fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'meta_inference_{domain}_{rainfall_level}.png')
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close('all')  # Close all figures to free memory
        
        print(f"Saved visualization to {save_path}")
        
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
        import traceback
        traceback.print_exc()

def save_results(domain, rainfall_level, pred_result, output_dir):
    """
    Save inference results to CSV files.
    
    Args:
        domain: Domain name
        rainfall_level: Rainfall level
        pred_result: Dictionary with prediction results
        output_dir: Directory to save results
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract jaccard scores
        jaccard_data = []
        
        # Add meta-model row
        meta_jaccard = pred_result['meta_jaccard']
        jaccard_data.append({
            'Domain': domain,
            'Rainfall': rainfall_level,
            'Model': 'Meta-Model',
            'Binary_Jaccard': meta_jaccard['binary'],
            'NoFlood_Jaccard': meta_jaccard['no_flood'],
            'Nuisance_Jaccard': meta_jaccard['nuisance'],
            'Minor_Jaccard': meta_jaccard['minor'],
            'Medium_Jaccard': meta_jaccard['medium'],
            'Major_Jaccard': meta_jaccard['major'],
            'Binary_F1': meta_jaccard['binary_f1'],
            'NoFlood_F1': meta_jaccard['no_flood_f1'],
            'Nuisance_F1': meta_jaccard['nuisance_f1'],
            'Minor_F1': meta_jaccard['minor_f1'],
            'Medium_F1': meta_jaccard['medium_f1'],
            'Major_F1': meta_jaccard['major_f1']
        })
        
        # Add rows for each base model
        for model_name, model_jaccard in pred_result['jaccard_scores'].items():
            jaccard_data.append({
                'Domain': domain,
                'Rainfall': rainfall_level,
                'Model': model_name,
                'Binary_Jaccard': model_jaccard['binary'],
                'NoFlood_Jaccard': model_jaccard['no_flood'],
                'Nuisance_Jaccard': model_jaccard['nuisance'],
                'Minor_Jaccard': model_jaccard['minor'],
                'Medium_Jaccard': model_jaccard['medium'],
                'Major_Jaccard': model_jaccard['major'],
                'Binary_F1': model_jaccard['binary_f1'],
                'NoFlood_F1': model_jaccard['no_flood_f1'],
                'Nuisance_F1': model_jaccard['nuisance_f1'],
                'Minor_F1': model_jaccard['minor_f1'],
                'Medium_F1': model_jaccard['medium_f1'],
                'Major_F1': model_jaccard['major_f1']
            })
        
        # Save to CSV
        jaccard_df = pd.DataFrame(jaccard_data)
        csv_path = os.path.join(output_dir, f'meta_inference_results_{domain}_{rainfall_level}_{timestamp}.csv')
        jaccard_df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")
        
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def main(args):
    """Main function to run meta-model inference."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"meta_inference_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load meta-model
    meta_model = load_meta_model(args.meta_model)
    if meta_model is None:
        print("Failed to load meta-model. Exiting.")
        return
    
    # Load base models
    base_models = load_base_models()
    if len(base_models) == 0:
        print("Failed to load any base models. Exiting.")
        return
    
    # Get list of domains to process
    domains = args.domains.split(',') if args.domains else ["HOU001", "HOU007", "SF001", "SF002", "NYC002", "LA002", "DAL002", "AUS002", "MIA002"]
    
    # Get list of rainfall levels to process
    if args.rainfall_levels:
        rainfall_levels = args.rainfall_levels.split(',')
    else:
        # Use all available rainfall levels for the first domain
        if domains:
            rainfall_levels = extract_rainfall_levels(domains[0])
        else:
            rainfall_levels = ["100mm", "150mm", "200mm", "250mm", "300mm"]
    
    print(f"Processing domains: {', '.join(domains)}")
    print(f"Processing rainfall levels: {', '.join(rainfall_levels)}")
    
    # Process each domain and rainfall level
    for domain in domains:
        for rainfall_level in rainfall_levels:
            print(f"\n{'='*80}")
            print(f"Processing domain: {domain}, rainfall: {rainfall_level}")
            print(f"{'='*80}")
            
            result = run_meta_model_inference(domain, rainfall_level, meta_model, base_models, output_dir)
            
            if result:
                save_results(domain, rainfall_level, result, output_dir)
                
                # Clear memory
                del result
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    print("\nInference completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference using a trained meta-model')
    parser.add_argument('--meta_model', type=str, 
                        default='/home/users/li1995/global_flood/FloodRisk-DL/terratorch/meta_learning_results_20250512_101047/meta_model_random_forest.joblib',
                        help='Path to the trained meta-model')
    parser.add_argument('--output_dir', type=str, default='./meta_inference_results',
                        help='Directory to save inference results')
    parser.add_argument('--domains', type=str, default=None,
                        help='Comma-separated list of domains to process (e.g., HOU001,SF001)')
    parser.add_argument('--rainfall_levels', type=str, default=None,
                        help='Comma-separated list of rainfall levels to process (e.g., 100mm,200mm)')
    
    args = parser.parse_args()
    main(args)

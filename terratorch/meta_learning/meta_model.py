# coding: utf-8

import os
import sys
sys.path.append('/home/users/li1995/global_flood/FloodRisk-DL/terratorch')
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from enum import Enum
from collections import defaultdict
import pickle

# ML libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import jaccard_score, f1_score
import joblib

# Import necessary TerraTorch components
from terratorch.tasks import SemanticSegmentationTask


from test_metric import (
    load_model, classify_depths, calc_slope, load_full_dataset, 
    calculate_jaccard_scores, extract_rainfall_levels, 
    run_bathtub_model, RAINFALL_DICT, FLOOD_COLORS, FloodCategory
)

# Import model arguments
from task_class import (
    model_args_res50, model_args_res101, model_args_res152, 
    model_args_tiny, model_args_100, model_args_300, model_args_600
)

# Import model paths
from model_pth import FloodCategory as ModelPaths

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MetaLearningModel:
    def __init__(self, train_domains=None, output_dir=None):
        """
        Initialize the meta-learning model.
        
        Args:
            train_domains: List of domains to use for training
            output_dir: Directory to save results
        """
        self.device = DEVICE
        print(f"Using device: {self.device}")
        
        # Set training domains (cities)
        self.train_domains = train_domains or ["HOU001", "SF001", "NYC002", "LA002"]
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or os.path.join(os.path.dirname(os.path.dirname(__file__)), f'meta_learning_results_{timestamp}')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Dictionary to store model paths and names
        self.model_paths = {
            'RES50': ModelPaths.MODEL_RES50.value,
            'RES101': ModelPaths.MODEL_RES101.value,
            'RES152': ModelPaths.MODEL_RES152.value,
            'TINY': ModelPaths.MODEL_TINY.value,
            '100M': ModelPaths.MODEL_100M.value,
            '300M': ModelPaths.MODEL_300M.value,
            '600M': ModelPaths.MODEL_600M.value
        }
        
        # Initialize models dictionary
        self.models = {}
        
        # Initialize the meta-model
        self.meta_model = None
        
        # Data storage for meta-learning
        self.training_data = {
            'features': [],
            'labels': []
        }
        
        # Cache for storing model predictions
        self.model_cache = {}
        
    def load_base_models(self):
        """Load all base models."""
        print("Loading base models...")
        
        for model_name, model_path in self.model_paths.items():
            try:
                print(f"Loading model: {model_name}")
                # Get the appropriate model args based on model name
                if model_name == 'RES50':
                    model_args = model_args_res50
                elif model_name == 'RES101':
                    model_args = model_args_res101
                elif model_name == 'RES152':
                    model_args = model_args_res152
                elif model_name == 'TINY':
                    model_args = model_args_tiny
                elif model_name == '100M':
                    model_args = model_args_100
                elif model_name == '300M':
                    model_args = model_args_300
                elif model_name == '600M':
                    model_args = model_args_600
                else:
                    raise ValueError(f"Unknown model type: {model_name}")
                
                # Load the model with the appropriate model args
                self.models[model_name] = load_model(model_path, model_name)
                self.models[model_name] = self.models[model_name].to(self.device)
            except Exception as e:
                print(f"  Error loading model {model_name}: {e}")
                
        print(f"Loaded {len(self.models)} models successfully")
    
    def _get_rainfall_levels(self, domain):
        """Get rainfall levels for a domain."""
        return extract_rainfall_levels(domain, RAINFALL_DICT)
    
    def _get_model_predictions(self, domain, rainfall_level):
        """
        Get predictions from all models for a domain and rainfall level.
        
        Args:
            domain: The domain to use (e.g., 'HOU001')
            rainfall_level: The rainfall level to use (e.g., '98mm')
            
        Returns:
            Dictionary of model predictions and related data
        """
        # Check if we have cached predictions for this domain + rainfall level
        cache_key = f"{domain}_{rainfall_level}"
        if cache_key in self.model_cache:
            print(f"Using cached predictions for {domain}, {rainfall_level}")
            return self.model_cache[cache_key]
        
        print(f"Getting predictions for {domain}, {rainfall_level}")
        
        try:
            # Load dataset with error handling
            dem, slope, rainfall, flood_cat, target, dem_mean, dem_std = load_full_dataset(domain, rainfall_level)
        except FileNotFoundError as e:
            print(f"Error loading dataset: {str(e)}")
            print("Returning empty prediction data")
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
        
        height, width = dem.shape
        print(f"Dataset dimensions: {height} x {width}")
        
        # Initialize dictionary of predictions
        predictions = {}
        probs = {}  # For storing class probabilities
        
        # Skip bathtub model - Removed bathtub model run
        
        # Process each ML model using fixed 512x512 moving window
        window_size = 512  # Fixed window size for processing
        
        for model_name, model in self.models.items():
            try:
                print(f"  Running {model_name} model with 512x512 moving window...")
                
                # Initialize output arrays for full prediction area
                pred_cat = np.zeros((height, width), dtype=np.int64)
                probs_out = np.zeros((5, height, width), dtype=np.float32)  # 5 classes
                
                # Process in fixed 512x512 windows with stride
                # Use stride of 512 to ensure non-overlapping windows for memory efficiency
                for y in range(0, height, window_size):
                    y_end = min(y + window_size, height)
                    
                    # Skip incomplete windows at the end if they are too small
                    if (y_end - y) < window_size // 4:
                        continue
                        
                    for x in range(0, width, window_size):
                        x_end = min(x + window_size, width)
                        
                        # Skip incomplete windows at the end if they are too small
                        if (x_end - x) < window_size // 4:
                            continue
                        
                        # print(f"    Processing window ({y}:{y_end}, {x}:{x_end})")
                        
                        # Extract window
                        dem_window = dem[y:y_end, x:x_end]
                        slope_window = slope[y:y_end, x:x_end]
                        rainfall_window = rainfall[y:y_end, x:x_end]
                        
                        # Prepare input tensor
                        model_input = np.stack([dem_window, slope_window, rainfall_window])
                        input_tensor = torch.from_numpy(model_input).unsqueeze(0).to(self.device)
                        
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
        
        # Calculate jaccard scores
        jaccard_scores = {}
        for model_name, pred in predictions.items():
            jaccard_scores[model_name] = calculate_jaccard_scores(flood_cat, pred)
        
        # Store results in cache
        self.model_cache[cache_key] = {
            'dem': dem,
            'slope': slope,
            'rainfall': rainfall,
            'flood_cat': flood_cat,
            'target': target,  # Store full target
            'predictions': predictions,
            'probabilities': probs,
            'jaccard_scores': jaccard_scores,
            'dem_mean': dem_mean,
            'dem_std': dem_std
        }
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return self.model_cache[cache_key]
    
    def _extract_pixel_features(self, result, sample_size=10000):
        """
        Extract features for each pixel for meta-learning.
        
        Args:
            result: Dictionary containing model predictions and data
            sample_size: Number of pixels to sample (use -1 for all pixels)
            
        Returns:
            features_array, labels_array
        """
        dem = result['dem']
        slope = result['slope']
        rainfall = result['rainfall']
        flood_cat = result['flood_cat']
        probs = result['probabilities']
        
        # Check if we have valid data
        if dem.size == 0:
            print("Warning: Empty data arrays. Returning empty features.")
            return np.array([]), np.array([])
        
        height, width = dem.shape
        total_pixels = height * width
        
        # Determine sampling strategy
        if sample_size <= 0 or sample_size >= total_pixels:
            # Use all pixels
            sample_indices = np.arange(total_pixels)
            sample_size = total_pixels
        else:
            # Stratified sampling to ensure we get enough samples of each flood category
            # This ensures better representation of rarer flood categories
            indices_by_category = {}
            for cat in range(5):  # 0 to 4 for flood categories
                cat_indices = np.where(flood_cat.flatten() == cat)[0]
                indices_by_category[cat] = cat_indices
            
            # Calculate how many samples to take from each category
            # Take samples proportional to square root of category frequency to boost rare classes
            cat_counts = {cat: len(indices) for cat, indices in indices_by_category.items()}
            total_count = sum(cat_counts.values())
            
            # Use square root for frequency balancing
            cat_weights = {cat: np.sqrt(count / total_count) for cat, count in cat_counts.items()}
            weight_sum = sum(cat_weights.values())
            cat_samples = {cat: max(100, int(sample_size * (weight / weight_sum))) 
                          for cat, weight in cat_weights.items()}
            
            # Adjust to ensure we get exactly sample_size samples
            total_samples = sum(cat_samples.values())
            if total_samples != sample_size:
                # Distribute the remainder among categories with most samples
                remainder = sample_size - total_samples
                sorted_cats = sorted(cat_samples.keys(), key=lambda k: cat_samples[k], reverse=True)
                for i in range(abs(remainder)):
                    cat = sorted_cats[i % len(sorted_cats)]
                    cat_samples[cat] += 1 if remainder > 0 else -1
            
            # Sample from each category
            sampled_indices = []
            for cat, count in cat_samples.items():
                if cat in indices_by_category and len(indices_by_category[cat]) > 0:
                    if count >= len(indices_by_category[cat]):
                        # Take all indices if we need more than available
                        sampled_indices.extend(indices_by_category[cat])
                    else:
                        # Random sample without replacement
                        sampled_indices.extend(
                            np.random.choice(indices_by_category[cat], 
                                            size=count, 
                                            replace=False)
                        )
            
            sample_indices = np.array(sampled_indices)
            sample_size = len(sample_indices)
        
        # Allocate arrays for features and labels
        # Features: DEM, slope, rainfall, deep learning model probabilities
        num_base_features = 3  # DEM, slope, rainfall
        
        # Only include deep learning models (exclude BATHTUB)
        dl_models = ['RES50', 'RES101', 'RES152', 'TINY', '100M', '300M', '600M']
        available_dl_models = [model for model in dl_models if model in probs]
        
        # Each DL model gives 5 probabilities (one per flood category)
        num_model_features = len(available_dl_models) * 5  
        num_features = num_base_features + num_model_features
        
        features = np.zeros((sample_size, num_features), dtype=np.float32)
        labels = np.zeros(sample_size, dtype=np.int64)
        
        # Flatten arrays to make sampling easier
        dem_flat = dem.flatten()[sample_indices]
        slope_flat = slope.flatten()[sample_indices]
        rainfall_flat = rainfall.flatten()[sample_indices]
        flood_cat_flat = flood_cat.flatten()[sample_indices]
        
        # Fill in base features
        features[:, 0] = dem_flat
        features[:, 1] = slope_flat
        features[:, 2] = rainfall_flat
        
        # Fill in model probabilities
        feature_idx = num_base_features
        for model_name in available_dl_models:
            # Skip if model is not available
            if model_name not in probs:
                continue
                
            # Add model probabilities
            model_probs = probs[model_name]
            if model_probs is not None:
                for c in range(5):  # 5 flood categories
                    if len(model_probs.shape) == 3:  # (classes, height, width)
                        prob_flat = model_probs[c].flatten()[sample_indices]
                    else:  # Handle older format
                        prob_flat = np.zeros_like(dem_flat)
                    features[:, feature_idx] = prob_flat
                    feature_idx += 1
        
        # Fill in labels
        labels = flood_cat_flat
        
        return features, labels
    
    def prepare_training_data(self):
        """Prepare training data for meta-learning from all training domains."""
        print("Preparing training data...")
        
        all_features = []
        all_labels = []
        
        # Maximum number of samples per domain-rainfall combination
        max_samples_per_combo = 50000
        
        # Process each training domain
        for domain in self.train_domains:
            print(f"Processing domain: {domain}")
            rainfall_levels = self._get_rainfall_levels(domain)
            
            for rainfall_level in rainfall_levels:
                print(f"  Processing rainfall level: {rainfall_level}")
                
                # Get model predictions
                result = self._get_model_predictions(domain, rainfall_level)
                
                # Check if we got valid data
                if result['dem'].size == 0:
                    print(f"  No valid data for {domain}, {rainfall_level}. Skipping.")
                    continue
                
                # Extract features and labels (using stratified sampling to limit memory)
                features, labels = self._extract_pixel_features(result, sample_size=max_samples_per_combo)
                
                if features.size > 0:
                    # Store features and labels
                    all_features.append(features)
                    all_labels.append(labels)
                
                # Clear the cache for this domain-rainfall combo to free memory
                cache_key = f"{domain}_{rainfall_level}"
                if cache_key in self.model_cache:
                    # Keep only jaccard scores and metadata, not the full arrays
                    jaccard_scores = self.model_cache[cache_key]['jaccard_scores']
                    dem_mean = self.model_cache[cache_key]['dem_mean']
                    dem_std = self.model_cache[cache_key]['dem_std']
                    
                    # Replace with minimal data
                    self.model_cache[cache_key] = {
                        'jaccard_scores': jaccard_scores,
                        'dem_mean': dem_mean,
                        'dem_std': dem_std,
                        'cleaned': True  # Flag to indicate this is cleaned data
                    }
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Combine all features and labels
        if all_features:
            print(f"Combining {len(all_features)} feature sets...")
            
            # To avoid memory issues, process in batches if needed
            max_batch_size = 1000000  # Maximum batch size
            
            if sum(f.shape[0] for f in all_features) > max_batch_size:
                print(f"Too many samples ({sum(f.shape[0] for f in all_features)}). Processing in batches.")
                
                # Initialize with first batch
                self.training_data['features'] = all_features[0]
                self.training_data['labels'] = all_labels[0]
                
                # Process remaining batches
                total_samples = all_features[0].shape[0]
                
                for i in range(1, len(all_features)):
                    # Add batch
                    self.training_data['features'] = np.vstack([self.training_data['features'], all_features[i]])
                    self.training_data['labels'] = np.concatenate([self.training_data['labels'], all_labels[i]])
                    total_samples += all_features[i].shape[0]
                    
                    # Save intermediate results if getting too large
                    if total_samples > max_batch_size:
                        print(f"  Saving intermediate batch with {total_samples} samples...")
                        
                        # Save training data
                        training_data_path = os.path.join(self.output_dir, f'training_data_batch_{len(all_features)}.npz')
                        np.savez_compressed(
                            training_data_path, 
                            features=self.training_data['features'],
                            labels=self.training_data['labels']
                        )
                        
                        # Keep only a random subsample for further processing
                        subsample_size = max_batch_size // 2
                        indices = np.random.choice(total_samples, size=subsample_size, replace=False)
                        self.training_data['features'] = self.training_data['features'][indices]
                        self.training_data['labels'] = self.training_data['labels'][indices]
                        total_samples = subsample_size
                        
                        # Force garbage collection
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            else:
                # If not too large, combine all at once
                self.training_data['features'] = np.vstack(all_features)
                self.training_data['labels'] = np.concatenate(all_labels)
            
            print(f"Prepared {len(self.training_data['features'])} samples for training")
            
            # Save training data
            training_data_path = os.path.join(self.output_dir, 'training_data.npz')
            np.savez_compressed(
                training_data_path, 
                features=self.training_data['features'],
                labels=self.training_data['labels']
            )
            print(f"Saved training data to {training_data_path}")
        else:
            print("No training data prepared!")
    
    def train_meta_model(self, model_type='decision_tree', max_depth=10):
        """
        Train the meta-learning model on the prepared data.
        
        Args:
            model_type: Type of meta-model ('decision_tree' or 'random_forest')
            max_depth: Maximum depth of the tree
            
        Returns:
            Trained meta-model
        """
        print("Training meta-model...")
        
        # Check if training data is available
        if len(self.training_data['features']) == 0:
            print("No training data available! Run prepare_training_data() first.")
            return None
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            self.training_data['features'], 
            self.training_data['labels'],
            test_size=0.2,
            random_state=42
        )
        
        print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
        
        # Create and train meta-model
        if model_type == 'random_forest':
            print(f"Training RandomForest (max_depth={max_depth})...")
            self.meta_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=max_depth,
                random_state=42,
                n_jobs=-1
            )
        else:  # Default to decision tree
            print(f"Training DecisionTree (max_depth={max_depth})...")
            self.meta_model = DecisionTreeClassifier(
                max_depth=max_depth,
                random_state=42
            )
        
        # Train the model
        start_time = time.time()
        self.meta_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate on validation set
        val_pred = self.meta_model.predict(X_val)
        
        # Calculate Jaccard scores and F1 scores for each class
        jaccard_scores = {}
        f1_scores = {}
        for i in range(5):  # 5 flood categories
            # Create binary masks for this class
            true_cls = (y_val == i).astype(int)
            pred_cls = (val_pred == i).astype(int)
            
            # Calculate Jaccard score and F1 score for this class
            jaccard_scores[i] = jaccard_score(true_cls, pred_cls, average='binary')
            f1_scores[i] = f1_score(true_cls, pred_cls, average='binary')
        
        # Calculate binary Jaccard score and F1 score (flood vs no-flood)
        true_binary = (y_val > 0).astype(int)
        pred_binary = (val_pred > 0).astype(int)
        binary_jaccard = jaccard_score(true_binary, pred_binary, average='binary')
        binary_f1 = f1_score(true_binary, pred_binary, average='binary')
        
        print("Validation F1 Scores:")
        print(f"  Binary (Flood vs No-Flood): {binary_f1:.4f}")
        print(f"  No Flood (Cat 0): {f1_scores[0]:.4f}")
        print(f"  Nuisance Flood (Cat 1): {f1_scores[1]:.4f}")
        print(f"  Minor Flood (Cat 2): {f1_scores[2]:.4f}")
        print(f"  Medium Flood (Cat 3): {f1_scores[3]:.4f}")
        print(f"  Major Flood (Cat 4): {f1_scores[4]:.4f}")
        
        print("\nValidation Jaccard Scores:")
        print(f"  Binary (Flood vs No-Flood): {binary_jaccard:.4f}")
        print(f"  No Flood (Cat 0): {jaccard_scores[0]:.4f}")
        print(f"  Nuisance Flood (Cat 1): {jaccard_scores[1]:.4f}")
        print(f"  Minor Flood (Cat 2): {jaccard_scores[2]:.4f}")
        print(f"  Medium Flood (Cat 3): {jaccard_scores[3]:.4f}")
        print(f"  Major Flood (Cat 4): {jaccard_scores[4]:.4f}")
        
        # Save meta-model
        model_path = os.path.join(self.output_dir, f'meta_model_{model_type}.joblib')
        joblib.dump(self.meta_model, model_path)
        print(f"Saved meta-model to {model_path}")
        
        # Save validation results
        val_results = {
            'binary_jaccard': binary_jaccard,
            'binary_f1': binary_f1,
            'class_jaccard': jaccard_scores,
            'class_f1': f1_scores,
            'training_time': training_time,
            'model_type': model_type,
            'max_depth': max_depth
        }
        
        with open(os.path.join(self.output_dir, 'validation_results.json'), 'w') as f:
            json.dump(val_results, f, indent=2)
        
        return self.meta_model
    
    def predict(self, domain, rainfall_level):
        """
        Make predictions using the meta-model for a domain and rainfall level.
        
        Args:
            domain: The domain to predict for
            rainfall_level: The rainfall level to predict for
            
        Returns:
            Meta-model predictions
        """
        if self.meta_model is None:
            print("Meta-model not trained! Run train_meta_model() first.")
            return None
        
        print(f"Making meta-model predictions for {domain}, {rainfall_level}")
        
        # Get base model predictions
        result = self._get_model_predictions(domain, rainfall_level)
        
        # Check if we have valid data
        if result['dem'].size == 0:
            print(f"No valid data for {domain}, {rainfall_level}. Returning empty predictions.")
            return {
                'dem': np.array([]),
                'slope': np.array([]),
                'rainfall': np.array([]),
                'flood_cat': np.array([]),
                'meta_prediction': np.array([]),
                'base_predictions': {}
            }
        
        # For very large datasets, use batch processing
        height, width = result['dem'].shape
        total_pixels = height * width
        max_batch_size = 5000000  # Maximum batch size for meta-model prediction
        
        if total_pixels > max_batch_size:
            print(f"Large dataset ({height}x{width}={total_pixels} pixels). Using batch processing.")
            
            # Initialize output array
            meta_predictions = np.zeros((height, width), dtype=np.int64)
            
            # Process in batches
            batch_size = max_batch_size
            num_batches = (total_pixels + batch_size - 1) // batch_size
            
            print(f"Processing in {num_batches} batches of size {batch_size}")
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_pixels)
                
                print(f"Processing batch {batch_idx+1}/{num_batches} (pixels {start_idx}-{end_idx})")
                
                # Extract features for this batch
                features_batch, _ = self._extract_pixel_features(
                    result, 
                    sample_size=-1  # Use all pixels
                )
                
                # Take the batch slice
                features_batch = features_batch[start_idx:end_idx]
                
                # Make predictions for this batch
                pred_batch = self.meta_model.predict(features_batch)
                
                # Map predictions back to the right places in the output array
                batch_indices = np.arange(start_idx, end_idx)
                rows = batch_indices // width
                cols = batch_indices % width
                meta_predictions[rows, cols] = pred_batch
                
                # Clean up memory
                del features_batch
                del pred_batch
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        else:
            # Extract features (all pixels)
            features, _ = self._extract_pixel_features(result, sample_size=-1)
            
            # Make predictions
            predictions = self.meta_model.predict(features)
            
            # Reshape predictions to match the original shape
            meta_predictions = predictions.reshape(height, width)
            
            # Clean up memory
            del features
            del predictions
        
        return {
            'dem': result['dem'],
            'slope': result['slope'],
            'rainfall': result['rainfall'],
            'flood_cat': result['flood_cat'],  # Ground truth
            'meta_prediction': meta_predictions,
            'base_predictions': result['predictions']
        }
    
    def evaluate(self, test_domains=None):
        """
        Evaluate the meta-model on test domains.
        
        Args:
            test_domains: List of domains to test on (default: all domains not in train_domains)
            
        Returns:
            Dictionary of evaluation results
        """
        if self.meta_model is None:
            print("Meta-model not trained! Run train_meta_model() first.")
            return None
        
        # Set test domains if not provided
        if test_domains is None:
            # Use domains not in train_domains
            all_domains = ["HOU001", "HOU007", "SF001", "SF002", "NYC002", "LA002", "DAL002", "AUS002", "MIA002"]
            test_domains = [d for d in all_domains if d not in self.train_domains]
        
        print(f"Evaluating meta-model on domains: {', '.join(test_domains)}")
        
        # Store results for each domain and rainfall level
        results = {}
        
        # Process each test domain
        for domain in test_domains:
            rainfall_levels = self._get_rainfall_levels(domain)
            
            if not rainfall_levels:
                print(f"No rainfall data available for domain {domain}. Skipping.")
                continue
                
            domain_results = {}
            
            for rainfall_level in rainfall_levels:
                print(f"Evaluating domain {domain}, rainfall {rainfall_level}")
                
                try:
                    # Make predictions
                    pred_result = self.predict(domain, rainfall_level)
                    
                    # Check if we got valid predictions
                    if pred_result is None or pred_result['dem'].size == 0:
                        print(f"No valid predictions for {domain}, {rainfall_level}. Skipping.")
                        continue
                    
                    # Calculate Jaccard scores for meta-model
                    meta_jaccard = calculate_jaccard_scores(
                        pred_result['flood_cat'], 
                        pred_result['meta_prediction']
                    )
                    
                    # Store base model Jaccard scores
                    base_jaccard = {}
                    for model_name, pred in pred_result['base_predictions'].items():
                        base_jaccard[model_name] = calculate_jaccard_scores(
                            pred_result['flood_cat'], 
                            pred
                        )
                    
                    # Store results
                    domain_results[rainfall_level] = {
                        'meta_jaccard': meta_jaccard,
                        'base_jaccard': base_jaccard
                    }
                    
                    # Create visualization
                    self._create_comparison_visualization(
                        domain, rainfall_level, pred_result,
                        meta_jaccard, base_jaccard
                    )
                    
                    # Save intermediate results to avoid losing progress
                    if len(domain_results) % 2 == 0:  # Save every 2 rainfall levels
                        intermediate_results = {domain: domain_results}
                        self._save_evaluation_results(intermediate_results, suffix=f"{domain}_{len(domain_results)}")
                    
                    # Clean up memory
                    del pred_result
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error evaluating {domain}, {rainfall_level}: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            results[domain] = domain_results
            
            # Save results after each domain
            self._save_evaluation_results({domain: domain_results}, suffix=domain)
        
        # Save complete results
        self._save_evaluation_results(results)
        
        print(f"Saved evaluation results to {self.output_dir}")
        
        return results
    
    def _save_evaluation_results(self, results, suffix=None):
        """Save evaluation results to a JSON file."""
        # Convert results to serializable format
        serializable_results = {}
        for domain, domain_data in results.items():
            serializable_results[domain] = {}
            for rainfall, rainfall_data in domain_data.items():
                serializable_results[domain][rainfall] = {
                    'meta_jaccard': {
                        k: float(v) for k, v in rainfall_data['meta_jaccard'].items()
                    },
                    'base_jaccard': {
                        model: {k: float(v) for k, v in scores.items()}
                        for model, scores in rainfall_data['base_jaccard'].items()
                    }
                }
        
        # Create filename
        filename = 'evaluation_results'
        if suffix:
            filename += f'_{suffix}'
        filename += '.json'
        
        results_path = os.path.join(self.output_dir, filename)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return results_path
    
    def _create_comparison_visualization(self, domain, rainfall_level, pred_result, meta_jaccard, base_jaccard):
        """Create visualization comparing meta-model with base models."""
        print(f"Creating visualization for {domain}, {rainfall_level}")
        
        try:
            # Create colormap for flood categories
            flood_colors = [FLOOD_COLORS[FloodCategory(i)] for i in range(len(FloodCategory))]
            cmap = ListedColormap(flood_colors)
            
            # Get data
            flood_cat = pred_result['flood_cat']
            meta_pred = pred_result['meta_prediction']
            base_preds = pred_result['base_predictions']
            
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
                
                # Find the best performing models based on binary Jaccard score
                model_scores = [(model, scores['binary']) for model, scores in base_jaccard.items()]
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
                binary_f1 = base_jaccard[model_name]['binary_f1']
                axes[idx].set_title(f"{model_name} - F1: {binary_f1:.3f}")
                axes[idx].imshow(pred, cmap=cmap, vmin=0, vmax=4)
            
            # Add overall title
            fig.suptitle(f"Model Comparison: {domain}, Rainfall: {rainfall_level}", fontsize=16)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
            
            # Save figure
            save_path = os.path.join(self.output_dir, f'comparison_{domain}_{rainfall_level}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close('all')  # Close all figures to free memory
            
            print(f"Saved visualization to {save_path}")
            
            # Clear memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            import traceback
            traceback.print_exc()

def main(args):
    """Main function to run the meta-learning process."""
    # Parse training domains
    train_domains = args.train_domains.split(',') if args.train_domains else None
    
    # Create meta-learning model
    meta_model = MetaLearningModel(train_domains=train_domains)
    
    # Load base models
    meta_model.load_base_models()
    
    # Prepare training data
    meta_model.prepare_training_data()
    
    # Train meta-model
    meta_model.train_meta_model(
        model_type=args.model_type,
        max_depth=args.max_depth
    )
    
    # Evaluate meta-model
    test_domains = args.test_domains.split(',') if args.test_domains else None
    meta_model.evaluate(test_domains=test_domains)
    
    print("Meta-learning process completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meta-learning for flood prediction')
    parser.add_argument('--train_domains', type=str, default="HOU001,SF001,NYC002", 
                        help='Comma-separated list of domains to use for training')
    parser.add_argument('--test_domains', type=str, default="HOU007,LA002,DAL002", 
                        help='Comma-separated list of domains to use for testing')
    parser.add_argument('--model_type', type=str, default='decision_tree', 
                        choices=['decision_tree', 'random_forest'],
                        help='Type of meta-model to train')
    parser.add_argument('--max_depth', type=int, default=10, 
                        help='Maximum depth of the decision tree or random forest')
    
    args = parser.parse_args()
    main(args) 
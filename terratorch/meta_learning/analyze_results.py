# coding: utf-8

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def load_results(results_dir):
    """Load evaluation results from a meta-learning run."""
    eval_path = os.path.join(results_dir, 'evaluation_results.json')
    val_path = os.path.join(results_dir, 'validation_results.json')
    
    results = {}
    
    if os.path.exists(eval_path):
        with open(eval_path, 'r') as f:
            results['evaluation'] = json.load(f)
    else:
        print(f"Warning: Evaluation results not found at {eval_path}")
        results['evaluation'] = {}
    
    if os.path.exists(val_path):
        with open(val_path, 'r') as f:
            results['validation'] = json.load(f)
    else:
        print(f"Warning: Validation results not found at {val_path}")
        results['validation'] = {}
    
    return results

def create_performance_dataframe(results):
    """Create a DataFrame of model performance across all domains and rainfall levels."""
    rows = []
    
    # Process each domain
    for domain, domain_data in results['evaluation'].items():
        # Process each rainfall level
        for rainfall, rainfall_data in domain_data.items():
            # Extract meta-model Jaccard scores
            meta_jaccard = rainfall_data['meta_jaccard']
            
            # Add meta-model row
            rows.append({
                'Domain': domain,
                'Rainfall': rainfall,
                'Model': 'Meta-Model',
                'Binary_Jaccard': meta_jaccard['binary'],
                'NoFlood_Jaccard': meta_jaccard['no_flood'],
                'Nuisance_Jaccard': meta_jaccard['nuisance'],
                'Minor_Jaccard': meta_jaccard['minor'],
                'Medium_Jaccard': meta_jaccard['medium'],
                'Major_Jaccard': meta_jaccard['major']
            })
            
            # Add rows for each base model
            for model_name, model_jaccard in rainfall_data['base_jaccard'].items():
                rows.append({
                    'Domain': domain,
                    'Rainfall': rainfall,
                    'Model': model_name,
                    'Binary_Jaccard': model_jaccard['binary'],
                    'NoFlood_Jaccard': model_jaccard['no_flood'],
                    'Nuisance_Jaccard': model_jaccard['nuisance'],
                    'Minor_Jaccard': model_jaccard['minor'],
                    'Medium_Jaccard': model_jaccard['medium'],
                    'Major_Jaccard': model_jaccard['major']
                })
    
    return pd.DataFrame(rows)

def plot_binary_performance(performance_df, output_dir):
    """Plot binary classification performance (flood vs no-flood) for all models."""
    plt.figure(figsize=(12, 8))
    
    # Group by model and calculate mean Jaccard score
    model_avg = performance_df.groupby('Model')['Binary_Jaccard'].mean().reset_index()
    model_avg = model_avg.sort_values('Binary_Jaccard', ascending=False)
    
    # Create bar plot
    sns.barplot(x='Model', y='Binary_Jaccard', data=model_avg)
    
    plt.title('Average Binary Jaccard Score (Flood vs No-Flood)')
    plt.ylabel('Jaccard Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Highlight meta-model
    for i, model in enumerate(model_avg['Model']):
        if model == 'Meta-Model':
            plt.gca().get_children()[i].set_facecolor('red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'binary_performance.png'), dpi=300)
    plt.close()

def plot_category_performance(performance_df, output_dir):
    """Plot performance for each flood category."""
    # Define category columns
    categories = ['NoFlood_Jaccard', 'Nuisance_Jaccard', 'Minor_Jaccard', 'Medium_Jaccard', 'Major_Jaccard']
    category_names = ['No Flood', 'Nuisance', 'Minor', 'Medium', 'Major']
    
    # Reshape data for plotting
    melted_df = pd.melt(
        performance_df, 
        id_vars=['Domain', 'Rainfall', 'Model'],
        value_vars=categories,
        var_name='Category', 
        value_name='Jaccard'
    )
    
    # Map category names
    category_map = {cat: name for cat, name in zip(categories, category_names)}
    melted_df['Category'] = melted_df['Category'].map(lambda x: category_map.get(x, x))
    
    # Plot
    plt.figure(figsize=(14, 10))
    
    # Group by model and category, then calculate mean
    model_cat_avg = melted_df.groupby(['Model', 'Category'])['Jaccard'].mean().reset_index()
    
    # Create grouped bar plot
    sns.barplot(x='Model', y='Jaccard', hue='Category', data=model_cat_avg)
    
    plt.title('Average Jaccard Score by Flood Category')
    plt.ylabel('Jaccard Score')
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Flood Category')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'category_performance.png'), dpi=300)
    plt.close()

def plot_domain_comparison(performance_df, output_dir):
    """Compare performance across domains."""
    plt.figure(figsize=(14, 8))
    
    # Group by domain and model, calculate mean binary Jaccard
    domain_model_avg = performance_df.groupby(['Domain', 'Model'])['Binary_Jaccard'].mean().reset_index()
    
    # Create grouped bar plot
    sns.barplot(x='Domain', y='Binary_Jaccard', hue='Model', data=domain_model_avg)
    
    plt.title('Binary Jaccard Score by Domain')
    plt.ylabel('Jaccard Score')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'domain_comparison.png'), dpi=300)
    plt.close()

def plot_rainfall_trends(performance_df, output_dir):
    """Plot performance trends across rainfall levels."""
    # Extract numeric rainfall values
    performance_df['Rainfall_Value'] = performance_df['Rainfall'].str.extract('(\d+)').astype(int)
    
    # Group by rainfall value and model, calculate mean binary Jaccard
    rainfall_model_avg = performance_df.groupby(['Rainfall_Value', 'Model'])['Binary_Jaccard'].mean().reset_index()
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Separate out the meta-model for highlighting
    meta_df = rainfall_model_avg[rainfall_model_avg['Model'] == 'Meta-Model']
    base_df = rainfall_model_avg[rainfall_model_avg['Model'] != 'Meta-Model']
    
    # Plot base models
    for model in base_df['Model'].unique():
        model_data = base_df[base_df['Model'] == model]
        plt.plot(model_data['Rainfall_Value'], model_data['Binary_Jaccard'], 
                 marker='o', label=model, alpha=0.7)
    
    # Plot meta-model with emphasis
    plt.plot(meta_df['Rainfall_Value'], meta_df['Binary_Jaccard'], 
             marker='s', linewidth=3, color='red', label='Meta-Model')
    
    plt.title('Binary Jaccard Score Trend by Rainfall Level')
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Jaccard Score')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rainfall_trends.png'), dpi=300)
    plt.close()

def create_summary_table(performance_df, output_dir):
    """Create a summary table of model performance."""
    # Group by model and calculate statistics
    summary = performance_df.groupby('Model').agg({
        'Binary_Jaccard': ['mean', 'std', 'min', 'max'],
        'NoFlood_Jaccard': 'mean',
        'Nuisance_Jaccard': 'mean',
        'Minor_Jaccard': 'mean',
        'Medium_Jaccard': 'mean',
        'Major_Jaccard': 'mean'
    }).reset_index()
    
    # Flatten multi-index columns
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    
    # Rename columns for readability
    summary = summary.rename(columns={
        'Binary_Jaccard_mean': 'Binary_Mean',
        'Binary_Jaccard_std': 'Binary_StdDev',
        'Binary_Jaccard_min': 'Binary_Min',
        'Binary_Jaccard_max': 'Binary_Max',
        'NoFlood_Jaccard_mean': 'NoFlood',
        'Nuisance_Jaccard_mean': 'Nuisance',
        'Minor_Jaccard_mean': 'Minor',
        'Medium_Jaccard_mean': 'Medium',
        'Major_Jaccard_mean': 'Major'
    })
    
    # Sort by binary mean descending
    summary = summary.sort_values('Binary_Mean', ascending=False)
    
    # Save to CSV
    summary.to_csv(os.path.join(output_dir, 'model_summary.csv'), index=False)
    
    # Create a formatted markdown table for easy viewing
    with open(os.path.join(output_dir, 'model_summary.md'), 'w') as f:
        f.write("# Model Performance Summary\n\n")
        f.write("## Binary Classification (Flood vs No-Flood)\n\n")
        
        # Binary performance table
        f.write("| Model | Mean | StdDev | Min | Max |\n")
        f.write("|-------|------|--------|-----|-----|\n")
        for _, row in summary.iterrows():
            f.write(f"| {row['Model']} | {row['Binary_Mean']:.4f} | {row['Binary_StdDev']:.4f} | {row['Binary_Min']:.4f} | {row['Binary_Max']:.4f} |\n")
        
        f.write("\n## Category Performance (Mean Jaccard Score)\n\n")
        
        # Category performance table
        f.write("| Model | No Flood | Nuisance | Minor | Medium | Major |\n")
        f.write("|-------|----------|----------|-------|--------|-------|\n")
        for _, row in summary.iterrows():
            f.write(f"| {row['Model']} | {row['NoFlood']:.4f} | {row['Nuisance']:.4f} | {row['Minor']:.4f} | {row['Medium']:.4f} | {row['Major']:.4f} |\n")
    
    return summary

def main(args):
    """Main function to analyze meta-learning results."""
    # Load results
    results = load_results(args.results_dir)
    
    # Create output directory for analysis
    output_dir = os.path.join(args.results_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create performance DataFrame
    performance_df = create_performance_dataframe(results)
    
    # Save DataFrame for further analysis
    performance_df.to_csv(os.path.join(output_dir, 'performance_data.csv'), index=False)
    
    # Create visualizations
    plot_binary_performance(performance_df, output_dir)
    plot_category_performance(performance_df, output_dir)
    plot_domain_comparison(performance_df, output_dir)
    plot_rainfall_trends(performance_df, output_dir)
    
    # Create summary table
    summary = create_summary_table(performance_df, output_dir)
    
    print(f"Analysis completed. Results saved to {output_dir}")
    
    # Print a summary of results
    meta_model_row = summary[summary['Model'] == 'Meta-Model']
    if not meta_model_row.empty:
        meta_binary = meta_model_row['Binary_Mean'].values[0]
        print(f"\nMeta-Model Average Binary Jaccard: {meta_binary:.4f}")
    
    # Find the best performing base model
    base_models = summary[summary['Model'] != 'Meta-Model']
    if not base_models.empty:
        best_base_model = base_models.iloc[0]['Model']
        best_base_score = base_models.iloc[0]['Binary_Mean']
        print(f"Best Base Model: {best_base_model} (Binary Jaccard: {best_base_score:.4f})")
    
    # Print improvement percentage if meta-model is better
    if not meta_model_row.empty and not base_models.empty:
        improvement = (meta_binary - best_base_score) / best_base_score * 100
        if improvement > 0:
            print(f"Meta-Model improvement: +{improvement:.2f}%")
        else:
            print(f"Meta-Model improvement: {improvement:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze meta-learning results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing meta-learning results')
    
    args = parser.parse_args()
    main(args) 
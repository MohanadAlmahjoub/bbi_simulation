#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter Sweep Module for Brain-to-Brain Interface Simulation

This module provides utilities for running parameter sweeps and sensitivity analyses
on the brain-to-brain interface simulation.
"""

import numpy as np
import pandas as pd
import os
import time
import itertools
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from copy import deepcopy

from simulation.bbi_simulation import BBISimulation
from config.config_loader import ConfigLoader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ParameterSweep')


class ParameterSweep:
    """
    A class for running parameter sweeps and sensitivity analyses.
    
    The ParameterSweep class runs multiple simulations with different parameter
    combinations, collects metrics, and generates summary plots.
    
    Attributes:
        base_config (Dict): Base configuration dictionary
        sweep_config (Dict): Parameter sweep configuration
        output_dir (str): Directory for output files
        output_csv (str): Path to output CSV file
        params (Dict): Parameters to sweep
        repetitions (int): Number of repetitions per parameter combination
        results (pd.DataFrame): Results of parameter sweep
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the ParameterSweep.
        
        Args:
            config (Dict, optional): Configuration dictionary
            config_path (str, optional): Path to configuration file
            output_dir (str, optional): Directory for output files
        """
        # Load configuration
        if config is None:
            config_loader = ConfigLoader(config_path)
            self.base_config = config_loader.get_config()
        else:
            self.base_config = deepcopy(config)
        
        # Extract parameter sweep configuration
        self.sweep_config = self.base_config.get('analysis', {}).get('parameter_sweep', {})
        
        # Set output directory
        if output_dir is None:
            self.output_dir = 'results'
        else:
            self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set output CSV file
        self.output_csv = self.sweep_config.get('output_csv', os.path.join(self.output_dir, 'parameter_sweep.csv'))
        
        # Ensure output directory exists for CSV
        os.makedirs(os.path.dirname(os.path.abspath(self.output_csv)), exist_ok=True)
        
        # Extract parameters to sweep
        self.params = self.sweep_config.get('params', {})
        
        # Extract number of repetitions
        self.repetitions = self.sweep_config.get('repetitions', 1)
        
        # Initialize results DataFrame
        self.results = None
    
    def _update_config(self, base_config: Dict[str, Any], param_name: str, param_value: Any) -> Dict[str, Any]:
        """
        Update configuration with parameter value.
        
        Args:
            base_config (Dict): Base configuration dictionary
            param_name (str): Parameter name (e.g., 'tau', 'window_size')
            param_value (Any): Parameter value
            
        Returns:
            Dict: Updated configuration dictionary
        """
        # Create a deep copy of the base configuration
        config = deepcopy(base_config)
        
        # Update configuration based on parameter name
        if param_name == 'tau':
            config['stimulator']['tau'] = param_value
        elif param_name == 'window_size':
            config['decoder']['window_size'] = param_value
        elif param_name == 'noise_std':
            config['decoder']['noise_std'] = param_value
        elif param_name == 'threshold':
            config['stimulator']['threshold'] = param_value
        elif param_name == 'latency_jitter':
            config['simulation']['latency_jitter'] = param_value
        elif param_name == 'freq':
            config['emotion_signal']['freq'] = param_value
        elif param_name == 'dims':
            config['emotion_signal']['dims'] = param_value
        elif param_name == 'channels':
            config['emotion_signal']['channels'] = param_value
        elif param_name == 'model_type':
            config['decoder']['ml']['model_type'] = param_value
        else:
            logger.warning(f"Unknown parameter: {param_name}")
        
        return config
    
    def run(self, progress_callback: Optional[Callable[[float], None]] = None) -> pd.DataFrame:
        """
        Run parameter sweep.
        
        Args:
            progress_callback (Callable, optional): Callback function for progress updates
            
        Returns:
            pd.DataFrame: Results of parameter sweep
        """
        # Check if parameter sweep is enabled
        if not self.sweep_config.get('enabled', False):
            logger.warning("Parameter sweep is not enabled in configuration")
            return pd.DataFrame()
        
        # Check if there are parameters to sweep
        if not self.params:
            logger.warning("No parameters specified for sweep")
            return pd.DataFrame()
        
        # Generate parameter combinations
        param_names = list(self.params.keys())
        param_values = list(self.params.values())
        param_combinations = list(itertools.product(*param_values))
        
        # Calculate total number of simulations
        total_simulations = len(param_combinations) * self.repetitions
        logger.info(f"Running parameter sweep with {len(param_combinations)} parameter combinations "
                   f"and {self.repetitions} repetitions ({total_simulations} simulations total)")
        
        # Initialize results list
        results_list = []
        
        # Run simulations for each parameter combination
        for i, param_combination in enumerate(param_combinations):
            # Create parameter dictionary
            param_dict = {name: value for name, value in zip(param_names, param_combination)}
            
            # Log parameter combination
            param_str = ", ".join([f"{name}={value}" for name, value in param_dict.items()])
            logger.info(f"Running simulations for parameters: {param_str} "
                       f"({i+1}/{len(param_combinations)} combinations)")
            
            # Update configuration with parameter combination
            config = deepcopy(self.base_config)
            for name, value in param_dict.items():
                config = self._update_config(config, name, value)
            
            # Run repetitions
            for rep in range(self.repetitions):
                # Create simulation log directory
                sim_log_dir = os.path.join(
                    self.output_dir,
                    f"sweep_{i+1}_{rep+1}"
                )
                os.makedirs(sim_log_dir, exist_ok=True)
                
                # Create and run simulation
                sim = BBISimulation(config=config, log_dir=sim_log_dir)
                
                # Run simulation
                logger.info(f"Running simulation {rep+1}/{self.repetitions}")
                results = sim.run()
                
                # Extract metrics
                metrics = results['metrics']
                
                # Create result dictionary
                result_dict = {
                    'combination_id': i + 1,
                    'repetition': rep + 1,
                    **param_dict  # Include parameter values
                }
                
                # Add metrics
                if 'mean_decode_correlation' in metrics:
                    result_dict['decode_correlation'] = metrics['mean_decode_correlation']
                if 'mean_decode_rmse' in metrics:
                    result_dict['decode_rmse'] = metrics['mean_decode_rmse']
                
                if 'mean_correlation' in metrics:
                    result_dict['response_correlation'] = metrics['mean_correlation']
                    result_dict['response_rmse'] = metrics['mean_rmse']
                    result_dict['response_delay'] = metrics['mean_delay']
                else:
                    result_dict['response_correlation'] = metrics['correlation']
                    result_dict['response_rmse'] = metrics['rmse']
                    result_dict['response_delay'] = metrics['delay']
                
                # Add latency metrics
                result_dict['processing_latency'] = np.mean(results['processing_latencies'][sim.decoder.window_size:]) * 1000  # ms
                result_dict['jitter_magnitude'] = np.mean(np.abs(results['jitter_latencies'][sim.decoder.window_size:])) * 1000  # ms
                result_dict['total_latency'] = np.mean(results['total_latencies'][sim.decoder.window_size:]) * 1000  # ms
                
                # Add simulation time
                result_dict['simulation_time'] = results['simulation_time']
                
                # Add to results list
                results_list.append(result_dict)
                
                # Update progress
                if progress_callback is not None:
                    progress = ((i * self.repetitions) + rep + 1) / total_simulations
                    progress_callback(progress)
        
        # Create DataFrame from results list
        self.results = pd.DataFrame(results_list)
        
        # Save results to CSV
        self.results.to_csv(self.output_csv, index=False)
        logger.info(f"Parameter sweep results saved to {self.output_csv}")
        
        return self.results
    
    def generate_plots(self, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate summary plots from parameter sweep results.
        
        Args:
            output_dir (str, optional): Directory for output plots
            
        Returns:
            Dict: Dictionary mapping plot names to file paths
        """
        if self.results is None or len(self.results) == 0:
            logger.warning("No results available for plotting")
            return {}
        
        # Set output directory
        if output_dir is None:
            output_dir = self.output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Dictionary to store plot file paths
        plot_files = {}
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 1. Accuracy vs. Latency plot
        if 'decode_correlation' in self.results.columns and 'total_latency' in self.results.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Group by parameter combinations and calculate mean
            param_names = [col for col in self.results.columns if col in self.params.keys()]
            if param_names:
                grouped = self.results.groupby(param_names).agg({
                    'decode_correlation': 'mean',
                    'total_latency': 'mean',
                    'response_correlation': 'mean'
                }).reset_index()
                
                # Create scatter plot with size based on response correlation
                scatter = ax.scatter(
                    grouped['total_latency'],
                    grouped['decode_correlation'],
                    s=grouped['response_correlation'] * 100,
                    alpha=0.7,
                    c=grouped['response_correlation'],
                    cmap='viridis'
                )
                
                # Add colorbar
                cbar = plt.colorbar(scatter)
                cbar.set_label('Response Correlation')
                
                # Add parameter labels
                for i, row in grouped.iterrows():
                    label = ", ".join([f"{name}={row[name]}" for name in param_names])
                    ax.annotate(
                        label,
                        (row['total_latency'], row['decode_correlation']),
                        fontsize=8,
                        alpha=0.7,
                        ha='center',
                        va='bottom',
                        xytext=(0, 5),
                        textcoords='offset points'
                    )
            else:
                # Simple scatter plot if no parameter groups
                ax.scatter(
                    self.results['total_latency'],
                    self.results['decode_correlation'],
                    alpha=0.7
                )
            
            ax.set_title('Decoding Accuracy vs. Latency Trade-off')
            ax.set_xlabel('Total Latency (ms)')
            ax.set_ylabel('Decoding Correlation')
            ax.grid(True)
            
            # Save plot
            accuracy_latency_path = os.path.join(output_dir, 'accuracy_vs_latency.png')
            plt.tight_layout()
            plt.savefig(accuracy_latency_path, dpi=300)
            plt.close(fig)
            
            plot_files['accuracy_vs_latency'] = accuracy_latency_path
            logger.info(f"Accuracy vs. Latency plot saved to {accuracy_latency_path}")
        
        # 2. Parameter effect plots
        for param_name in self.params.keys():
            if param_name in self.results.columns:
                # Create figure with subplots
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                axes = axes.flatten()
                
                # Plot decode correlation vs. parameter
                sns.boxplot(x=param_name, y='decode_correlation', data=self.results, ax=axes[0])
                axes[0].set_title(f'Decode Correlation vs. {param_name}')
                axes[0].set_xlabel(param_name)
                axes[0].set_ylabel('Decode Correlation')
                
                # Plot response correlation vs. parameter
                sns.boxplot(x=param_name, y='response_correlation', data=self.results, ax=axes[1])
                axes[1].set_title(f'Response Correlation vs. {param_name}')
                axes[1].set_xlabel(param_name)
                axes[1].set_ylabel('Response Correlation')
                
                # Plot total latency vs. parameter
                sns.boxplot(x=param_name, y='total_latency', data=self.results, ax=axes[2])
                axes[2].set_title(f'Total Latency vs. {param_name}')
                axes[2].set_xlabel(param_name)
                axes[2].set_ylabel('Total Latency (ms)')
                
                # Plot response delay vs. parameter
                sns.boxplot(x=param_name, y='response_delay', data=self.results, ax=axes[3])
                axes[3].set_title(f'Response Delay vs. {param_name}')
                axes[3].set_xlabel(param_name)
                axes[3].set_ylabel('Response Delay (s)')
                
                # Save plot
                param_plot_path = os.path.join(output_dir, f'effect_of_{param_name}.png')
                plt.tight_layout()
                plt.savefig(param_plot_path, dpi=300)
                plt.close(fig)
                
                plot_files[f'effect_of_{param_name}'] = param_plot_path
                logger.info(f"Effect of {param_name} plot saved to {param_plot_path}")
        
        # 3. Correlation heatmap
        if len(self.results) > 5:  # Only create heatmap if we have enough data points
            # Select numeric columns for correlation
            numeric_cols = self.results.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove repetition and combination_id columns
            if 'repetition' in numeric_cols:
                numeric_cols.remove('repetition')
            if 'combination_id' in numeric_cols:
                numeric_cols.remove('combination_id')
            
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Calculate correlation matrix
                corr_matrix = self.results[numeric_cols].corr()
                
                # Create heatmap
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
                ax.set_title('Correlation Matrix of Parameters and Metrics')
                
                # Save plot
                heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
                plt.tight_layout()
                plt.savefig(heatmap_path, dpi=300)
                plt.close(fig)
                
                plot_files['correlation_heatmap'] = heatmap_path
                logger.info(f"Correlation heatmap saved to {heatmap_path}")
        
        # 4. Summary table for manuscript
        if len(self.results) > 0:
            # Group by parameter combinations
            param_names = [col for col in self.results.columns if col in self.params.keys()]
            
            if param_names:
                # Calculate mean and std for key metrics
                summary = self.results.groupby(param_names).agg({
                    'decode_correlation': ['mean', 'std'],
                    'response_correlation': ['mean', 'std'],
                    'total_latency': ['mean', 'std'],
                    'response_delay': ['mean', 'std']
                }).reset_index()
                
                # Save summary table
                summary_path = os.path.join(output_dir, 'summary_table.csv')
                summary.to_csv(summary_path, index=False)
                
                plot_files['summary_table'] = summary_path
                logger.info(f"Summary table saved to {summary_path}")
                
                # Create a formatted table for the manuscript
                fig, ax = plt.subplots(figsize=(12, len(summary) * 0.5 + 1))
                ax.axis('tight')
                ax.axis('off')
                
                # Prepare table data
                table_data = []
                header = []
                
                # Add parameter columns
                for param in param_names:
                    header.append(param)
                
                # Add metric columns
                header.extend([
                    'Decode Corr',
                    'Response Corr',
                    'Latency (ms)',
                    'Delay (ms)'
                ])
                
                # Add rows
                for _, row in summary.iterrows():
                    table_row = []
                    
                    # Add parameter values
                    for param in param_names:
                        table_row.append(row[param])
                    
                    # Add metric values with standard deviation
                    table_row.append(f"{row[('decode_correlation', 'mean')]:.3f} ± {row[('decode_correlation', 'std')]:.3f}")
                    table_row.append(f"{row[('response_correlation', 'mean')]:.3f} ± {row[('response_correlation', 'std')]:.3f}")
                    table_row.append(f"{row[('total_latency', 'mean')]:.1f} ± {row[('total_latency', 'std')]:.1f}")
                    table_row.append(f"{row[('response_delay', 'mean')] * 1000:.1f} ± {row[('response_delay', 'std')] * 1000:.1f}")
                    
                    table_data.append(table_row)
                
                # Create table
                table = ax.table(
                    cellText=table_data,
                    colLabels=header,
                    loc='center',
                    cellLoc='center'
                )
                
                # Style table
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                
                # Save table as image
                table_img_path = os.path.join(output_dir, 'summary_table.png')
                plt.tight_layout()
                plt.savefig(table_img_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                
                plot_files['summary_table_img'] = table_img_path
                logger.info(f"Summary table image saved to {table_img_path}")
        
        return plot_files


# Example usage
if __name__ == "__main__":
    # Create parameter sweep
    sweep = ParameterSweep(output_dir='results/parameter_sweep')
    
    # Run parameter sweep
    results = sweep.run()
    
    # Generate plots
    plot_files = sweep.generate_plots()
    
    # Print results summary
    if len(results) > 0:
        print("\nParameter Sweep Results Summary:")
        print(f"Total simulations: {len(results)}")
        
        # Group by parameter combinations
        param_names = [col for col in results.columns if col in sweep.params.keys()]
        if param_names:
            grouped = results.groupby(param_names).agg({
                'decode_correlation': 'mean',
                'response_correlation': 'mean',
                'total_latency': 'mean'
            }).reset_index()
            
            print("\nResults by parameter combination:")
            for i, row in grouped.iterrows():
                param_str = ", ".join([f"{name}={row[name]}" for name in param_names])
                print(f"\nParameters: {param_str}")
                print(f"  Decode Correlation: {row['decode_correlation']:.4f}")
                print(f"  Response Correlation: {row['response_correlation']:.4f}")
                print(f"  Total Latency: {row['total_latency']:.2f} ms")
        
        # Print best parameter combination for different metrics
        print("\nBest parameter combinations:")
        
        # Best decode correlation
        best_decode = results.loc[results['decode_correlation'].idxmax()]
        param_str = ", ".join([f"{name}={best_decode[name]}" for name in param_names])
        print(f"Best Decode Correlation ({best_decode['decode_correlation']:.4f}): {param_str}")
        
        # Best response correlation
        best_response = results.loc[results['response_correlation'].idxmax()]
        param_str = ", ".join([f"{name}={best_response[name]}" for name in param_names])
        print(f"Best Response Correlation ({best_response['response_correlation']:.4f}): {param_str}")
        
        # Lowest latency
        best_latency = results.loc[results['total_latency'].idxmin()]
        param_str = ", ".join([f"{name}={best_latency[name]}" for name in param_names])
        print(f"Lowest Total Latency ({best_latency['total_latency']:.2f} ms): {param_str}")
        
        # Print plot files
        print("\nGenerated plots:")
        for name, path in plot_files.items():
            print(f"  {name}: {path}")
    else:
        print("No parameter sweep results available.")

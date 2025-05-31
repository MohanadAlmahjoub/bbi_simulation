#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter sweep example script for Brain-to-Brain Interface simulation.

This script demonstrates how to run a parameter sweep to analyze the
sensitivity of the simulation to different parameter values.
"""

import os
import matplotlib.pyplot as plt
from bbi_simulation.config.config_loader import ConfigLoader
from bbi_simulation.analysis.parameter_sweep import ParameterSweep


def main():
    """Run a parameter sweep example."""
    # Create output directory
    output_dir = 'sweep_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load default configuration
    config_loader = ConfigLoader()
    config = config_loader.get_config()
    
    # Modify configuration for parameter sweep
    # Enable parameter sweep
    if 'analysis' not in config:
        config['analysis'] = {}
    if 'parameter_sweep' not in config['analysis']:
        config['analysis']['parameter_sweep'] = {}
    
    # Configure parameter sweep
    config['analysis']['parameter_sweep']['enabled'] = True
    config['analysis']['parameter_sweep']['repetitions'] = 2
    config['analysis']['parameter_sweep']['params'] = {
        'tau': [0.05, 0.1, 0.2],
        'window_size': [25, 50]
    }
    
    # Modify simulation for faster execution
    config['simulation']['duration'] = 2.0  # 2 seconds
    
    # Create and run parameter sweep
    sweep = ParameterSweep(config=config, output_dir=output_dir)
    results = sweep.run()
    
    # Generate plots
    plot_files = sweep.generate_plots()
    
    # Print results summary
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
    
    # Print plot files
    print("\nGenerated plots:")
    for name, path in plot_files.items():
        print(f"  {name}: {path}")
    
    # Display accuracy vs latency plot
    if 'accuracy_vs_latency' in plot_files:
        plt.figure(figsize=(10, 8))
        img = plt.imread(plot_files['accuracy_vs_latency'])
        plt.imshow(img)
        plt.axis('off')
        plt.title("Accuracy vs. Latency Trade-off")
        plt.show()
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

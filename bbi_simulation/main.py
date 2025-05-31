#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main module for Brain-to-Brain Interface Simulation.

This module provides command-line interfaces for running simulations and parameter sweeps.
"""

import os
import argparse
import logging
import yaml
from typing import Dict, Any, Optional

from simulation.bbi_simulation import BBISimulation
from analysis.parameter_sweep import ParameterSweep
from config.config_loader import ConfigLoader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Brain-to-Brain Interface Simulation'
    )
    
    # General arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='results',
        help='Output directory'
    )
    
    # Simulation mode
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--sweep', '-s',
        action='store_true',
        help='Run parameter sweep'
    )
    group.add_argument(
        '--real-time', '-r',
        action='store_true',
        help='Run simulation in real-time'
    )
    
    # Visualization options
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable plotting'
    )
    
    return parser.parse_args()


def main():
    """Run a single simulation."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.get_config()
    
    # Save configuration to output directory
    config_path = os.path.join(args.output, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create and run simulation
    sim = BBISimulation(config=config, log_dir=args.output)
    results = sim.run(real_time=args.real_time)
    
    # Print metrics
    metrics = results['metrics']
    print("\nSimulation Metrics:")
    print(f"Mean decode correlation: {metrics['mean_decode_correlation']:.4f}")
    print(f"Mean decode RMSE: {metrics['mean_decode_rmse']:.4f}")
    
    if 'mean_correlation' in metrics:
        print(f"Mean response correlation: {metrics['mean_correlation']:.4f}")
        print(f"Mean response RMSE: {metrics['mean_rmse']:.4f}")
        print(f"Mean response delay: {metrics['mean_delay']*1000:.2f} ms")
    else:
        print(f"Response correlation: {metrics['correlation']:.4f}")
        print(f"Response RMSE: {metrics['rmse']:.4f}")
        print(f"Response delay: {metrics['delay']*1000:.2f} ms")
    
    # Plot results if not disabled
    if not args.no_plot:
        import matplotlib.pyplot as plt
        
        # Extract results
        t = results['t']
        emotion_signals = results['emotion_signals']
        channel_signals = results['channel_signals']
        decoded = results['decoded']
        response = results['response']
        processing_latencies = results['processing_latencies']
        jitter_latencies = results['jitter_latencies']
        total_latencies = results['total_latencies']
        
        # Plot results
        plt.figure(figsize=(12, 15))
        
        # Plot emotion signals
        plt.subplot(5, 1, 1)
        for d in range(emotion_signals.shape[1]):
            plt.plot(t, emotion_signals[:, d], label=f"Dim {d+1}")
        plt.title("Raw Emotion Signals")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        
        # Plot channel signals (first 4 channels)
        plt.subplot(5, 1, 2)
        for c in range(min(4, channel_signals.shape[1])):
            plt.plot(t, channel_signals[:, c], label=f"Ch {c+1}")
        plt.title("EEG Channel Signals (First 4 Channels)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        
        # Plot decoded signals
        plt.subplot(5, 1, 3)
        for d in range(decoded.shape[1]):
            plt.plot(t, decoded[:, d], label=f"Dim {d+1}")
        plt.title("Decoded Signals")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        
        # Plot response signals
        plt.subplot(5, 1, 4)
        for d in range(response.shape[1]):
            plt.plot(t, response[:, d], label=f"Dim {d+1}")
        plt.title("Stimulation Responses")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        
        # Plot latencies
        plt.subplot(5, 1, 5)
        plt.plot(t, processing_latencies * 1000, label="Processing")
        plt.plot(t, jitter_latencies * 1000, label="Jitter")
        plt.plot(t, total_latencies * 1000, label="Total")
        plt.title("Latencies")
        plt.xlabel("Time (s)")
        plt.ylabel("Latency (ms)")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(args.output, 'simulation_results.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        
        logger.info(f"Results figure saved to {fig_path}")


def run_sweep():
    """Run a parameter sweep."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.get_config()
    
    # Save configuration to output directory
    config_path = os.path.join(args.output, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create and run parameter sweep
    sweep = ParameterSweep(config=config, output_dir=args.output)
    results = sweep.run()
    
    # Generate plots if not disabled
    if not args.no_plot:
        plot_files = sweep.generate_plots()
        
        # Print plot files
        print("\nGenerated plots:")
        for name, path in plot_files.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Run appropriate mode
    if args.sweep:
        run_sweep()
    else:
        main()

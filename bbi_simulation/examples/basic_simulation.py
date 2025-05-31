#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic example script for running a Brain-to-Brain Interface simulation.

This script demonstrates how to create and run a simple simulation using
the default configuration.
"""

import os
import matplotlib.pyplot as plt
from bbi_simulation.config.config_loader import ConfigLoader
from bbi_simulation.simulation.bbi_simulation import BBISimulation


def main():
    """Run a basic simulation example."""
    # Create output directory
    output_dir = 'example_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load default configuration
    config_loader = ConfigLoader()
    config = config_loader.get_config()
    
    # Modify configuration for a shorter simulation
    config['simulation']['duration'] = 5.0  # 5 seconds
    
    # Create and run simulation
    sim = BBISimulation(config=config, log_dir=output_dir)
    results = sim.run()
    
    # Extract results
    t = results['t']
    emotion_signals = results['emotion_signals']
    channel_signals = results['channel_signals']
    decoded = results['decoded']
    response = results['response']
    metrics = results['metrics']
    
    # Print metrics
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
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot emotion signals
    plt.subplot(3, 1, 1)
    for d in range(emotion_signals.shape[1]):
        plt.plot(t, emotion_signals[:, d], label=f"Dim {d+1}")
    plt.title("Raw Emotion Signals")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Plot decoded signals
    plt.subplot(3, 1, 2)
    for d in range(decoded.shape[1]):
        plt.plot(t, decoded[:, d], label=f"Dim {d+1}")
    plt.title("Decoded Signals")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Plot response signals
    plt.subplot(3, 1, 3)
    for d in range(response.shape[1]):
        plt.plot(t, response[:, d], label=f"Dim {d+1}")
    plt.title("Stimulation Responses")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = os.path.join(output_dir, 'basic_simulation_results.png')
    plt.savefig(fig_path, dpi=300)
    plt.show()
    
    print(f"\nResults saved to {output_dir}")
    print(f"Figure saved to {fig_path}")


if __name__ == "__main__":
    main()

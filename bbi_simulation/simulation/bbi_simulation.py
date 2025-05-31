#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BBISimulation Controller Module for Brain-to-Brain Interface Simulation

This module provides the BBISimulation class that integrates the EmotionSignal,
Decoder, and Stimulator modules to run a complete closed-loop brain-to-brain
interface simulation.
"""

import numpy as np
import time
import os
import csv
import logging
from typing import Dict, List, Optional, Tuple, Union, Any

from bbi_simulation.core.emotion_signal import EmotionSignal
from bbi_simulation.core.decoder import Decoder
from bbi_simulation.core.stimulator import Stimulator
from bbi_simulation.config.config_loader import ConfigLoader


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('BBISimulation')


class BBISimulation:
    """
    A class for running closed-loop brain-to-brain interface simulations.
    
    The BBISimulation class integrates the EmotionSignal, Decoder, and Stimulator
    modules to create a complete closed-loop simulation. It handles time series
    generation, sliding window decoding, and stimulation response calculation.
    
    Attributes:
        config (Dict): Configuration dictionary
        duration (float): Total simulation time in seconds
        dt (float): Simulation time step in seconds
        signal (EmotionSignal): EmotionSignal instance for generating emotion signals
        decoder (Decoder): Decoder instance for decoding EEG signals
        stimulator (Stimulator): Stimulator instance for simulating neurostimulation
        latency_jitter (float): Maximum latency/jitter in seconds (±latency_jitter)
        log_dir (str): Directory for logging output
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
        log_dir: Optional[str] = None
    ):
        """
        Initialize the BBISimulation controller.
        
        Args:
            config (Dict, optional): Configuration dictionary
            config_path (str, optional): Path to configuration file
            log_dir (str, optional): Directory for logging output
        """
        # Load configuration
        if config is None:
            config_loader = ConfigLoader(config_path)
            self.config = config_loader.get_config()
        else:
            self.config = config
        
        # Extract simulation parameters
        sim_config = self.config['simulation']
        self.duration = sim_config['duration']
        self.dt = sim_config['dt']
        self.latency_jitter = sim_config.get('latency_jitter', 0.0)
        
        # Set random seed if specified
        if 'random_seed' in sim_config:
            np.random.seed(sim_config['random_seed'])
        
        # Initialize modules
        self.signal = EmotionSignal(self.config['emotion_signal'])
        self.decoder = Decoder(self.config['decoder'])
        self.stimulator = Stimulator(self.config['stimulator'])
        
        # Set up logging directory
        self.log_dir = log_dir
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
    
    def run(self, real_time: bool = False) -> Dict[str, Any]:
        """
        Run the closed-loop brain-to-brain interface simulation.
        
        Args:
            real_time (bool): Whether to run in real-time or as fast as possible
            
        Returns:
            Dict: Simulation results including time series, metrics, and latencies
        """
        logger.info("Starting simulation...")
        
        # Create time array
        t = np.arange(0, self.duration, self.dt)
        n_samples = len(t)
        
        # Generate raw emotion signals and channel signals
        logger.info("Generating emotion signals...")
        start_time = time.time()
        emotion_signals, channel_signals = self.signal.generate(t)
        generation_time = time.time() - start_time
        logger.info(f"Signal generation completed in {generation_time:.4f} seconds")
        
        # Get dimensions and channels
        dims = emotion_signals.shape[1]
        channels = channel_signals.shape[1]
        
        # Initialize arrays for decoded signals, responses, and latencies
        decoded = np.zeros((n_samples, dims))
        response = np.zeros((n_samples, dims))
        
        # Track latencies
        processing_latencies = np.zeros(n_samples)  # Processing time for each step
        jitter_latencies = np.zeros(n_samples)      # Applied jitter
        total_latencies = np.zeros(n_samples)       # Total latency
        
        # Get window size from decoder
        window_size = self.decoder.window_size
        
        # Run the simulation
        logger.info("Running closed-loop simulation...")
        sim_start_time = time.time()
        
        for i in range(window_size, n_samples):
            step_start_time = time.time()
            
            # Extract window for decoding
            window = channel_signals[i-window_size:i, :]
            
            # Apply latency/jitter if enabled
            jitter = 0.0
            if self.latency_jitter > 0:
                # Random latency between -latency_jitter and +latency_jitter
                jitter = np.random.uniform(-self.latency_jitter, self.latency_jitter)
                
                # Convert latency to samples
                jitter_samples = int(jitter / self.dt)
                
                # Adjust window based on latency
                if jitter_samples > 0:
                    # Positive latency: use older data
                    if i - window_size - jitter_samples >= 0:
                        window = channel_signals[i-window_size-jitter_samples:i-jitter_samples, :]
                elif jitter_samples < 0:
                    # Negative latency: use newer data (prediction)
                    if i - jitter_samples < n_samples:
                        window = channel_signals[i-window_size-jitter_samples:i-jitter_samples, :]
                
                # Store the applied jitter
                jitter_latencies[i] = jitter
            
            # Decode the window
            decode_start_time = time.time()
            decoded[i, :] = self.decoder.decode(window)
            decode_time = time.time() - decode_start_time
            
            # Apply stimulation to the current decoded value
            stim_start_time = time.time()
            # For real-time simulation, we only have access to current and past values
            response[i, :] = self.stimulator.stimulate(decoded[:i+1, :])[-1, :]
            stim_time = time.time() - stim_start_time
            
            # Calculate processing latency
            processing_latency = decode_time + stim_time
            processing_latencies[i] = processing_latency
            
            # Calculate total latency
            total_latencies[i] = processing_latency + jitter
            
            # If running in real-time, wait until the next time step
            if real_time:
                elapsed = time.time() - step_start_time
                if elapsed < self.dt:
                    time.sleep(self.dt - elapsed)
        
        # Apply stimulation to the full decoded series for final result
        # This ensures the response includes the full convolution effect
        response = self.stimulator.stimulate(decoded)
        
        # Calculate simulation time
        sim_time = time.time() - sim_start_time
        logger.info(f"Simulation completed in {sim_time:.4f} seconds")
        
        # Compute metrics
        logger.info("Computing performance metrics...")
        metrics = self.compute_metrics(emotion_signals, decoded, response)
        
        # Log latency statistics
        mean_processing_latency = np.mean(processing_latencies[window_size:])
        mean_jitter = np.mean(np.abs(jitter_latencies[window_size:]))
        mean_total_latency = np.mean(total_latencies[window_size:])
        
        logger.info(f"Mean processing latency: {mean_processing_latency*1000:.2f} ms")
        logger.info(f"Mean jitter magnitude: {mean_jitter*1000:.2f} ms")
        logger.info(f"Mean total latency: {mean_total_latency*1000:.2f} ms")
        
        # Save results if log_dir is specified
        if self.log_dir is not None:
            self.save_results(
                t, emotion_signals, channel_signals, decoded, response,
                processing_latencies, jitter_latencies, total_latencies,
                metrics
            )
        
        # Return results
        return {
            't': t,
            'emotion_signals': emotion_signals,
            'channel_signals': channel_signals,
            'decoded': decoded,
            'response': response,
            'processing_latencies': processing_latencies,
            'jitter_latencies': jitter_latencies,
            'total_latencies': total_latencies,
            'metrics': metrics,
            'simulation_time': sim_time,
            'generation_time': generation_time
        }
    
    def compute_metrics(
        self, 
        emotion_signals: np.ndarray, 
        decoded: np.ndarray, 
        response: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute performance metrics for the simulation.
        
        Args:
            emotion_signals (np.ndarray): Raw emotion signals
            decoded (np.ndarray): Decoded emotion signals
            response (np.ndarray): Stimulation responses
            
        Returns:
            Dict: Dictionary containing performance metrics
        """
        # Use stimulator's compute_response_metrics for response evaluation
        response_metrics = self.stimulator.compute_response_metrics(emotion_signals, response)
        
        # Compute additional metrics between raw and decoded
        dims = emotion_signals.shape[1]
        decode_correlations = np.zeros(dims)
        decode_rmses = np.zeros(dims)
        
        for d in range(dims):
            # Compute correlation coefficient
            corr = np.corrcoef(emotion_signals[:, d], decoded[:, d])[0, 1]
            decode_correlations[d] = corr
            
            # Compute RMSE
            rmse = np.sqrt(np.mean((emotion_signals[:, d] - decoded[:, d])**2))
            decode_rmses[d] = rmse
        
        # Combine metrics
        metrics = {
            'decode_correlations': decode_correlations,
            'mean_decode_correlation': np.mean(decode_correlations),
            'decode_rmses': decode_rmses,
            'mean_decode_rmse': np.mean(decode_rmses),
            **response_metrics  # Include response metrics
        }
        
        # Add simulation parameters to metrics
        metrics['signal_params'] = self.config['emotion_signal']
        metrics['decoder_params'] = self.config['decoder']
        metrics['stimulator_params'] = self.config['stimulator']
        metrics['latency_jitter'] = self.latency_jitter
        
        return metrics
    
    def save_results(
        self,
        t: np.ndarray,
        emotion_signals: np.ndarray,
        channel_signals: np.ndarray,
        decoded: np.ndarray,
        response: np.ndarray,
        processing_latencies: np.ndarray,
        jitter_latencies: np.ndarray,
        total_latencies: np.ndarray,
        metrics: Dict[str, Any]
    ) -> None:
        """
        Save simulation results to files.
        
        Args:
            t (np.ndarray): Time points array
            emotion_signals (np.ndarray): Raw emotion signals
            channel_signals (np.ndarray): EEG channel signals
            decoded (np.ndarray): Decoded emotion signals
            response (np.ndarray): Stimulation responses
            processing_latencies (np.ndarray): Processing latencies
            jitter_latencies (np.ndarray): Applied jitter latencies
            total_latencies (np.ndarray): Total latencies
            metrics (Dict): Performance metrics
        """
        # Save time series data
        time_series_path = os.path.join(self.log_dir, 'time_series.npz')
        np.savez(
            time_series_path,
            t=t,
            emotion_signals=emotion_signals,
            channel_signals=channel_signals,
            decoded=decoded,
            response=response,
            processing_latencies=processing_latencies,
            jitter_latencies=jitter_latencies,
            total_latencies=total_latencies
        )
        logger.info(f"Time series data saved to {time_series_path}")
        
        # Save metrics to CSV
        metrics_path = os.path.join(self.log_dir, 'metrics.csv')
        
        # Flatten metrics dictionary for CSV
        flat_metrics = {}
        
        # Add scalar metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float, str, bool)):
                flat_metrics[key] = value
            elif isinstance(value, np.ndarray):
                for i, val in enumerate(value):
                    flat_metrics[f"{key}_{i+1}"] = val
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (int, float, str, bool)):
                        flat_metrics[f"{key}_{subkey}"] = subvalue
        
        # Write to CSV
        with open(metrics_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=flat_metrics.keys())
            writer.writeheader()
            writer.writerow(flat_metrics)
        
        logger.info(f"Metrics saved to {metrics_path}")


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create simulation
    sim = BBISimulation(log_dir='results')
    
    # Run simulation
    results = sim.run()
    
    # Extract results
    t = results['t']
    emotion_signals = results['emotion_signals']
    channel_signals = results['channel_signals']
    decoded = results['decoded']
    response = results['response']
    processing_latencies = results['processing_latencies']
    jitter_latencies = results['jitter_latencies']
    total_latencies = results['total_latencies']
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
    plt.show()

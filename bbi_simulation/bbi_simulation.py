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
from typing import Optional, Union, Tuple, List, Dict

from bbi_simulation.emotion_signal import EmotionSignal
from bbi_simulation.decoder import Decoder
from bbi_simulation.stimulator import Stimulator


class BBISimulation:
    """
    A class for running closed-loop brain-to-brain interface simulations.
    
    The BBISimulation class integrates the EmotionSignal, Decoder, and Stimulator
    modules to create a complete closed-loop simulation. It handles time series
    generation, sliding window decoding, and stimulation response calculation.
    
    Attributes:
        duration (float): Total simulation time in seconds
        dt (float): Simulation time step in seconds
        signal (EmotionSignal): EmotionSignal instance for generating emotion signals
        decoder (Decoder): Decoder instance for decoding EEG signals
        stimulator (Stimulator): Stimulator instance for simulating neurostimulation
        latency_jitter (float): Maximum latency/jitter in seconds (±latency_jitter)
    """
    
    def __init__(
        self,
        duration: float = 10.0,
        dt: float = 0.01,
        signal_params: Optional[Dict] = None,
        decoder_params: Optional[Dict] = None,
        stimulator_params: Optional[Dict] = None,
        latency_jitter: float = 0.0
    ):
        """
        Initialize the BBISimulation controller.
        
        Args:
            duration (float): Total simulation time in seconds
            dt (float): Simulation time step in seconds
            signal_params (Dict, optional): Parameters for EmotionSignal initialization
            decoder_params (Dict, optional): Parameters for Decoder initialization
            stimulator_params (Dict, optional): Parameters for Stimulator initialization
            latency_jitter (float): Maximum latency/jitter in seconds (±latency_jitter)
        """
        self.duration = duration
        self.dt = dt
        self.latency_jitter = latency_jitter
        
        # Set default parameters if not provided
        if signal_params is None:
            signal_params = {"kind": "sine", "freq": 0.2, "dims": 2}
        
        if decoder_params is None:
            decoder_params = {"noise_std": 0.1, "window_size": 50}
        
        if stimulator_params is None:
            stimulator_params = {"tau": 0.1, "dt": dt}
        
        # Initialize modules
        self.signal = EmotionSignal(**signal_params)
        self.decoder = Decoder(**decoder_params)
        self.stimulator = Stimulator(**stimulator_params)
        
        # Store parameters for reference
        self.signal_params = signal_params
        self.decoder_params = decoder_params
        self.stimulator_params = stimulator_params
    
    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run the closed-loop brain-to-brain interface simulation.
        
        Returns:
            Tuple containing:
                t (np.ndarray): Time points array
                raw (np.ndarray): Raw emotion signals
                decoded (np.ndarray): Decoded emotion signals
                response (np.ndarray): Stimulation responses
        """
        # Create time array
        t = np.arange(0, self.duration, self.dt)
        n_samples = len(t)
        
        # Generate raw emotion signals
        raw = self.signal.generate(t)
        dims = raw.shape[1]
        
        # Initialize arrays for decoded signals and responses
        decoded = np.zeros((n_samples, dims))
        response = np.zeros((n_samples, dims))
        
        # Get window size from decoder
        window_size = self.decoder.window_size
        
        # Track latency for each time step
        latencies = np.zeros(n_samples)
        
        # Run the simulation
        for i in range(window_size, n_samples):
            # Extract window for each dimension
            window = raw[i-window_size:i, :]
            
            # Apply latency/jitter if enabled
            if self.latency_jitter > 0:
                # Random latency between -latency_jitter and +latency_jitter
                latency = np.random.uniform(-self.latency_jitter, self.latency_jitter)
                
                # Convert latency to samples
                latency_samples = int(latency / self.dt)
                
                # Adjust window based on latency
                if latency_samples > 0:
                    # Positive latency: use older data
                    if i - window_size - latency_samples >= 0:
                        window = raw[i-window_size-latency_samples:i-latency_samples, :]
                elif latency_samples < 0:
                    # Negative latency: use newer data (prediction)
                    if i - latency_samples < n_samples:
                        window = raw[i-window_size-latency_samples:i-latency_samples, :]
                
                # Store the applied latency
                latencies[i] = latency
            
            # Decode the window
            decoded[i, :] = self.decoder.decode(window)
        
        # Apply stimulation to the full decoded series
        response = self.stimulator.stimulate(decoded)
        
        return t, raw, decoded, response, latencies
    
    def compute_metrics(
        self, 
        raw: np.ndarray, 
        decoded: np.ndarray, 
        response: np.ndarray
    ) -> Dict:
        """
        Compute performance metrics for the simulation.
        
        Args:
            raw (np.ndarray): Raw emotion signals
            decoded (np.ndarray): Decoded emotion signals
            response (np.ndarray): Stimulation responses
            
        Returns:
            Dict: Dictionary containing performance metrics
        """
        metrics = {}
        dims = raw.shape[1]
        
        # Cross-correlation coefficients between raw and response
        cross_corr = np.zeros(dims)
        for d in range(dims):
            # Compute correlation coefficient
            corr = np.corrcoef(raw[:, d], response[:, d])[0, 1]
            cross_corr[d] = corr
        
        metrics['cross_correlation'] = cross_corr
        metrics['mean_cross_correlation'] = np.mean(cross_corr)
        
        # RMSE between raw and decoded
        rmse = np.zeros(dims)
        for d in range(dims):
            # Compute RMSE
            rmse[d] = np.sqrt(np.mean((raw[:, d] - decoded[:, d])**2))
        
        metrics['rmse'] = rmse
        metrics['mean_rmse'] = np.mean(rmse)
        
        # Add simulation parameters to metrics
        metrics['signal_params'] = self.signal_params
        metrics['decoder_params'] = self.decoder_params
        metrics['stimulator_params'] = self.stimulator_params
        metrics['latency_jitter'] = self.latency_jitter
        
        return metrics


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create simulation
    sim = BBISimulation(
        duration=5.0,
        dt=0.01,
        signal_params={"kind": "sine", "freq": 0.5, "dims": 2},
        decoder_params={"noise_std": 0.2, "window_size": 50},
        stimulator_params={"tau": 0.1, "dt": 0.01},
        latency_jitter=0.01  # ±10 ms jitter
    )
    
    # Run simulation
    t, raw, decoded, response, latencies = sim.run()
    
    # Compute metrics
    metrics = sim.compute_metrics(raw, decoded, response)
    
    # Print metrics
    print("Simulation Metrics:")
    print(f"Cross-correlation: {metrics['cross_correlation']}")
    print(f"Mean cross-correlation: {metrics['mean_cross_correlation']:.4f}")
    print(f"RMSE: {metrics['rmse']}")
    print(f"Mean RMSE: {metrics['mean_rmse']:.4f}")
    
    # Plot results
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Raw signals
    axes[0].plot(t, raw[:, 0], label="Dim 1")
    axes[0].plot(t, raw[:, 1], label="Dim 2")
    axes[0].set_title("Raw Emotion Signals")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True)
    
    # Decoded signals
    axes[1].plot(t, decoded[:, 0], label="Dim 1")
    axes[1].plot(t, decoded[:, 1], label="Dim 2")
    axes[1].set_title("Decoded Signals")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True)
    
    # Response signals
    axes[2].plot(t, response[:, 0], label="Dim 1")
    axes[2].plot(t, response[:, 1], label="Dim 2")
    axes[2].set_title("Stimulation Responses")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()
    axes[2].grid(True)
    
    # Latencies
    axes[3].plot(t, latencies * 1000)  # Convert to ms for better visualization
    axes[3].set_title("Applied Latency/Jitter")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Latency (ms)")
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stimulator Module for Brain-to-Brain Interface Simulation

This module provides the Stimulator class for simulating neurostimulation
responses to decoded emotion signals using an exponential decay kernel.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any


class Stimulator:
    """
    A class for simulating neurostimulation responses to decoded emotion signals.
    
    The Stimulator class creates an exponential decay kernel and convolves it with
    decoded emotion signals to simulate the brain's response to neurostimulation.
    
    Attributes:
        tau (float): Decay time constant in seconds
        dt (float): Simulation time step in seconds
        threshold (float): Activation threshold for stimulation
        kernel (np.ndarray): Exponential decay kernel
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Stimulator.
        
        Args:
            config (Dict): Configuration dictionary containing parameters for stimulation
        """
        # Extract parameters from config
        self.tau = config.get('tau', 0.1)
        self.dt = config.get('dt', 0.01)  # Get from config or use default
        self.threshold = config.get('threshold', 0.0)
        
        # Build the exponential kernel
        self._build_kernel()
    
    def _build_kernel(self) -> None:
        """
        Build the exponential decay kernel.
        
        The kernel is defined as k[i] = exp(-i*dt/tau) for i >= 0.
        The kernel length is set to 5*tau to capture most of the decay.
        """
        # Calculate kernel length (5*tau should capture most of the decay)
        kernel_length = int(5 * self.tau / self.dt)
        
        # Ensure kernel length is at least 1
        kernel_length = max(1, kernel_length)
        
        # Create time points for the kernel
        t_kernel = np.arange(kernel_length) * self.dt
        
        # Calculate exponential decay
        self.kernel = np.exp(-t_kernel / self.tau)
        
        # Normalize the kernel to preserve signal amplitude
        self.kernel /= np.sum(self.kernel)
    
    def stimulate(self, decoded_series: np.ndarray) -> np.ndarray:
        """
        Simulate neurostimulation response by convolving decoded signals with the kernel.
        
        Args:
            decoded_series (np.ndarray): Decoded emotion signals of shape (n_samples,) or (n_samples, dims)
            
        Returns:
            np.ndarray: Stimulation response of same shape as decoded_series
        """
        # Check if input is 1D or 2D
        if decoded_series.ndim == 1:
            # 1D case: single dimension
            response = np.convolve(decoded_series, self.kernel, mode='same')
            
            # Apply threshold
            if self.threshold != 0.0:
                response = np.where(response > self.threshold, response, 0)
                
            return response
        else:
            # 2D case: multiple dimensions
            n_samples, dims = decoded_series.shape
            response = np.zeros_like(decoded_series)
            
            # Convolve each dimension separately
            for d in range(dims):
                response[:, d] = np.convolve(decoded_series[:, d], self.kernel, mode='same')
                
                # Apply threshold
                if self.threshold != 0.0:
                    response[:, d] = np.where(response[:, d] > self.threshold, response[:, d], 0)
            
            return response
    
    def update_parameters(self, params: Dict[str, Any]) -> None:
        """
        Update stimulator parameters and rebuild the kernel.
        
        Args:
            params (Dict): Dictionary of parameters to update
                Supported keys: 'tau', 'dt', 'threshold'
        """
        # Update parameters
        if 'tau' in params:
            self.tau = params['tau']
        
        if 'dt' in params:
            self.dt = params['dt']
        
        if 'threshold' in params:
            self.threshold = params['threshold']
        
        # Rebuild the kernel with new parameters
        self._build_kernel()
    
    def compute_response_metrics(self, original: np.ndarray, response: np.ndarray) -> Dict[str, Any]:
        """
        Compute metrics for the stimulation response.
        
        Args:
            original (np.ndarray): Original emotion signals
            response (np.ndarray): Stimulation response signals
            
        Returns:
            Dict: Dictionary of response metrics
        """
        metrics = {}
        
        # Check if inputs are 1D or 2D
        if original.ndim == 1:
            # 1D case: single dimension
            # Compute correlation coefficient
            corr = np.corrcoef(original, response)[0, 1]
            metrics['correlation'] = corr
            
            # Compute RMSE
            rmse = np.sqrt(np.mean((original - response)**2))
            metrics['rmse'] = rmse
            
            # Compute delay (time shift for maximum correlation)
            max_delay = int(2 * self.tau / self.dt)  # Maximum delay to check
            delays = np.arange(-max_delay, max_delay + 1)
            corrs = np.zeros_like(delays, dtype=float)
            
            for i, delay in enumerate(delays):
                if delay < 0:
                    corrs[i] = np.corrcoef(original[:delay], response[-delay:])[0, 1]
                elif delay > 0:
                    corrs[i] = np.corrcoef(original[delay:], response[:-delay])[0, 1]
                else:
                    corrs[i] = corr
            
            best_delay_idx = np.argmax(corrs)
            best_delay = delays[best_delay_idx] * self.dt
            metrics['delay'] = best_delay
            metrics['max_correlation'] = corrs[best_delay_idx]
        else:
            # 2D case: multiple dimensions
            n_dims = original.shape[1]
            
            # Initialize arrays for metrics
            correlations = np.zeros(n_dims)
            rmses = np.zeros(n_dims)
            delays = np.zeros(n_dims)
            max_correlations = np.zeros(n_dims)
            
            # Compute metrics for each dimension
            for d in range(n_dims):
                # Compute correlation coefficient
                corr = np.corrcoef(original[:, d], response[:, d])[0, 1]
                correlations[d] = corr
                
                # Compute RMSE
                rmse = np.sqrt(np.mean((original[:, d] - response[:, d])**2))
                rmses[d] = rmse
                
                # Compute delay (time shift for maximum correlation)
                max_delay = int(2 * self.tau / self.dt)  # Maximum delay to check
                delays_d = np.arange(-max_delay, max_delay + 1)
                corrs = np.zeros_like(delays_d, dtype=float)
                
                for i, delay in enumerate(delays_d):
                    if delay < 0:
                        corrs[i] = np.corrcoef(original[:delay, d], response[-delay:, d])[0, 1]
                    elif delay > 0:
                        corrs[i] = np.corrcoef(original[delay:, d], response[:-delay, d])[0, 1]
                    else:
                        corrs[i] = corr
                
                best_delay_idx = np.argmax(corrs)
                delays[d] = delays_d[best_delay_idx] * self.dt
                max_correlations[d] = corrs[best_delay_idx]
            
            # Store metrics
            metrics['correlations'] = correlations
            metrics['mean_correlation'] = np.mean(correlations)
            metrics['rmses'] = rmses
            metrics['mean_rmse'] = np.mean(rmses)
            metrics['delays'] = delays
            metrics['mean_delay'] = np.mean(delays)
            metrics['max_correlations'] = max_correlations
            metrics['mean_max_correlation'] = np.mean(max_correlations)
        
        return metrics


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from config.config_loader import ConfigLoader
    from core.emotion_signal import EmotionSignal
    from core.decoder import Decoder
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.get_config()
    
    # Create signal generator
    signal_gen = EmotionSignal(config['emotion_signal'])
    
    # Create time array
    dt = config['simulation']['dt']
    duration = 5.0  # 5 seconds
    t = np.arange(0, duration, dt)
    
    # Generate signals
    emotion_signals, channel_signals = signal_gen.generate(t)
    
    # Create decoder
    decoder = Decoder(config['decoder'])
    
    # Decode signals
    n_samples = len(t)
    window_size = decoder.window_size
    decoded = np.zeros((n_samples, signal_gen.dims))
    
    for i in range(window_size, n_samples):
        # Extract window
        window = channel_signals[i-window_size:i, :]
        
        # Decode
        decoded[i, :] = decoder.decode(window)
    
    # Create stimulator with different tau values
    tau_values = [0.05, 0.1, 0.2, 0.5]
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    # Original and decoded signals
    plt.subplot(len(tau_values) + 1, 1, 1)
    for d in range(signal_gen.dims):
        plt.plot(t, emotion_signals[:, d], label=f"Original Dim {d+1}")
        plt.plot(t, decoded[:, d], '--', label=f"Decoded Dim {d+1}")
    plt.title("Original and Decoded Signals")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Stimulation responses for different tau values
    for i, tau in enumerate(tau_values):
        # Create stimulator
        stim_config = config['stimulator'].copy()
        stim_config['tau'] = tau
        stimulator = Stimulator(stim_config)
        
        # Stimulate
        response = stimulator.stimulate(decoded)
        
        # Compute metrics
        metrics = stimulator.compute_response_metrics(emotion_signals, response)
        
        # Plot
        plt.subplot(len(tau_values) + 1, 1, i + 2)
        for d in range(signal_gen.dims):
            plt.plot(t, response[:, d], label=f"Dim {d+1}")
        
        # Add metrics to title
        if 'mean_correlation' in metrics:
            title = f"Stimulation Response (tau = {tau} s, corr = {metrics['mean_correlation']:.2f}, delay = {metrics['mean_delay']*1000:.1f} ms)"
        else:
            title = f"Stimulation Response (tau = {tau} s, corr = {metrics['correlation']:.2f}, delay = {metrics['delay']*1000:.1f} ms)"
        
        plt.title(title)
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.legend()
    
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

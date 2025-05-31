#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stimulator Module for Brain-to-Brain Interface Simulation

This module provides the Stimulator class for simulating neurostimulation
responses to decoded emotion signals using an exponential decay kernel.
"""

import numpy as np
from typing import Optional, Union, Tuple, List


class Stimulator:
    """
    A class for simulating neurostimulation responses to decoded emotion signals.
    
    The Stimulator class creates an exponential decay kernel and convolves it with
    decoded emotion signals to simulate the brain's response to neurostimulation.
    
    Attributes:
        tau (float): Decay time constant in seconds
        dt (float): Simulation time step in seconds
        kernel (np.ndarray): Exponential decay kernel
    """
    
    def __init__(self, tau: float = 0.1, dt: float = 0.01):
        """
        Initialize the Stimulator.
        
        Args:
            tau (float): Decay time constant in seconds
            dt (float): Simulation time step in seconds
        """
        self.tau = tau
        self.dt = dt
        
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
            return np.convolve(decoded_series, self.kernel, mode='same')
        else:
            # 2D case: multiple dimensions
            n_samples, dims = decoded_series.shape
            response = np.zeros_like(decoded_series)
            
            # Convolve each dimension separately
            for d in range(dims):
                response[:, d] = np.convolve(decoded_series[:, d], self.kernel, mode='same')
            
            return response
    
    def update_parameters(self, tau: Optional[float] = None, dt: Optional[float] = None) -> None:
        """
        Update stimulator parameters and rebuild the kernel.
        
        Args:
            tau (float, optional): New decay time constant in seconds
            dt (float, optional): New simulation time step in seconds
        """
        if tau is not None:
            self.tau = tau
        
        if dt is not None:
            self.dt = dt
        
        # Rebuild the kernel with new parameters
        self._build_kernel()


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from emotion_signal import EmotionSignal
    from decoder import Decoder
    
    # Create time array
    dt = 0.01  # 10 ms time step
    duration = 5.0  # 5 seconds
    t = np.arange(0, duration, dt)
    
    # Create signal generator
    signal_gen = EmotionSignal(kind="sine", freq=0.5, dims=1)
    
    # Generate signals
    signals = signal_gen.generate(t)
    
    # Create decoder
    window_size = 50  # 0.5 seconds at dt=0.01
    decoder = Decoder(noise_std=0.2, window_size=window_size)
    
    # Decode signals
    n_samples = len(t)
    decoded = np.zeros((n_samples, 1))
    
    for i in range(window_size, n_samples):
        # Extract window
        window = signals[i-window_size:i, :]
        
        # Decode
        decoded[i, :] = decoder.decode(window)
    
    # Create stimulator with different tau values
    tau_values = [0.05, 0.1, 0.2, 0.5]
    
    # Plot
    fig, axes = plt.subplots(len(tau_values) + 1, 1, figsize=(10, 10), sharex=True)
    
    # Original and decoded signals
    axes[0].plot(t, signals[:, 0], label="Original")
    axes[0].plot(t, decoded[:, 0], label="Decoded")
    axes[0].set_title("Original and Decoded Signals")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True)
    
    # Stimulation responses for different tau values
    for i, tau in enumerate(tau_values):
        stimulator = Stimulator(tau=tau, dt=dt)
        response = stimulator.stimulate(decoded)
        
        axes[i+1].plot(t, response[:, 0])
        axes[i+1].set_title(f"Stimulation Response (tau = {tau} s)")
        axes[i+1].set_ylabel("Amplitude")
        axes[i+1].grid(True)
    
    axes[-1].set_xlabel("Time (s)")
    
    plt.tight_layout()
    plt.show()

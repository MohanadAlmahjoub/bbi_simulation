#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionSignal Module for Brain-to-Brain Interface Simulation

This module provides the EmotionSignal class for generating synthetic emotion signals
with different waveform types (square, sine, or synthetic data).
"""

import numpy as np
from typing import Callable, Optional, Union, List, Tuple


class EmotionSignal:
    """
    A class for generating synthetic emotion signals with different waveforms.
    
    The EmotionSignal class can generate square waves, sine waves, or more complex
    synthetic data that mimics realistic EEG patterns. It supports multiple emotion
    dimensions (e.g., valence and arousal).
    
    Attributes:
        kind (str): Type of waveform ("square", "sine", or "data")
        freq (float): Base frequency in Hz
        dims (int): Number of emotion channels/dimensions
        data_loader (Callable, optional): Function to load external data
    """
    
    def __init__(
        self, 
        kind: str = "sine", 
        freq: float = 0.2, 
        dims: int = 2,
        data_loader: Optional[Callable] = None
    ):
        """
        Initialize the EmotionSignal generator.
        
        Args:
            kind (str): Type of waveform ("square", "sine", or "data")
            freq (float): Base frequency in Hz
            dims (int): Number of emotion channels/dimensions
            data_loader (Callable, optional): Function to load external data
                                             when kind="data" and external data is needed
        
        Raises:
            ValueError: If kind is not one of "square", "sine", or "data"
        """
        if kind not in ["square", "sine", "data"]:
            raise ValueError("kind must be one of 'square', 'sine', or 'data'")
        
        self.kind = kind
        self.freq = freq
        self.dims = dims
        self.data_loader = data_loader
        
        # Phase shifts for different dimensions to create variation
        self.phase_shifts = np.linspace(0, 2*np.pi, dims, endpoint=False)
        
        # Frequency variations for different dimensions
        self.freq_factors = np.linspace(0.8, 1.2, dims)
    
    def _generate_square(self, t_array: np.ndarray) -> np.ndarray:
        """
        Generate square wave signals for each dimension.
        
        Args:
            t_array (np.ndarray): Time points array
            
        Returns:
            np.ndarray: Array of shape (len(t_array), dims) containing square wave values
        """
        result = np.zeros((len(t_array), self.dims))
        
        for d in range(self.dims):
            # Create square wave with phase shift and frequency variation
            effective_freq = self.freq * self.freq_factors[d]
            phase = self.phase_shifts[d]
            result[:, d] = np.sign(np.sin(2 * np.pi * effective_freq * t_array + phase))
            
        return result
    
    def _generate_sine(self, t_array: np.ndarray) -> np.ndarray:
        """
        Generate sine wave signals for each dimension.
        
        Args:
            t_array (np.ndarray): Time points array
            
        Returns:
            np.ndarray: Array of shape (len(t_array), dims) containing sine wave values
        """
        result = np.zeros((len(t_array), self.dims))
        
        for d in range(self.dims):
            # Create sine wave with phase shift and frequency variation
            effective_freq = self.freq * self.freq_factors[d]
            phase = self.phase_shifts[d]
            result[:, d] = np.sin(2 * np.pi * effective_freq * t_array + phase)
            
        return result
    
    def _generate_synthetic_data(self, t_array: np.ndarray) -> np.ndarray:
        """
        Generate synthetic data that mimics realistic EEG patterns.
        
        This creates a sum of sinusoids with different frequencies plus noise,
        which approximates realistic brain activity patterns.
        
        Args:
            t_array (np.ndarray): Time points array
            
        Returns:
            np.ndarray: Array of shape (len(t_array), dims) containing synthetic data
        """
        result = np.zeros((len(t_array), self.dims))
        
        # Number of component frequencies to mix
        n_components = 5
        
        for d in range(self.dims):
            signal = np.zeros(len(t_array))
            
            # Create a mix of different frequency components
            for i in range(n_components):
                # Generate frequencies in typical EEG bands (0.5-30 Hz)
                component_freq = self.freq * (i + 1) * self.freq_factors[d]
                # Amplitude decreases for higher frequencies
                amplitude = 1.0 / (i + 1)
                # Random phase
                phase = np.random.uniform(0, 2*np.pi)
                
                signal += amplitude * np.sin(2 * np.pi * component_freq * t_array + phase)
            
            # Add some Gaussian noise
            noise = np.random.normal(0, 0.1, len(t_array))
            signal += noise
            
            # Normalize to range [-1, 1]
            signal = signal / (np.max(np.abs(signal)) + 1e-10)
            
            result[:, d] = signal
            
        return result
    
    def _load_external_data(self, t_array: np.ndarray) -> np.ndarray:
        """
        Load external data using the provided data_loader function.
        
        Args:
            t_array (np.ndarray): Time points array
            
        Returns:
            np.ndarray: Array of shape (len(t_array), dims) containing loaded data
            
        Raises:
            ValueError: If data_loader is not provided when kind="data" and using external data
        """
        if self.data_loader is None:
            raise ValueError("data_loader must be provided when kind='data' and using external data")
        
        return self.data_loader(t_array, self.dims)
    
    def generate(self, t_array: np.ndarray) -> np.ndarray:
        """
        Generate emotion signals based on the specified waveform type.
        
        Args:
            t_array (np.ndarray): Time points array
            
        Returns:
            np.ndarray: Array of shape (len(t_array), dims) containing emotion values vs. time
            
        Raises:
            ValueError: If an invalid kind is specified
        """
        if self.kind == "square":
            return self._generate_square(t_array)
        elif self.kind == "sine":
            return self._generate_sine(t_array)
        elif self.kind == "data":
            # If data_loader is provided, use it; otherwise generate synthetic data
            if self.data_loader is not None:
                return self._load_external_data(t_array)
            else:
                return self._generate_synthetic_data(t_array)
        else:
            raise ValueError(f"Invalid kind: {self.kind}")


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create time array
    dt = 0.01  # 10 ms time step
    duration = 5.0  # 5 seconds
    t = np.arange(0, duration, dt)
    
    # Test different waveforms
    signal_types = ["square", "sine", "data"]
    
    fig, axes = plt.subplots(len(signal_types), 1, figsize=(10, 8), sharex=True)
    
    for i, kind in enumerate(signal_types):
        # Create signal generator
        signal_gen = EmotionSignal(kind=kind, freq=0.5, dims=2)
        
        # Generate signals
        signals = signal_gen.generate(t)
        
        # Plot
        axes[i].plot(t, signals[:, 0], label="Dimension 1")
        axes[i].plot(t, signals[:, 1], label="Dimension 2")
        axes[i].set_title(f"{kind.capitalize()} Waveform")
        axes[i].set_ylabel("Amplitude")
        axes[i].legend()
        axes[i].grid(True)
    
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EmotionSignal Module for Brain-to-Brain Interface Simulation

This module provides the EmotionSignal class for generating synthetic emotion signals
with different waveform types (square, sine, or data) across multiple channels,
with configurable noise and artifacts.
"""

import numpy as np
import pandas as pd
import os
from typing import Callable, Dict, List, Optional, Tuple, Union, Any

# Conditional import for EDF support
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False


class EmotionSignal:
    """
    A class for generating synthetic emotion signals with different waveforms across multiple channels.
    
    The EmotionSignal class can generate square waves, sine waves, or more complex
    synthetic data that mimics realistic EEG patterns. It supports multiple emotion
    dimensions (e.g., valence and arousal) and multiple EEG channels with configurable
    noise and artifacts.
    
    Attributes:
        kind (str): Type of waveform ("square", "sine", or "data")
        freq (float): Base frequency in Hz
        dims (int): Number of emotion dimensions/channels
        channels (int): Number of EEG channels
        channel_names (List[str]): Names of EEG channels
        channel_amplitudes (List[float]): Relative amplitude per channel
        noise_config (Dict): Configuration for noise generation
        data_loading_config (Dict): Configuration for data loading
        data_loader (Callable, optional): Function to load external data
        external_data (np.ndarray, optional): Loaded external data
    """
    
    def __init__(
        self, 
        config: Dict[str, Any]
    ):
        """
        Initialize the EmotionSignal generator.
        
        Args:
            config (Dict): Configuration dictionary containing parameters for signal generation
                
        Raises:
            ValueError: If configuration is invalid
        """
        # Extract parameters from config
        self.kind = config.get('kind', 'sine')
        if self.kind not in ["square", "sine", "data"]:
            raise ValueError("kind must be one of 'square', 'sine', or 'data'")
        
        self.freq = config.get('freq', 0.2)
        self.dims = config.get('dims', 2)
        self.channels = config.get('channels', 8)
        
        # Channel configuration
        self.channel_names = config.get('channel_names', 
                                       [f"Ch{i+1}" for i in range(self.channels)])
        
        # Ensure we have the correct number of channel names
        if len(self.channel_names) != self.channels:
            self.channel_names = [f"Ch{i+1}" for i in range(self.channels)]
        
        # Channel amplitudes
        self.channel_amplitudes = config.get('channel_amplitudes', 
                                           [1.0] * self.channels)
        
        # Ensure we have the correct number of channel amplitudes
        if len(self.channel_amplitudes) != self.channels:
            self.channel_amplitudes = [1.0] * self.channels
        
        # Noise configuration
        self.noise_config = config.get('noise', {
            'gaussian': {'enabled': True, 'std': 0.1},
            'powerline': {'enabled': False, 'freq': 50, 'amplitude': 0.05}
        })
        
        # Data loading configuration
        self.data_loading_config = config.get('data_loading', {
            'enabled': False,
            'file_path': '',
            'format': 'csv'
        })
        
        # Phase shifts for different dimensions to create variation
        self.phase_shifts = np.linspace(0, 2*np.pi, self.dims, endpoint=False)
        
        # Frequency variations for different dimensions
        self.freq_factors = np.linspace(0.8, 1.2, self.dims)
        
        # Initialize external data
        self.external_data = None
        self.data_loader = None
        
        # Load external data if enabled
        if self.data_loading_config['enabled'] and self.kind == 'data':
            self.load_from_file(
                self.data_loading_config['file_path'],
                self.data_loading_config['format']
            )
    
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
        Use loaded external data.
        
        Args:
            t_array (np.ndarray): Time points array
            
        Returns:
            np.ndarray: Array of shape (len(t_array), dims) containing loaded data
            
        Raises:
            ValueError: If external data is not loaded
        """
        if self.external_data is None:
            raise ValueError("External data must be loaded before using")
        
        # Get the number of samples in the external data
        n_samples_external = self.external_data.shape[0]
        
        # Get the number of samples requested
        n_samples_requested = len(t_array)
        
        # If requested samples is greater than available samples, loop the data
        if n_samples_requested > n_samples_external:
            # Calculate how many times to repeat the data
            repeat_count = int(np.ceil(n_samples_requested / n_samples_external))
            
            # Repeat the data
            repeated_data = np.tile(self.external_data, (repeat_count, 1))
            
            # Trim to requested length
            result = repeated_data[:n_samples_requested, :self.dims]
        else:
            # Use a subset of the data
            result = self.external_data[:n_samples_requested, :self.dims]
        
        return result
    
    def load_from_file(self, file_path: str, format: str = 'csv') -> None:
        """
        Load signals from file.
        
        Args:
            file_path (str): Path to the file
            format (str): File format ('csv' or 'edf')
            
        Raises:
            ValueError: If format is not supported or file cannot be loaded
            ImportError: If required dependencies are not installed
        """
        if not os.path.exists(file_path):
            raise ValueError(f"File not found: {file_path}")
        
        if format.lower() == 'csv':
            # Load CSV file
            try:
                data = pd.read_csv(file_path)
                self.external_data = data.values
            except Exception as e:
                raise ValueError(f"Failed to load CSV file: {e}")
        elif format.lower() == 'edf':
            # Load EDF file
            if not MNE_AVAILABLE:
                raise ImportError("MNE is required for EDF support. Install with 'pip install mne'")
            
            try:
                raw = mne.io.read_raw_edf(file_path, preload=True)
                data = raw.get_data().T  # Transpose to get (samples, channels)
                self.external_data = data
            except Exception as e:
                raise ValueError(f"Failed to load EDF file: {e}")
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Check if we have enough dimensions
        if self.external_data.shape[1] < self.dims:
            raise ValueError(f"External data has {self.external_data.shape[1]} dimensions, but {self.dims} are required")
    
    def add_artifacts(self, signals: np.ndarray) -> np.ndarray:
        """
        Add artifacts to signals.
        
        Args:
            signals (np.ndarray): Signals array of shape (samples, channels)
            
        Returns:
            np.ndarray: Signals with artifacts
        """
        result = signals.copy()
        
        # Add Gaussian noise if enabled
        if self.noise_config['gaussian']['enabled']:
            std = self.noise_config['gaussian']['std']
            noise = np.random.normal(0, std, size=result.shape)
            result += noise
        
        # Add power-line artifacts if enabled
        if self.noise_config['powerline']['enabled']:
            freq = self.noise_config['powerline']['freq']
            amplitude = self.noise_config['powerline']['amplitude']
            
            # Get time array based on the number of samples
            n_samples = result.shape[0]
            dt = 1.0 / (freq * 10)  # Assuming 10 samples per cycle
            t = np.arange(n_samples) * dt
            
            # Generate power-line noise
            powerline_noise = amplitude * np.sin(2 * np.pi * freq * t)
            
            # Add to all channels
            for c in range(result.shape[1]):
                result[:, c] += powerline_noise
        
        return result
    
    def map_to_channels(self, emotion_signals: np.ndarray) -> np.ndarray:
        """
        Map emotion dimension signals to EEG channels.
        
        Args:
            emotion_signals (np.ndarray): Emotion signals of shape (samples, dims)
            
        Returns:
            np.ndarray: EEG channel signals of shape (samples, channels)
        """
        n_samples = emotion_signals.shape[0]
        channel_signals = np.zeros((n_samples, self.channels))
        
        # For each channel, create a weighted sum of emotion dimensions
        for c in range(self.channels):
            # Generate weights for this channel (normalized random weights)
            weights = np.random.rand(self.dims)
            weights /= np.sum(weights)
            
            # Apply weights to emotion dimensions
            for d in range(self.dims):
                channel_signals[:, c] += weights[d] * emotion_signals[:, d]
            
            # Apply channel amplitude
            channel_signals[:, c] *= self.channel_amplitudes[c]
        
        return channel_signals
    
    def generate(self, t_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate emotion signals and map to EEG channels.
        
        Args:
            t_array (np.ndarray): Time points array
            
        Returns:
            Tuple containing:
                emotion_signals (np.ndarray): Emotion signals of shape (samples, dims)
                channel_signals (np.ndarray): EEG channel signals of shape (samples, channels)
            
        Raises:
            ValueError: If an invalid kind is specified
        """
        # Generate emotion dimension signals
        if self.kind == "square":
            emotion_signals = self._generate_square(t_array)
        elif self.kind == "sine":
            emotion_signals = self._generate_sine(t_array)
        elif self.kind == "data":
            if self.external_data is not None:
                emotion_signals = self._load_external_data(t_array)
            else:
                emotion_signals = self._generate_synthetic_data(t_array)
        else:
            raise ValueError(f"Invalid kind: {self.kind}")
        
        # Map emotion dimensions to EEG channels
        channel_signals = self.map_to_channels(emotion_signals)
        
        # Add artifacts
        channel_signals = self.add_artifacts(channel_signals)
        
        return emotion_signals, channel_signals


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from config.config_loader import ConfigLoader
    
    # Load configuration
    config_loader = ConfigLoader()
    config = config_loader.get_section('emotion_signal')
    
    # Create signal generator
    signal_gen = EmotionSignal(config)
    
    # Create time array
    dt = 0.01  # 10 ms time step
    duration = 5.0  # 5 seconds
    t = np.arange(0, duration, dt)
    
    # Generate signals
    emotion_signals, channel_signals = signal_gen.generate(t)
    
    # Plot emotion signals
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for d in range(signal_gen.dims):
        plt.plot(t, emotion_signals[:, d], label=f"Emotion Dim {d+1}")
    plt.title("Emotion Dimension Signals")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Plot channel signals (first 4 channels)
    plt.subplot(2, 1, 2)
    for c in range(min(4, signal_gen.channels)):
        plt.plot(t, channel_signals[:, c], label=f"Channel {signal_gen.channel_names[c]}")
    plt.title("EEG Channel Signals (First 4 Channels)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

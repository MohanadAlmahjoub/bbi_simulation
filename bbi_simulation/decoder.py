#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decoder Module for Brain-to-Brain Interface Simulation

This module provides the Decoder class for decoding simulated EEG signals
into emotion values, with support for both simple averaging and ML-based approaches.
"""

import numpy as np
from typing import Optional, Union, Tuple, List
from sklearn.base import BaseEstimator
from sklearn.svm import SVR


class Decoder:
    """
    A class for decoding simulated EEG signals into emotion values.
    
    The Decoder class can use either a simple averaging approach with additive
    Gaussian noise or an ML-based approach using scikit-learn models.
    
    Attributes:
        noise_std (float): Standard deviation for additive Gaussian noise
        use_ml (bool): Whether to use ML-based decoding
        model (BaseEstimator): Scikit-learn model for ML-based decoding
        window_size (int): Size of the sliding window for decoding
        is_fitted (bool): Whether the ML model has been fitted
    """
    
    def __init__(
        self, 
        noise_std: float = 0.1, 
        use_ml: bool = False,
        window_size: int = 50,
        model: Optional[BaseEstimator] = None
    ):
        """
        Initialize the Decoder.
        
        Args:
            noise_std (float): Standard deviation for additive Gaussian noise
            use_ml (bool): Whether to use ML-based decoding
            window_size (int): Size of the sliding window for decoding
            model (BaseEstimator, optional): Scikit-learn model for ML-based decoding
                                           (defaults to SVR if None and use_ml=True)
        """
        self.noise_std = noise_std
        self.use_ml = use_ml
        self.window_size = window_size
        self.is_fitted = False
        
        # Set up the ML model if requested
        if use_ml:
            if model is None:
                # Default to Support Vector Regression
                self.model = SVR(kernel='rbf', gamma='scale')
            else:
                self.model = model
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the ML model with training data.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, window_size, dims)
            y (np.ndarray): Target values of shape (n_samples, dims)
            
        Raises:
            ValueError: If use_ml is False but fit is called
        """
        if not self.use_ml:
            raise ValueError("Cannot fit model when use_ml=False")
        
        # Reshape X to 2D for scikit-learn
        n_samples, window_size, dims = X.shape
        X_reshaped = X.reshape(n_samples, window_size * dims)
        
        # Fit a separate model for each dimension
        self.models = []
        for d in range(y.shape[1]):
            model_copy = clone_model(self.model)
            model_copy.fit(X_reshaped, y[:, d])
            self.models.append(model_copy)
        
        self.is_fitted = True
    
    def _decode_simple(self, eeg_chunk: np.ndarray) -> np.ndarray:
        """
        Decode EEG chunk using simple averaging with additive Gaussian noise.
        
        Args:
            eeg_chunk (np.ndarray): EEG data chunk of shape (window_size, dims)
            
        Returns:
            np.ndarray: Decoded emotion values of shape (dims,)
        """
        # Average over the window
        decoded = np.mean(eeg_chunk, axis=0)
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_std, size=decoded.shape)
        decoded += noise
        
        return decoded
    
    def _decode_ml(self, eeg_chunk: np.ndarray) -> np.ndarray:
        """
        Decode EEG chunk using ML-based approach.
        
        Args:
            eeg_chunk (np.ndarray): EEG data chunk of shape (window_size, dims)
            
        Returns:
            np.ndarray: Decoded emotion values of shape (dims,)
            
        Raises:
            ValueError: If the model has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decoding")
        
        # Reshape for scikit-learn
        window_size, dims = eeg_chunk.shape
        X = eeg_chunk.reshape(1, window_size * dims)
        
        # Predict using each dimension's model
        decoded = np.zeros(dims)
        for d in range(dims):
            decoded[d] = self.models[d].predict(X)[0]
        
        # Add a small amount of noise for realism
        noise = np.random.normal(0, self.noise_std * 0.1, size=decoded.shape)
        decoded += noise
        
        return decoded
    
    def decode(self, eeg_chunk: np.ndarray) -> np.ndarray:
        """
        Decode EEG chunk into emotion values.
        
        Args:
            eeg_chunk (np.ndarray): EEG data chunk of shape (window_size, dims)
            
        Returns:
            np.ndarray: Decoded emotion values of shape (dims,)
        """
        if self.use_ml:
            return self._decode_ml(eeg_chunk)
        else:
            return self._decode_simple(eeg_chunk)


def clone_model(model: BaseEstimator) -> BaseEstimator:
    """
    Create a deep copy of a scikit-learn model.
    
    Args:
        model (BaseEstimator): Model to clone
        
    Returns:
        BaseEstimator: Cloned model
    """
    from sklearn.base import clone
    return clone(model)


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from emotion_signal import EmotionSignal
    
    # Create time array
    dt = 0.01  # 10 ms time step
    duration = 5.0  # 5 seconds
    t = np.arange(0, duration, dt)
    
    # Create signal generator
    signal_gen = EmotionSignal(kind="sine", freq=0.5, dims=2)
    
    # Generate signals
    signals = signal_gen.generate(t)
    
    # Create decoder
    window_size = 50  # 0.5 seconds at dt=0.01
    decoder = Decoder(noise_std=0.2, window_size=window_size)
    
    # Decode signals
    n_samples = len(t)
    decoded = np.zeros((n_samples, 2))
    
    for i in range(window_size, n_samples):
        # Extract window
        window = signals[i-window_size:i, :]
        
        # Decode
        decoded[i, :] = decoder.decode(window)
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Original signals
    axes[0].plot(t, signals[:, 0], label="Original Dim 1")
    axes[0].plot(t, signals[:, 1], label="Original Dim 2")
    axes[0].set_title("Original Signals")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True)
    
    # Decoded signals
    axes[1].plot(t, decoded[:, 0], label="Decoded Dim 1")
    axes[1].plot(t, decoded[:, 1], label="Decoded Dim 2")
    axes[1].set_title("Decoded Signals")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

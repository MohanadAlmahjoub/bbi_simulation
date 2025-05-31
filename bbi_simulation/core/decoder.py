#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Decoder Module for Brain-to-Brain Interface Simulation

This module provides the Decoder class for decoding simulated EEG signals
into emotion values, with support for both simple averaging and ML-based approaches.
"""

import numpy as np
import os
import pickle
import joblib
from typing import Dict, List, Optional, Tuple, Union, Any

from sklearn.base import BaseEstimator
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class Decoder:
    """
    A class for decoding simulated EEG signals into emotion values.
    
    The Decoder class can use either a simple averaging approach with additive
    Gaussian noise or an ML-based approach using scikit-learn models.
    
    Attributes:
        type (str): Type of decoder ("simple" or "ml")
        noise_std (float): Standard deviation for additive Gaussian noise
        window_size (int): Size of the sliding window for decoding
        ml_config (Dict): Configuration for ML-based decoding
        model_type (str): Type of ML model ("svm", "rf", "mlp", or "auto")
        models (List[BaseEstimator]): List of trained models (one per dimension)
        scaler (StandardScaler): Feature scaler for ML models
        is_fitted (bool): Whether the ML model has been fitted
    """
    
    def __init__(
        self, 
        config: Dict[str, Any]
    ):
        """
        Initialize the Decoder.
        
        Args:
            config (Dict): Configuration dictionary containing parameters for decoding
                
        Raises:
            ValueError: If configuration is invalid
        """
        # Extract parameters from config
        self.type = config.get('type', 'simple')
        if self.type not in ["simple", "ml"]:
            raise ValueError("type must be one of 'simple' or 'ml'")
        
        self.noise_std = config.get('noise_std', 0.1)
        self.window_size = config.get('window_size', 50)
        
        # ML configuration
        self.ml_config = config.get('ml', {})
        self.model_type = self.ml_config.get('model_type', 'auto')
        if self.model_type not in ["svm", "rf", "mlp", "auto"]:
            raise ValueError(f"Invalid model_type: {self.model_type}")
        
        # Initialize models
        self.models = []
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Load model if specified
        if self.type == "ml" and self.ml_config.get('save_model', False):
            model_path = self.ml_config.get('model_path', '')
            if model_path and os.path.exists(model_path):
                self.load_model(model_path)
    
    def _create_model(self, model_type: str) -> BaseEstimator:
        """
        Create a new model instance based on the specified type.
        
        Args:
            model_type (str): Type of model ("svm", "rf", or "mlp")
            
        Returns:
            BaseEstimator: New model instance
            
        Raises:
            ValueError: If model_type is invalid
        """
        if model_type == "svm":
            # Get SVM parameters from config
            svm_config = self.ml_config.get('svm', {})
            kernel = svm_config.get('kernel', 'rbf')
            C = svm_config.get('C', 1.0)
            gamma = svm_config.get('gamma', 'scale')
            
            return SVR(kernel=kernel, C=C, gamma=gamma)
        
        elif model_type == "rf":
            # Get Random Forest parameters from config
            rf_config = self.ml_config.get('rf', {})
            n_estimators = rf_config.get('n_estimators', 100)
            max_depth = rf_config.get('max_depth', None)
            min_samples_split = rf_config.get('min_samples_split', 2)
            
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
        
        elif model_type == "mlp":
            # Get MLP parameters from config
            mlp_config = self.ml_config.get('mlp', {})
            hidden_layer_sizes = mlp_config.get('hidden_layer_sizes', [100])
            activation = mlp_config.get('activation', 'relu')
            solver = mlp_config.get('solver', 'adam')
            max_iter = mlp_config.get('max_iter', 200)
            
            return MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=activation,
                solver=solver,
                max_iter=max_iter,
                random_state=42
            )
        
        else:
            raise ValueError(f"Invalid model_type: {model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train ML models and select the best one based on cross-validated F1-score.
        
        Args:
            X (np.ndarray): Training data of shape (n_samples, window_size, channels)
            y (np.ndarray): Target values of shape (n_samples, dims)
            
        Returns:
            Dict: Training results including best model type and performance metrics
            
        Raises:
            ValueError: If type is not "ml"
        """
        if self.type != "ml":
            raise ValueError("Cannot train model when type is not 'ml'")
        
        # Reshape X to 2D for scikit-learn
        n_samples, window_size, channels = X.shape
        X_reshaped = X.reshape(n_samples, window_size * channels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_reshaped)
        
        # Number of dimensions
        n_dims = y.shape[1]
        
        # Number of cross-validation folds
        cv_folds = self.ml_config.get('cv_folds', 5)
        
        # Results dictionary
        results = {
            'model_types': [],
            'f1_scores': [],
            'accuracy_scores': [],
            'best_model_type': '',
            'best_f1_score': 0.0,
            'best_accuracy_score': 0.0
        }
        
        # If model_type is "auto", try all model types
        model_types = ["svm", "rf", "mlp"] if self.model_type == "auto" else [self.model_type]
        
        # Train and evaluate each model type
        for model_type in model_types:
            # Create models for each dimension
            models = []
            f1_scores = []
            accuracy_scores = []
            
            for d in range(n_dims):
                # Create model
                model = self._create_model(model_type)
                
                # Binarize targets for F1 score calculation
                # (assuming values > 0 are positive class)
                y_binary = (y[:, d] > 0).astype(int)
                
                # Perform cross-validation
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                # Store predictions for each fold
                y_pred_all = np.zeros_like(y_binary)
                
                # Perform manual cross-validation to get predictions
                for train_idx, test_idx in cv.split(X_scaled, y_binary):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y[train_idx, d], y[test_idx, d]
                    
                    # Fit model
                    model.fit(X_train, y_train)
                    
                    # Predict
                    y_pred = model.predict(X_test)
                    
                    # Binarize predictions
                    y_pred_binary = (y_pred > 0).astype(int)
                    
                    # Store predictions
                    y_pred_all[test_idx] = y_pred_binary
                
                # Calculate metrics
                f1 = f1_score(y_binary, y_pred_all, average='macro')
                acc = accuracy_score(y_binary, y_pred_all)
                
                f1_scores.append(f1)
                accuracy_scores.append(acc)
                
                # Fit model on all data
                model.fit(X_scaled, y[:, d])
                models.append(model)
            
            # Calculate mean metrics
            mean_f1 = np.mean(f1_scores)
            mean_acc = np.mean(accuracy_scores)
            
            # Store results
            results['model_types'].append(model_type)
            results['f1_scores'].append(mean_f1)
            results['accuracy_scores'].append(mean_acc)
            
            # Update best model if this one is better
            if mean_f1 > results['best_f1_score'] or (
                mean_f1 == results['best_f1_score'] and mean_acc > results['best_accuracy_score']
            ):
                results['best_model_type'] = model_type
                results['best_f1_score'] = mean_f1
                results['best_accuracy_score'] = mean_acc
                
                # Store best models
                self.models = models
        
        # If we tried multiple model types, train the best one
        if self.model_type == "auto" and results['best_model_type'] != model_types[-1]:
            # Create and train models of the best type
            self.models = []
            for d in range(n_dims):
                model = self._create_model(results['best_model_type'])
                model.fit(X_scaled, y[:, d])
                self.models.append(model)
        
        self.is_fitted = True
        
        # Save model if configured
        if self.ml_config.get('save_model', False):
            model_path = self.ml_config.get('model_path', 'models/decoder_model.pkl')
            self.save_model(model_path)
        
        return results
    
    def save_model(self, path: str) -> None:
        """
        Save trained model to file.
        
        Args:
            path (str): Path to save the model
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save model and scaler
        with open(path, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted
            }, f)
    
    def load_model(self, path: str) -> None:
        """
        Load trained model from file.
        
        Args:
            path (str): Path to load the model from
            
        Raises:
            ValueError: If file does not exist or is invalid
        """
        if not os.path.exists(path):
            raise ValueError(f"Model file not found: {path}")
        
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                
                self.models = data['models']
                self.scaler = data['scaler']
                self.is_fitted = data['is_fitted']
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")
    
    def _decode_simple(self, eeg_chunk: np.ndarray) -> np.ndarray:
        """
        Decode EEG chunk using simple averaging with additive Gaussian noise.
        
        Args:
            eeg_chunk (np.ndarray): EEG data chunk of shape (window_size, channels)
            
        Returns:
            np.ndarray: Decoded emotion values of shape (dims,)
        """
        # For simple decoding, we assume the first 'dims' channels correspond to emotion dimensions
        # and average over the window for each channel
        n_dims = min(eeg_chunk.shape[1], 2)  # Default to 2 dimensions if not specified
        
        # Average over the window for each channel
        decoded = np.mean(eeg_chunk, axis=0)[:n_dims]
        
        # Add Gaussian noise
        noise = np.random.normal(0, self.noise_std, size=decoded.shape)
        decoded += noise
        
        return decoded
    
    def _decode_ml(self, eeg_chunk: np.ndarray) -> np.ndarray:
        """
        Decode EEG chunk using ML-based approach.
        
        Args:
            eeg_chunk (np.ndarray): EEG data chunk of shape (window_size, channels)
            
        Returns:
            np.ndarray: Decoded emotion values of shape (dims,)
            
        Raises:
            ValueError: If the model has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before decoding")
        
        # Reshape for scikit-learn
        window_size, channels = eeg_chunk.shape
        X = eeg_chunk.reshape(1, window_size * channels)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict using each dimension's model
        n_dims = len(self.models)
        decoded = np.zeros(n_dims)
        
        for d in range(n_dims):
            decoded[d] = self.models[d].predict(X_scaled)[0]
        
        # Add a small amount of noise for realism
        noise = np.random.normal(0, self.noise_std * 0.1, size=decoded.shape)
        decoded += noise
        
        return decoded
    
    def decode(self, eeg_chunk: np.ndarray) -> np.ndarray:
        """
        Decode EEG chunk into emotion values.
        
        Args:
            eeg_chunk (np.ndarray): EEG data chunk of shape (window_size, channels)
            
        Returns:
            np.ndarray: Decoded emotion values of shape (dims,)
        """
        if self.type == "ml":
            return self._decode_ml(eeg_chunk)
        else:
            return self._decode_simple(eeg_chunk)


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from config.config_loader import ConfigLoader
    from core.emotion_signal import EmotionSignal
    
    # Load configuration
    config_loader = ConfigLoader()
    es_config = config_loader.get_section('emotion_signal')
    dec_config = config_loader.get_section('decoder')
    
    # Create signal generator
    signal_gen = EmotionSignal(es_config)
    
    # Create time array
    dt = 0.01  # 10 ms time step
    duration = 5.0  # 5 seconds
    t = np.arange(0, duration, dt)
    
    # Generate signals
    emotion_signals, channel_signals = signal_gen.generate(t)
    
    # Create decoder
    decoder = Decoder(dec_config)
    
    # Decode signals
    n_samples = len(t)
    window_size = decoder.window_size
    decoded = np.zeros((n_samples, signal_gen.dims))
    
    for i in range(window_size, n_samples):
        # Extract window
        window = channel_signals[i-window_size:i, :]
        
        # Decode
        decoded[i, :] = decoder.decode(window)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    # Original emotion signals
    plt.subplot(2, 1, 1)
    for d in range(signal_gen.dims):
        plt.plot(t, emotion_signals[:, d], label=f"Original Dim {d+1}")
    plt.title("Original Emotion Signals")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    # Decoded signals
    plt.subplot(2, 1, 2)
    for d in range(signal_gen.dims):
        plt.plot(t, decoded[:, d], label=f"Decoded Dim {d+1}")
    plt.title("Decoded Signals")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Example of training ML models
    if decoder.type == "ml":
        print("Training ML models...")
        
        # Generate training data
        n_samples = 1000
        X_train = np.zeros((n_samples, window_size, signal_gen.channels))
        y_train = np.zeros((n_samples, signal_gen.dims))
        
        for i in range(n_samples):
            # Generate random time array
            t_sample = np.arange(0, window_size * dt, dt)
            
            # Generate signals
            emotion_sample, channel_sample = signal_gen.generate(t_sample)
            
            # Store samples
            X_train[i] = channel_sample
            y_train[i] = emotion_sample[-1]  # Use last value as target
        
        # Train models
        results = decoder.train(X_train, y_train)
        
        # Print results
        print(f"Best model type: {results['best_model_type']}")
        print(f"Best F1 score: {results['best_f1_score']:.4f}")
        print(f"Best accuracy score: {results['best_accuracy_score']:.4f}")

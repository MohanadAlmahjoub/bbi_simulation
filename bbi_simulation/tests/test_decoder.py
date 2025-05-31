#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Decoder module
"""

import pytest
import numpy as np
import os
import tempfile
import pickle
from bbi_simulation.core.decoder import Decoder


class TestDecoder:
    """Test suite for Decoder class"""
    
    def test_init_simple(self):
        """Test initialization with simple decoder"""
        config = {
            'type': 'simple',
            'noise_std': 0.1,
            'window_size': 50
        }
        decoder = Decoder(config)
        
        assert decoder.type == 'simple'
        assert decoder.noise_std == 0.1
        assert decoder.window_size == 50
        assert not decoder.is_fitted
    
    def test_init_ml(self):
        """Test initialization with ML decoder"""
        config = {
            'type': 'ml',
            'noise_std': 0.1,
            'window_size': 50,
            'ml': {
                'model_type': 'svm',
                'cv_folds': 5
            }
        }
        decoder = Decoder(config)
        
        assert decoder.type == 'ml'
        assert decoder.noise_std == 0.1
        assert decoder.window_size == 50
        assert decoder.model_type == 'svm'
        assert not decoder.is_fitted
    
    def test_init_invalid_type(self):
        """Test initialization with invalid decoder type"""
        config = {
            'type': 'invalid',
            'noise_std': 0.1,
            'window_size': 50
        }
        
        with pytest.raises(ValueError):
            Decoder(config)
    
    def test_decode_simple(self):
        """Test simple decoding"""
        config = {
            'type': 'simple',
            'noise_std': 0.0,  # No noise for deterministic testing
            'window_size': 10
        }
        decoder = Decoder(config)
        
        # Create test EEG chunk
        eeg_chunk = np.ones((10, 8))  # 10 samples, 8 channels
        
        # Decode
        decoded = decoder.decode(eeg_chunk)
        
        # Check shape (should be 2 dimensions by default)
        assert decoded.shape == (2,)
        
        # Check values (should be 1.0 for each dimension)
        assert np.allclose(decoded, np.ones(2), atol=1e-6)
    
    def test_decode_ml_not_fitted(self):
        """Test ML decoding without fitting"""
        config = {
            'type': 'ml',
            'noise_std': 0.1,
            'window_size': 10,
            'ml': {
                'model_type': 'svm'
            }
        }
        decoder = Decoder(config)
        
        # Create test EEG chunk
        eeg_chunk = np.ones((10, 8))  # 10 samples, 8 channels
        
        # Decode should raise error
        with pytest.raises(ValueError):
            decoder.decode(eeg_chunk)
    
    def test_train_simple(self):
        """Test training with simple decoder"""
        config = {
            'type': 'simple',
            'noise_std': 0.1,
            'window_size': 10
        }
        decoder = Decoder(config)
        
        # Create training data
        X = np.ones((100, 10, 8))  # 100 samples, 10 time steps, 8 channels
        y = np.ones((100, 2))  # 100 samples, 2 dimensions
        
        # Training should raise error for simple decoder
        with pytest.raises(ValueError):
            decoder.train(X, y)
    
    def test_train_ml(self):
        """Test training with ML decoder"""
        config = {
            'type': 'ml',
            'noise_std': 0.1,
            'window_size': 10,
            'ml': {
                'model_type': 'svm',
                'cv_folds': 2  # Small value for faster testing
            }
        }
        decoder = Decoder(config)
        
        # Create training data
        np.random.seed(42)  # For reproducibility
        X = np.random.randn(100, 10, 8)  # 100 samples, 10 time steps, 8 channels
        y = np.random.choice([-1, 1], size=(100, 2))  # Binary targets
        
        # Train
        results = decoder.train(X, y)
        
        # Check that model is fitted
        assert decoder.is_fitted
        assert len(decoder.models) == 2  # One model per dimension
        
        # Check results
        assert 'best_model_type' in results
        assert 'best_f1_score' in results
        assert 'best_accuracy_score' in results
    
    def test_train_ml_auto(self):
        """Test training with auto model selection"""
        config = {
            'type': 'ml',
            'noise_std': 0.1,
            'window_size': 10,
            'ml': {
                'model_type': 'auto',
                'cv_folds': 2  # Small value for faster testing
            }
        }
        decoder = Decoder(config)
        
        # Create training data
        np.random.seed(42)  # For reproducibility
        X = np.random.randn(100, 10, 8)  # 100 samples, 10 time steps, 8 channels
        y = np.random.choice([-1, 1], size=(100, 2))  # Binary targets
        
        # Train
        results = decoder.train(X, y)
        
        # Check that model is fitted
        assert decoder.is_fitted
        assert len(decoder.models) == 2  # One model per dimension
        
        # Check results
        assert 'best_model_type' in results
        assert results['best_model_type'] in ['svm', 'rf', 'mlp']
        assert 'best_f1_score' in results
        assert 'best_accuracy_score' in results
    
    def test_save_load_model(self):
        """Test saving and loading model"""
        config = {
            'type': 'ml',
            'noise_std': 0.1,
            'window_size': 10,
            'ml': {
                'model_type': 'svm',
                'cv_folds': 2  # Small value for faster testing
            }
        }
        decoder = Decoder(config)
        
        # Create training data
        np.random.seed(42)  # For reproducibility
        X = np.random.randn(100, 10, 8)  # 100 samples, 10 time steps, 8 channels
        y = np.random.choice([-1, 1], size=(100, 2))  # Binary targets
        
        # Train
        decoder.train(X, y)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save model
            decoder.save_model(temp_path)
            
            # Create new decoder
            new_decoder = Decoder(config)
            
            # Load model
            new_decoder.load_model(temp_path)
            
            # Check that model is loaded
            assert new_decoder.is_fitted
            assert len(new_decoder.models) == 2  # One model per dimension
            
            # Create test EEG chunk
            eeg_chunk = np.random.randn(10, 8)  # 10 samples, 8 channels
            
            # Decode with both decoders
            decoded1 = decoder.decode(eeg_chunk)
            decoded2 = new_decoder.decode(eeg_chunk)
            
            # Check that results are the same
            assert np.allclose(decoded1, decoded2)
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_save_model_not_fitted(self):
        """Test saving model that is not fitted"""
        config = {
            'type': 'ml',
            'noise_std': 0.1,
            'window_size': 10,
            'ml': {
                'model_type': 'svm'
            }
        }
        decoder = Decoder(config)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save model should raise error
            with pytest.raises(ValueError):
                decoder.save_model(temp_path)
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_load_model_invalid_path(self):
        """Test loading model from invalid path"""
        config = {
            'type': 'ml',
            'noise_std': 0.1,
            'window_size': 10,
            'ml': {
                'model_type': 'svm'
            }
        }
        decoder = Decoder(config)
        
        # Load model should raise error
        with pytest.raises(ValueError):
            decoder.load_model('/path/to/nonexistent/model.pkl')

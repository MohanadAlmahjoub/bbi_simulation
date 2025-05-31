#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for EmotionSignal module
"""

import pytest
import numpy as np
import os
import tempfile
import pandas as pd
from bbi_simulation.core.emotion_signal import EmotionSignal


class TestEmotionSignal:
    """Test suite for EmotionSignal class"""
    
    def test_init(self):
        """Test initialization with default parameters"""
        config = {
            'kind': 'sine',
            'freq': 0.2,
            'dims': 2,
            'channels': 8
        }
        signal = EmotionSignal(config)
        
        assert signal.kind == 'sine'
        assert signal.freq == 0.2
        assert signal.dims == 2
        assert signal.channels == 8
        assert len(signal.channel_names) == 8
        assert len(signal.channel_amplitudes) == 8
    
    def test_init_invalid_kind(self):
        """Test initialization with invalid kind"""
        config = {
            'kind': 'invalid',
            'freq': 0.2,
            'dims': 2,
            'channels': 8
        }
        
        with pytest.raises(ValueError):
            EmotionSignal(config)
    
    def test_generate_sine(self):
        """Test sine wave generation"""
        config = {
            'kind': 'sine',
            'freq': 0.2,
            'dims': 2,
            'channels': 8
        }
        signal = EmotionSignal(config)
        
        # Generate signals
        t = np.arange(0, 1.0, 0.01)  # 1 second at 100 Hz
        emotion_signals, channel_signals = signal.generate(t)
        
        # Check shapes
        assert emotion_signals.shape == (len(t), 2)
        assert channel_signals.shape == (len(t), 8)
        
        # Check range
        assert np.all(emotion_signals >= -1.0) and np.all(emotion_signals <= 1.0)
    
    def test_generate_square(self):
        """Test square wave generation"""
        config = {
            'kind': 'square',
            'freq': 0.2,
            'dims': 2,
            'channels': 8
        }
        signal = EmotionSignal(config)
        
        # Generate signals
        t = np.arange(0, 1.0, 0.01)  # 1 second at 100 Hz
        emotion_signals, channel_signals = signal.generate(t)
        
        # Check shapes
        assert emotion_signals.shape == (len(t), 2)
        assert channel_signals.shape == (len(t), 8)
        
        # Check values (should be -1 or 1)
        unique_values = np.unique(emotion_signals)
        assert len(unique_values) == 2
        assert -1.0 in unique_values
        assert 1.0 in unique_values
    
    def test_generate_data(self):
        """Test synthetic data generation"""
        config = {
            'kind': 'data',
            'freq': 0.2,
            'dims': 2,
            'channels': 8
        }
        signal = EmotionSignal(config)
        
        # Generate signals
        t = np.arange(0, 1.0, 0.01)  # 1 second at 100 Hz
        emotion_signals, channel_signals = signal.generate(t)
        
        # Check shapes
        assert emotion_signals.shape == (len(t), 2)
        assert channel_signals.shape == (len(t), 8)
        
        # Check range
        assert np.all(emotion_signals >= -1.0) and np.all(emotion_signals <= 1.0)
    
    def test_add_artifacts(self):
        """Test adding artifacts to signals"""
        config = {
            'kind': 'sine',
            'freq': 0.2,
            'dims': 2,
            'channels': 8,
            'noise': {
                'gaussian': {
                    'enabled': True,
                    'std': 0.1
                },
                'powerline': {
                    'enabled': True,
                    'freq': 50,
                    'amplitude': 0.05
                }
            }
        }
        signal = EmotionSignal(config)
        
        # Generate clean signals
        t = np.arange(0, 1.0, 0.01)  # 1 second at 100 Hz
        clean_signals = np.zeros((len(t), 8))
        
        # Add artifacts
        noisy_signals = signal.add_artifacts(clean_signals)
        
        # Check that artifacts were added (signals should be different)
        assert not np.array_equal(clean_signals, noisy_signals)
    
    def test_map_to_channels(self):
        """Test mapping emotion signals to channels"""
        config = {
            'kind': 'sine',
            'freq': 0.2,
            'dims': 2,
            'channels': 8,
            'channel_amplitudes': [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]
        }
        signal = EmotionSignal(config)
        
        # Create test emotion signals
        t = np.arange(0, 1.0, 0.01)  # 1 second at 100 Hz
        emotion_signals = np.ones((len(t), 2))
        
        # Map to channels
        channel_signals = signal.map_to_channels(emotion_signals)
        
        # Check shape
        assert channel_signals.shape == (len(t), 8)
        
        # Check that channel amplitudes were applied
        # (This is approximate since there's randomness in the mapping)
        max_amplitudes = np.max(np.abs(channel_signals), axis=0)
        assert np.all(max_amplitudes > 0)  # All channels should have signal
    
    def test_load_from_file_csv(self):
        """Test loading signals from CSV file"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Create test data
            t = np.arange(0, 1.0, 0.01)  # 1 second at 100 Hz
            data = np.zeros((len(t), 3))
            data[:, 0] = np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz sine wave
            data[:, 1] = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine wave
            data[:, 2] = np.random.normal(0, 0.1, len(t))  # Noise
            
            # Save to CSV
            pd.DataFrame(data).to_csv(temp_path, index=False)
        
        try:
            # Create EmotionSignal with data loading
            config = {
                'kind': 'data',
                'dims': 2,
                'channels': 8,
                'data_loading': {
                    'enabled': True,
                    'file_path': temp_path,
                    'format': 'csv'
                }
            }
            signal = EmotionSignal(config)
            
            # Check that data was loaded
            assert signal.external_data is not None
            assert signal.external_data.shape[0] == len(t)
            assert signal.external_data.shape[1] >= 2
            
            # Generate signals
            emotion_signals, channel_signals = signal.generate(t)
            
            # Check shapes
            assert emotion_signals.shape == (len(t), 2)
            assert channel_signals.shape == (len(t), 8)
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_load_from_file_invalid_path(self):
        """Test loading signals from non-existent file"""
        config = {
            'kind': 'data',
            'dims': 2,
            'channels': 8,
            'data_loading': {
                'enabled': True,
                'file_path': '/path/to/nonexistent/file.csv',
                'format': 'csv'
            }
        }
        
        with pytest.raises(ValueError):
            EmotionSignal(config)
    
    def test_load_from_file_invalid_format(self):
        """Test loading signals with invalid format"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            config = {
                'kind': 'data',
                'dims': 2,
                'channels': 8,
                'data_loading': {
                    'enabled': True,
                    'file_path': temp_path,
                    'format': 'invalid'
                }
            }
            
            with pytest.raises(ValueError):
                EmotionSignal(config)
        finally:
            # Clean up
            os.unlink(temp_path)

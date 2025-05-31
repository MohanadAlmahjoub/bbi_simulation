#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for BBISimulation module
"""

import pytest
import numpy as np
import os
import tempfile
from bbi_simulation.simulation.bbi_simulation import BBISimulation


class TestBBISimulation:
    """Test suite for BBISimulation class"""
    
    def test_init(self):
        """Test initialization with default parameters"""
        # Create minimal config
        config = {
            'simulation': {
                'duration': 1.0,
                'dt': 0.01,
                'latency_jitter': 0.01
            },
            'emotion_signal': {
                'kind': 'sine',
                'freq': 0.2,
                'dims': 2,
                'channels': 4
            },
            'decoder': {
                'type': 'simple',
                'noise_std': 0.1,
                'window_size': 10
            },
            'stimulator': {
                'tau': 0.1,
                'threshold': 0.0
            }
        }
        
        # Create simulation
        sim = BBISimulation(config=config)
        
        # Check attributes
        assert sim.duration == 1.0
        assert sim.dt == 0.01
        assert sim.latency_jitter == 0.01
        assert sim.signal is not None
        assert sim.decoder is not None
        assert sim.stimulator is not None
    
    def test_run(self):
        """Test running simulation"""
        # Create minimal config with short duration for faster testing
        config = {
            'simulation': {
                'duration': 0.5,  # Short duration
                'dt': 0.01,
                'latency_jitter': 0.0  # No jitter for deterministic testing
            },
            'emotion_signal': {
                'kind': 'sine',
                'freq': 0.2,
                'dims': 2,
                'channels': 4
            },
            'decoder': {
                'type': 'simple',
                'noise_std': 0.0,  # No noise for deterministic testing
                'window_size': 10
            },
            'stimulator': {
                'tau': 0.1,
                'threshold': 0.0
            }
        }
        
        # Create simulation
        sim = BBISimulation(config=config)
        
        # Run simulation
        results = sim.run()
        
        # Check results
        assert 't' in results
        assert 'emotion_signals' in results
        assert 'channel_signals' in results
        assert 'decoded' in results
        assert 'response' in results
        assert 'processing_latencies' in results
        assert 'jitter_latencies' in results
        assert 'total_latencies' in results
        assert 'metrics' in results
        
        # Check shapes
        n_samples = len(results['t'])
        assert results['emotion_signals'].shape == (n_samples, 2)
        assert results['channel_signals'].shape == (n_samples, 4)
        assert results['decoded'].shape == (n_samples, 2)
        assert results['response'].shape == (n_samples, 2)
        assert results['processing_latencies'].shape == (n_samples,)
        assert results['jitter_latencies'].shape == (n_samples,)
        assert results['total_latencies'].shape == (n_samples,)
        
        # Check metrics
        metrics = results['metrics']
        assert 'decode_correlations' in metrics
        assert 'mean_decode_correlation' in metrics
        assert 'decode_rmses' in metrics
        assert 'mean_decode_rmse' in metrics
    
    def test_run_with_log_dir(self):
        """Test running simulation with log directory"""
        # Create minimal config with short duration for faster testing
        config = {
            'simulation': {
                'duration': 0.5,  # Short duration
                'dt': 0.01,
                'latency_jitter': 0.0  # No jitter for deterministic testing
            },
            'emotion_signal': {
                'kind': 'sine',
                'freq': 0.2,
                'dims': 2,
                'channels': 4
            },
            'decoder': {
                'type': 'simple',
                'noise_std': 0.0,  # No noise for deterministic testing
                'window_size': 10
            },
            'stimulator': {
                'tau': 0.1,
                'threshold': 0.0
            }
        }
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create simulation with log directory
            sim = BBISimulation(config=config, log_dir=temp_dir)
            
            # Run simulation
            results = sim.run()
            
            # Check that files were created
            assert os.path.exists(os.path.join(temp_dir, 'time_series.npz'))
            assert os.path.exists(os.path.join(temp_dir, 'metrics.csv'))
    
    def test_compute_metrics(self):
        """Test computing metrics"""
        # Create minimal config
        config = {
            'simulation': {
                'duration': 1.0,
                'dt': 0.01,
                'latency_jitter': 0.0
            },
            'emotion_signal': {
                'kind': 'sine',
                'freq': 0.2,
                'dims': 2,
                'channels': 4
            },
            'decoder': {
                'type': 'simple',
                'noise_std': 0.1,
                'window_size': 10
            },
            'stimulator': {
                'tau': 0.1,
                'threshold': 0.0
            }
        }
        
        # Create simulation
        sim = BBISimulation(config=config)
        
        # Create test signals
        t = np.arange(0, 1.0, 0.01)  # 1 second at 100 Hz
        emotion_signals = np.zeros((len(t), 2))
        emotion_signals[:, 0] = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine wave
        emotion_signals[:, 1] = np.cos(2 * np.pi * 0.5 * t)  # 0.5 Hz cosine wave
        
        decoded = np.zeros((len(t), 2))
        decoded[:, 0] = np.sin(2 * np.pi * 0.5 * t) + 0.1  # Slightly offset
        decoded[:, 1] = np.cos(2 * np.pi * 0.5 * t) - 0.1  # Slightly offset
        
        response = np.zeros((len(t), 2))
        response[:, 0] = np.sin(2 * np.pi * 0.5 * t + np.pi/8)  # Phase-shifted
        response[:, 1] = np.cos(2 * np.pi * 0.5 * t + np.pi/8)  # Phase-shifted
        
        # Compute metrics
        metrics = sim.compute_metrics(emotion_signals, decoded, response)
        
        # Check metrics
        assert 'decode_correlations' in metrics
        assert 'mean_decode_correlation' in metrics
        assert 'decode_rmses' in metrics
        assert 'mean_decode_rmse' in metrics
        assert 'correlations' in metrics
        assert 'mean_correlation' in metrics
        assert 'rmses' in metrics
        assert 'mean_rmse' in metrics
        assert 'delays' in metrics
        assert 'mean_delay' in metrics
        
        # Check dimensions
        assert len(metrics['decode_correlations']) == 2
        assert len(metrics['decode_rmses']) == 2
        assert len(metrics['correlations']) == 2
        assert len(metrics['rmses']) == 2
        assert len(metrics['delays']) == 2

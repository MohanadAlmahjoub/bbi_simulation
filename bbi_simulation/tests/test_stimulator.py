#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for Stimulator module
"""

import pytest
import numpy as np
from bbi_simulation.core.stimulator import Stimulator


class TestStimulator:
    """Test suite for Stimulator class"""
    
    def test_init(self):
        """Test initialization with default parameters"""
        config = {
            'tau': 0.1,
            'dt': 0.01,
            'threshold': 0.0
        }
        stimulator = Stimulator(config)
        
        assert stimulator.tau == 0.1
        assert stimulator.dt == 0.01
        assert stimulator.threshold == 0.0
        assert stimulator.kernel is not None
        assert len(stimulator.kernel) > 0
    
    def test_build_kernel(self):
        """Test kernel building"""
        config = {
            'tau': 0.1,
            'dt': 0.01
        }
        stimulator = Stimulator(config)
        
        # Check kernel properties
        assert len(stimulator.kernel) == int(5 * 0.1 / 0.01)  # 5*tau/dt
        assert stimulator.kernel[0] > stimulator.kernel[-1]  # Decreasing
        assert np.isclose(np.sum(stimulator.kernel), 1.0)  # Normalized
    
    def test_stimulate_1d(self):
        """Test stimulation with 1D input"""
        config = {
            'tau': 0.1,
            'dt': 0.01,
            'threshold': 0.0
        }
        stimulator = Stimulator(config)
        
        # Create test decoded series
        t = np.arange(0, 1.0, 0.01)  # 1 second at 100 Hz
        decoded_series = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine wave
        
        # Stimulate
        response = stimulator.stimulate(decoded_series)
        
        # Check shape
        assert response.shape == decoded_series.shape
        
        # Check that response is not identical to input (convolution changes it)
        assert not np.array_equal(decoded_series, response)
    
    def test_stimulate_2d(self):
        """Test stimulation with 2D input"""
        config = {
            'tau': 0.1,
            'dt': 0.01,
            'threshold': 0.0
        }
        stimulator = Stimulator(config)
        
        # Create test decoded series
        t = np.arange(0, 1.0, 0.01)  # 1 second at 100 Hz
        decoded_series = np.zeros((len(t), 2))
        decoded_series[:, 0] = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine wave
        decoded_series[:, 1] = np.cos(2 * np.pi * 0.5 * t)  # 0.5 Hz cosine wave
        
        # Stimulate
        response = stimulator.stimulate(decoded_series)
        
        # Check shape
        assert response.shape == decoded_series.shape
        
        # Check that response is not identical to input (convolution changes it)
        assert not np.array_equal(decoded_series, response)
    
    def test_stimulate_with_threshold(self):
        """Test stimulation with threshold"""
        config = {
            'tau': 0.1,
            'dt': 0.01,
            'threshold': 0.5
        }
        stimulator = Stimulator(config)
        
        # Create test decoded series
        t = np.arange(0, 1.0, 0.01)  # 1 second at 100 Hz
        decoded_series = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine wave
        
        # Stimulate
        response = stimulator.stimulate(decoded_series)
        
        # Check that values below threshold are zero
        below_threshold = decoded_series <= 0.5
        assert np.all(response[below_threshold] == 0)
    
    def test_update_parameters(self):
        """Test parameter updating"""
        config = {
            'tau': 0.1,
            'dt': 0.01,
            'threshold': 0.0
        }
        stimulator = Stimulator(config)
        
        # Get original kernel
        original_kernel = stimulator.kernel.copy()
        
        # Update parameters
        stimulator.update_parameters({
            'tau': 0.2,
            'dt': 0.02,
            'threshold': 0.5
        })
        
        # Check that parameters were updated
        assert stimulator.tau == 0.2
        assert stimulator.dt == 0.02
        assert stimulator.threshold == 0.5
        
        # Check that kernel was rebuilt
        assert len(stimulator.kernel) == int(5 * 0.2 / 0.02)  # 5*tau/dt
        assert not np.array_equal(original_kernel, stimulator.kernel)
    
    def test_compute_response_metrics_1d(self):
        """Test response metrics computation with 1D input"""
        config = {
            'tau': 0.1,
            'dt': 0.01
        }
        stimulator = Stimulator(config)
        
        # Create test signals
        t = np.arange(0, 1.0, 0.01)  # 1 second at 100 Hz
        original = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine wave
        response = np.sin(2 * np.pi * 0.5 * t + np.pi/4)  # Phase-shifted sine wave
        
        # Compute metrics
        metrics = stimulator.compute_response_metrics(original, response)
        
        # Check metrics
        assert 'correlation' in metrics
        assert 'rmse' in metrics
        assert 'delay' in metrics
        assert 'max_correlation' in metrics
        
        # Check correlation (should be high but not 1.0 due to phase shift)
        assert metrics['correlation'] > 0.5
        assert metrics['correlation'] < 1.0
    
    def test_compute_response_metrics_2d(self):
        """Test response metrics computation with 2D input"""
        config = {
            'tau': 0.1,
            'dt': 0.01
        }
        stimulator = Stimulator(config)
        
        # Create test signals
        t = np.arange(0, 1.0, 0.01)  # 1 second at 100 Hz
        original = np.zeros((len(t), 2))
        original[:, 0] = np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz sine wave
        original[:, 1] = np.cos(2 * np.pi * 0.5 * t)  # 0.5 Hz cosine wave
        
        response = np.zeros((len(t), 2))
        response[:, 0] = np.sin(2 * np.pi * 0.5 * t + np.pi/4)  # Phase-shifted sine
        response[:, 1] = np.cos(2 * np.pi * 0.5 * t + np.pi/4)  # Phase-shifted cosine
        
        # Compute metrics
        metrics = stimulator.compute_response_metrics(original, response)
        
        # Check metrics
        assert 'correlations' in metrics
        assert 'mean_correlation' in metrics
        assert 'rmses' in metrics
        assert 'mean_rmse' in metrics
        assert 'delays' in metrics
        assert 'mean_delay' in metrics
        assert 'max_correlations' in metrics
        assert 'mean_max_correlation' in metrics
        
        # Check dimensions
        assert len(metrics['correlations']) == 2
        assert len(metrics['rmses']) == 2
        assert len(metrics['delays']) == 2
        assert len(metrics['max_correlations']) == 2

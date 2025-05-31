#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration schema validation for Brain-to-Brain Interface Simulation.

This module provides schema validation for configuration parameters.
"""

import yaml
from typing import Dict, Any, List, Optional


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration parameters against schema.
    
    Args:
        config (Dict): Configuration dictionary
        
    Returns:
        List[str]: List of validation errors, empty if valid
    """
    errors = []
    
    # Check required sections
    required_sections = ['simulation', 'emotion_signal', 'decoder', 'stimulator']
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required configuration section: {section}")
    
    # If any required sections are missing, return early
    if errors:
        return errors
    
    # Validate simulation parameters
    sim_config = config['simulation']
    if 'duration' not in sim_config:
        errors.append("Missing required parameter: simulation.duration")
    elif not isinstance(sim_config['duration'], (int, float)) or sim_config['duration'] <= 0:
        errors.append(f"Invalid simulation.duration: {sim_config['duration']}. Must be a positive number.")
        
    if 'dt' not in sim_config:
        errors.append("Missing required parameter: simulation.dt")
    elif not isinstance(sim_config['dt'], (int, float)) or sim_config['dt'] <= 0:
        errors.append(f"Invalid simulation.dt: {sim_config['dt']}. Must be a positive number.")
    
    # Validate emotion_signal parameters
    es_config = config['emotion_signal']
    if 'kind' not in es_config:
        errors.append("Missing required parameter: emotion_signal.kind")
    elif es_config['kind'] not in ['sine', 'square', 'data']:
        errors.append(f"Invalid emotion_signal.kind: {es_config['kind']}. Must be one of 'sine', 'square', or 'data'.")
        
    if 'dims' not in es_config:
        errors.append("Missing required parameter: emotion_signal.dims")
    elif not isinstance(es_config['dims'], int) or es_config['dims'] <= 0:
        errors.append(f"Invalid emotion_signal.dims: {es_config['dims']}. Must be a positive integer.")
        
    if 'channels' not in es_config:
        errors.append("Missing required parameter: emotion_signal.channels")
    elif not isinstance(es_config['channels'], int) or es_config['channels'] <= 0:
        errors.append(f"Invalid emotion_signal.channels: {es_config['channels']}. Must be a positive integer.")
    
    # Validate decoder parameters
    dec_config = config['decoder']
    if 'type' not in dec_config:
        errors.append("Missing required parameter: decoder.type")
    elif dec_config['type'] not in ['simple', 'ml']:
        errors.append(f"Invalid decoder.type: {dec_config['type']}. Must be one of 'simple' or 'ml'.")
        
    if 'window_size' not in dec_config:
        errors.append("Missing required parameter: decoder.window_size")
    elif not isinstance(dec_config['window_size'], int) or dec_config['window_size'] <= 0:
        errors.append(f"Invalid decoder.window_size: {dec_config['window_size']}. Must be a positive integer.")
    
    # If decoder type is 'ml', validate ML parameters
    if dec_config.get('type') == 'ml':
        ml_config = dec_config.get('ml', {})
        
        if 'model_type' not in ml_config:
            errors.append("Missing required parameter: decoder.ml.model_type")
        elif ml_config['model_type'] not in ['svm', 'rf', 'mlp', 'auto']:
            errors.append(f"Invalid decoder.ml.model_type: {ml_config['model_type']}. Must be one of 'svm', 'rf', 'mlp', or 'auto'.")
    
    # Validate stimulator parameters
    stim_config = config['stimulator']
    if 'tau' not in stim_config:
        errors.append("Missing required parameter: stimulator.tau")
    elif not isinstance(stim_config['tau'], (int, float)) or stim_config['tau'] <= 0:
        errors.append(f"Invalid stimulator.tau: {stim_config['tau']}. Must be a positive number.")
    
    return errors


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the configuration.
    
    Args:
        config (Dict): Configuration dictionary
    """
    print("Configuration Summary:")
    print("======================")
    
    # Simulation parameters
    sim_config = config.get('simulation', {})
    print("\nSimulation:")
    print(f"  Duration: {sim_config.get('duration', 'Not specified')} s")
    print(f"  Time step: {sim_config.get('dt', 'Not specified')} s")
    print(f"  Latency jitter: {sim_config.get('latency_jitter', 'Not specified')} s")
    
    # EmotionSignal parameters
    es_config = config.get('emotion_signal', {})
    print("\nEmotionSignal:")
    print(f"  Kind: {es_config.get('kind', 'Not specified')}")
    print(f"  Frequency: {es_config.get('freq', 'Not specified')} Hz")
    print(f"  Dimensions: {es_config.get('dims', 'Not specified')}")
    print(f"  Channels: {es_config.get('channels', 'Not specified')}")
    
    # Decoder parameters
    dec_config = config.get('decoder', {})
    print("\nDecoder:")
    print(f"  Type: {dec_config.get('type', 'Not specified')}")
    print(f"  Window size: {dec_config.get('window_size', 'Not specified')} samples")
    
    if dec_config.get('type') == 'ml':
        ml_config = dec_config.get('ml', {})
        print(f"  ML model type: {ml_config.get('model_type', 'Not specified')}")
        print(f"  CV folds: {ml_config.get('cv_folds', 'Not specified')}")
    
    # Stimulator parameters
    stim_config = config.get('stimulator', {})
    print("\nStimulator:")
    print(f"  Tau: {stim_config.get('tau', 'Not specified')} s")
    print(f"  Threshold: {stim_config.get('threshold', 'Not specified')}")
    
    # Analysis parameters
    analysis_config = config.get('analysis', {})
    sweep_config = analysis_config.get('parameter_sweep', {})
    print("\nParameter Sweep:")
    print(f"  Enabled: {sweep_config.get('enabled', False)}")
    if sweep_config.get('enabled', False):
        print(f"  Output CSV: {sweep_config.get('output_csv', 'Not specified')}")
        print("  Parameters to sweep:")
        params = sweep_config.get('params', {})
        for param, values in params.items():
            print(f"    {param}: {values}")


# Example usage
if __name__ == "__main__":
    import os
    
    # Load default configuration
    default_config_path = os.path.join(
        os.path.dirname(__file__), 
        'default_config.yaml'
    )
    
    with open(default_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate configuration
    errors = validate_config(config)
    
    if errors:
        print("Configuration validation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid.")
        print_config_summary(config)

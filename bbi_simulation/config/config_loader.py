#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration loader for Brain-to-Brain Interface Simulation.

This module provides utilities for loading and validating configuration parameters
from YAML files.
"""

import os
import yaml
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    A class for loading and validating configuration parameters.
    
    The ConfigLoader class loads configuration parameters from YAML files,
    applies environment variable overrides, and validates the configuration.
    
    Attributes:
        config (Dict): The loaded and validated configuration
        config_path (str): Path to the configuration file
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the ConfigLoader.
        
        Args:
            config_path (str, optional): Path to the configuration file.
                If None, the default configuration is used.
        """
        self.config_path = config_path
        self.config = {}
        
        # Load configuration
        self._load_config()
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate configuration
        self._validate_config()
    
    def _load_config(self) -> None:
        """
        Load configuration from file.
        
        If config_path is None, the default configuration is used.
        """
        if self.config_path is None:
            # Use default configuration
            default_config_path = os.path.join(
                os.path.dirname(__file__), 
                'default_config.yaml'
            )
            with open(default_config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            # Load configuration from specified file
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
    
    def _apply_env_overrides(self) -> None:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables with the prefix 'BBI_' are used to override
        configuration parameters. The format is:
        
        BBI_SECTION_SUBSECTION_PARAMETER=value
        
        For example:
        BBI_EMOTION_SIGNAL_FREQ=0.5
        """
        for env_var, value in os.environ.items():
            if env_var.startswith('BBI_'):
                # Remove prefix and split into parts
                parts = env_var[4:].lower().split('_')
                
                # Navigate to the correct section in the config
                config_section = self.config
                for part in parts[:-1]:
                    if part not in config_section:
                        config_section[part] = {}
                    config_section = config_section[part]
                
                # Set the parameter value
                param_name = parts[-1]
                
                # Try to convert value to appropriate type
                try:
                    # Try as int
                    config_section[param_name] = int(value)
                except ValueError:
                    try:
                        # Try as float
                        config_section[param_name] = float(value)
                    except ValueError:
                        # Try as boolean
                        if value.lower() in ('true', 'yes', '1'):
                            config_section[param_name] = True
                        elif value.lower() in ('false', 'no', '0'):
                            config_section[param_name] = False
                        else:
                            # Use as string
                            config_section[param_name] = value
    
    def _validate_config(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required sections
        required_sections = ['simulation', 'emotion_signal', 'decoder', 'stimulator']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate simulation parameters
        sim_config = self.config['simulation']
        if 'duration' not in sim_config:
            raise ValueError("Missing required parameter: simulation.duration")
        if 'dt' not in sim_config:
            raise ValueError("Missing required parameter: simulation.dt")
        
        # Validate emotion_signal parameters
        es_config = self.config['emotion_signal']
        if 'kind' not in es_config:
            raise ValueError("Missing required parameter: emotion_signal.kind")
        if es_config['kind'] not in ['sine', 'square', 'data']:
            raise ValueError(f"Invalid emotion_signal.kind: {es_config['kind']}")
        if 'dims' not in es_config:
            raise ValueError("Missing required parameter: emotion_signal.dims")
        if 'channels' not in es_config:
            raise ValueError("Missing required parameter: emotion_signal.channels")
        
        # Validate decoder parameters
        dec_config = self.config['decoder']
        if 'type' not in dec_config:
            raise ValueError("Missing required parameter: decoder.type")
        if dec_config['type'] not in ['simple', 'ml']:
            raise ValueError(f"Invalid decoder.type: {dec_config['type']}")
        if 'window_size' not in dec_config:
            raise ValueError("Missing required parameter: decoder.window_size")
        
        # Validate stimulator parameters
        stim_config = self.config['stimulator']
        if 'tau' not in stim_config:
            raise ValueError("Missing required parameter: stimulator.tau")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the loaded and validated configuration.
        
        Returns:
            Dict: The configuration dictionary
        """
        return self.config
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific section of the configuration.
        
        Args:
            section (str): The section name
            
        Returns:
            Dict: The section dictionary
            
        Raises:
            KeyError: If the section does not exist
        """
        if section not in self.config:
            raise KeyError(f"Configuration section not found: {section}")
        
        return self.config[section]


# Example usage
if __name__ == "__main__":
    # Load default configuration
    config_loader = ConfigLoader()
    config = config_loader.get_config()
    
    # Print configuration
    print(yaml.dump(config))

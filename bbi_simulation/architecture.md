# Brain-to-Brain Interface Simulation Framework Architecture

## Overview

This document outlines the enhanced architecture for the Brain-to-Brain Interface (BBI) Simulation Framework, designed to support all requested features and extensibility requirements.

## Core Components

### 1. Configuration Management

```
config/
├── config_loader.py     # YAML configuration loader with validation
├── default_config.yaml  # Default configuration parameters
└── schema.py           # Configuration schema validation
```

- **ConfigLoader**: Central class for loading, validating, and accessing configuration parameters
- **Configuration Schema**: Defines required and optional parameters with types and ranges
- **Environment Variables**: Support for overriding config via environment variables

### 2. Enhanced Core Modules

#### 2.1 EmotionSignal Module

```python
class EmotionSignal:
    def __init__(self, config):
        # Initialize based on configuration
        
    def generate(self, t_array):
        # Generate signals based on configuration
        
    def load_from_file(self, file_path, format='csv'):
        # Load signals from file (CSV or EDF)
        
    def add_artifacts(self, signals):
        # Add power-line artifacts and other noise
```

- Support for multi-channel waveforms
- Configurable Gaussian noise
- 50/60 Hz power-line artifacts
- Variable amplitudes per channel
- File loading for real/prerecorded EEG (CSV/EDF)

#### 2.2 Decoder Module

```python
class Decoder:
    def __init__(self, config):
        # Initialize based on configuration
        
    def decode(self, eeg_chunk):
        # Decode EEG chunk
        
    def train(self, X, y):
        # Train multiple models and select best
        
    def save_model(self, path):
        # Save trained model
        
    def load_model(self, path):
        # Load trained model
```

- Support for multiple ML models (SVM, Random Forest, MLP)
- Model training with cross-validation
- Model selection based on F1-score
- Model persistence

#### 2.3 Stimulator Module

```python
class Stimulator:
    def __init__(self, config):
        # Initialize based on configuration
        
    def stimulate(self, decoded_series):
        # Stimulate based on decoded series
        
    def update_parameters(self, params):
        # Update stimulator parameters
```

- Enhanced stimulation response modeling
- Configurable thresholds
- Additional response metrics

### 3. Simulation Framework

```python
class BBISimulation:
    def __init__(self, config):
        # Initialize based on configuration
        
    def run(self):
        # Run simulation
        
    def compute_metrics(self, raw, decoded, response):
        # Compute performance metrics
```

- End-to-end simulation
- Real-time or pseudo-real-time simulation
- Comprehensive logging
- Performance metrics

### 4. Analysis Tools

```python
class ParameterSweep:
    def __init__(self, config):
        # Initialize based on configuration
        
    def run(self):
        # Run parameter sweep
        
    def save_results(self, path):
        # Save results to CSV
        
    def plot_results(self):
        # Generate summary plots
```

- Parameter grid generation
- Multiple simulation runs
- CSV output
- Summary plot generation

### 5. Utilities

```
utils/
├── data_loader.py    # Data loading utilities
├── metrics.py        # Performance metrics
├── plotting.py       # Plotting utilities
└── file_io.py        # File I/O utilities
```

- Data loading utilities
- Performance metrics
- Plotting utilities
- File I/O utilities

## Package Structure

```
bbi_simulation/
├── __init__.py
├── config/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── default_config.yaml
│   └── schema.py
├── core/
│   ├── __init__.py
│   ├── emotion_signal.py
│   ├── decoder.py
│   └── stimulator.py
├── simulation/
│   ├── __init__.py
│   └── bbi_simulation.py
├── analysis/
│   ├── __init__.py
│   └── parameter_sweep.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── metrics.py
│   ├── plotting.py
│   └── file_io.py
├── tests/
│   ├── __init__.py
│   ├── test_emotion_signal.py
│   ├── test_decoder.py
│   ├── test_stimulator.py
│   └── test_simulation.py
├── examples/
│   ├── basic_simulation.py
│   ├── parameter_sweep.py
│   └── demo.ipynb
├── main.py
├── setup.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Dependencies

- **Core Dependencies**:
  - numpy
  - scipy
  - matplotlib
  - pyyaml
  - scikit-learn
  - pandas

- **EEG Data Loading**:
  - mne (for EDF support)

- **Testing**:
  - pytest

- **Packaging**:
  - setuptools
  - wheel

- **Documentation**:
  - sphinx

## Interface Diagrams

### Configuration Flow

```
YAML Config File → ConfigLoader → Validated Config → Module Initialization
```

### Simulation Flow

```
EmotionSignal.generate() → Decoder.decode() → Stimulator.stimulate() → Metrics Computation
```

### Parameter Sweep Flow

```
Parameter Grid → Multiple BBISimulation Runs → Results Collection → CSV Output → Summary Plots
```

## Extension Points

1. **Custom Signal Generators**: Extend EmotionSignal or implement custom generators
2. **Custom Decoders**: Extend Decoder or implement custom decoders
3. **Custom Stimulators**: Extend Stimulator or implement custom stimulators
4. **Custom Metrics**: Add custom performance metrics
5. **Custom Plotting**: Add custom plotting functions

This architecture provides a flexible, modular framework that can be extended to support additional features and use cases while maintaining a clean, well-documented codebase suitable for publication.

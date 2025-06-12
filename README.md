# Brain-to-Brain Interface Simulation Framework

[![PyPI version](https://badge.fury.io/py/bbi-simulation.svg)](https://badge.fury.io/py/bbi-simulation)
[![Documentation Status](https://readthedocs.io/en/latest/?badge=latest)](https://bbi-simulation.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://github.com/MohanadAlmahjoub/bbi_simulation/workflows/CI/badge.svg)](https://github.com/MohanadAlmahjoub/bbi_simulation/actions)
[![Coverage Status](https://coveralls.io/repos/github/MohanadAlmahjoub/bbi_simulation/badge.svg?branch=main)](https://coveralls.io/github/MohanadAlmahjoub/bbi_simulation?branch=main)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Last updated:** May 31, 2025

A comprehensive Python framework for simulating closed-loop, real-time affective brain-to-brain interfaces.

<p align="center">
  <img src="https://raw.githubusercontent.com/MohanadAlmahjoub/bbi_simulation/main/docs/images/bbi_simulation_overview.png" alt="BBI Simulation Overview" width="600"/>
</p>

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
   - [Prerequisites](#prerequisites)
   - [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Quick Start](#quick-start)
   - [Basic Simulation](#basic-simulation)
   - [Command-Line Interface](#command-line-interface)
5. [Examples](#examples)
   - [Valence-Arousal Trajectory](#valence-arousal-trajectory)
6. [Directory Structure](#directory-structure)
7. [Testing](#testing)
8. [Interactive Demo](#interactive-demo)
9. [Documentation](#documentation)
10. [Roadmap](#roadmap)
11. [FAQ & Troubleshooting](#faq--troubleshooting)
    - [Known Issues](#known-issues)
12. [Contributing](#contributing)
13. [Changelog](#changelog)
14. [Citation](#citation)
15. [License](#license)
16. [Author and Contact](#author-and-contact)

## Overview

This framework provides a comprehensive set of tools for simulating affective brain-to-brain interfaces, focusing on real-time emotion transfer and closed-loop control. The platform enables researchers to model, test, and validate BBI protocols before implementation in physical hardware.

Key features include:

- **Complete Simulation Pipeline**: Emotion generation, EEG-style decoding, neurostimulation response, and closed-loop control
- **Realistic EEG Signal Generation**: Configurable waveforms, noise, and artifacts
- **Multiple ML Models**: Support for SVM, Random Forest, and MLP decoders with auto-selection
- **Parameter Sweep Utility**: Sensitivity analysis across parameter combinations
- **Publication-Ready Outputs**: High-quality figures and tables for manuscripts
- **Extensible Plugin Architecture**: Easily add custom components
- **Comprehensive Documentation**: Detailed API docs and examples

## Installation

### Prerequisites

- Python 3.9 or higher
- Git (for repository cloning)
- pip (20.0+)
- Operating system: Linux, macOS, or Windows

### Installation Steps

For basic installation from PyPI:

```bash
pip install bbi-simulation
```

For development installation:

```bash
# Clone the repository
git clone https://github.com/MohanadAlmahjoub/bbi_simulation.git
cd bbi_simulation

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Linux/macOS
# or
venv\Scripts\activate     # On Windows

# Install in development mode with extra dependencies
pip install -e ".[dev,edf,notebook]"
```

To verify installation:

```bash
bbi-simulation info
```

## Configuration

All simulation settings are loaded from YAML configuration files. The default configuration file is located at `bbi_simulation/config/default_config.yaml`. Here's an example:

```yaml
simulation:
  duration: 10.0         # Simulation duration in seconds
  dt: 0.01               # Time step (seconds)
  latency_jitter: 0.01   # Random latency variation (seconds)

emotion_signal:
  kind: "sine"           # Signal type: sine, square, synthetic, data
  freq: 0.2              # Base frequency (Hz)
  dims: 2                # Emotion dimensions (e.g., valence, arousal)
  channels: 8            # Number of simulated EEG channels
  channel_names:         # Channel names (optional)
    - "Fp1"
    - "Fp2"
    - "F3"
    - "F4"
    - "C3"
    - "C4"
    - "P3"
    - "P4"
  noise:
    gaussian:
      enabled: true
      std: 0.1
    powerline:
      enabled: true
      freq: 50
      amplitude: 0.05

decoder:
  type: "ml"             # Decoder type: "avg" or "ml"
  window_size: 50        # Time window size for decoder (samples)
  ml:
    model_type: "auto"   # ML types: "auto", "svm", "rf", "mlp"
    cv_folds: 5          # Cross-validation folds

stimulator:
  tau: 0.1               # Time constant (seconds) for response
  threshold: 0.0         # Threshold for response activation

analysis:
  sweep:
    parameters:
      - name: "freq"
        values: [0.1, 0.2, 0.5]
      - name: "noise.gaussian.std"
        values: [0.05, 0.1, 0.2]
    repetitions: 5       # Number of repetitions per experiment
```

You can copy the default configuration and modify it for your needs, then pass the path to your custom configuration:

```bash
bbi-simulation simulate --config my_custom_config.yaml --output results/
```

## Quick Start

### Basic Simulation

Run a simple simulation with default parameters:

```python
from bbi_simulation.config.schema import ConfigManager
from bbi_simulation.simulation.bbi_simulation import BBISimulation
import matplotlib.pyplot as plt
import numpy as np

# Load default configuration
config_manager = ConfigManager()
config = config_manager.get_config()

# Create and run simulation
sim = BBISimulation(config=config)
results = sim.run()

# Extract results
t = results['t']
emotion_signals = results['emotion_signals']
decoded = results['decoded']
response = results['response']

# Plot results
plt.figure(figsize=(10, 8))

# Plot emotion signals
plt.subplot(3, 1, 1)
for d in range(emotion_signals.shape[1]):
    plt.plot(t, emotion_signals[:, d], label=f"Dim {d+1}")
plt.title("Raw Emotion Signals")
plt.legend()

# Plot decoded signals
plt.subplot(3, 1, 2)
for d in range(decoded.shape[1]):
    plt.plot(t, decoded[:, d], label=f"Dim {d+1}")
plt.title("Decoded Signals")
plt.legend()

# Plot response signals
plt.subplot(3, 1, 3)
for d in range(response.shape[1]):
    plt.plot(t, response[:, d], label=f"Dim {d+1}")
plt.title("Stimulation Responses")
plt.xlabel("Time (s)")
plt.legend()

plt.tight_layout()
plt.show()
```

### Command-Line Interface

The package provides a convenient command-line interface:

```bash
# Run a quick simulation
bbi-simulation simulate --duration 5.0 --signal-type sine

# Run a parameter sweep
bbi-simulation sweep --param "tau=0.05,0.1,0.2" --param "window_size=25,50"

# Train a decoder model
bbi-simulation train --data training_data.npz --model-type auto

# Display information
bbi-simulation info
```

## Examples

### Valence-Arousal Trajectory

This example demonstrates how to generate and visualize a valence-arousal trajectory:

```python
from bbi_simulation.core.emotion_signal import EmotionSignal
import numpy as np
import matplotlib.pyplot as plt

# Create signal generator
config = {
    'kind': 'sine',
    'freq': 0.2,
    'dims': 2,
    'channels': 8
}
signal_gen = EmotionSignal(config)

# Generate signals
t = np.arange(0, 10.0, 0.01)  # 10 seconds at 100 Hz
emotion_signals, _ = signal_gen.generate(t)

# Extract valence and arousal
valence = emotion_signals[:, 0]
arousal = emotion_signals[:, 1]

# Plot valence-arousal trajectory
plt.figure(figsize=(8, 8))
plt.plot(valence, arousal, 'b-', alpha=0.7)
plt.plot(valence[0], arousal[0], 'go', label='Start')
plt.plot(valence[-1], arousal[-1], 'ro', label='End')
plt.grid(True)
plt.xlabel('Valence')
plt.ylabel('Arousal')
plt.title('Valence-Arousal Trajectory')
plt.axis('equal')
plt.legend()
plt.show()
```

<p align="center">
  <img src="https://raw.githubusercontent.com/MohanadAlmahjoub/bbi_simulation/main/docs/images/valence_arousal_trajectory.png" alt="Valence-Arousal Trajectory" width="400"/>
</p>

For more examples, see the `examples/` directory:
- `basic_simulation.py`: Simple simulation with default parameters
- `parameter_sweep.py`: Sensitivity analysis across parameter combinations
- `demo.ipynb`: Interactive Jupyter notebook demonstration

## Directory Structure

```
bbi_simulation/
├── bbi_simulation/
│   ├── __init__.py
│   ├── config/
│   │   └── default_config.yaml
│   ├── core/
│   │   ├── emotion_signal.py
│   │   ├── decoder.py
│   │   └── stimulator.py
│   ├── simulation/
│   │   └── bbi_simulation.py
│   ├── analysis/
│   │   └── parameter_sweep.py
│   └── utils.py
├── examples/
│   ├── basic_simulation.py
│   ├── parameter_sweep.py
│   ├── demo.ipynb
│   ├── colab_demo.ipynb
│   └── binder_demo.ipynb
├── tests/
│   ├── test_emotion_signal.py
│   ├── test_decoder.py
│   ├── test_stimulator.py
│   ├── test_bbi_simulation.py
│   └── test_edge_cases.py
├── docs/
│   ├── images/
│   │   ├── bbi_simulation_overview.png
│   │   └── valence_arousal_trajectory.png
│   └── (Sphinx documentation files)
├── .github/
│   └── ISSUE_TEMPLATE/
│       └── bug_report.md
├── .gitignore
├── LICENSE
├── README.md
├── CHANGELOG.md
└── pyproject.toml
```

## Testing

To run the test suite:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=bbi_simulation

# Run specific test file
pytest tests/test_emotion_signal.py
```

All core functionality is covered by unit tests to ensure reliability and correctness.

## Interactive Demo

Try the interactive demo in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MohanadAlmahjoub/bbi_simulation/blob/main/examples/colab_demo.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MohanadAlmahjoub/bbi_simulation/main?filepath=examples%2Fdemo.ipynb)

These interactive notebooks allow you to experiment with the framework without installing anything locally.

## Documentation

Full documentation is available at [https://bbi-simulation.readthedocs.io](https://bbi-simulation.readthedocs.io)

The documentation includes:
- API reference
- Tutorials
- Example gallery
- Theory and background

To build the documentation locally:

```bash
cd docs
pip install -r requirements.txt
make html
```

Then open `docs/_build/html/index.html` in your browser.

## Roadmap

Future development plans include:

- **Q3 2025**: Integration with real EEG hardware and TMS devices
- **Q4 2025**: Deep learning models for emotion decoding (LSTM, Transformers)
- **Q1 2026**: Initial human pilot studies and open data sharing
- **Q2 2026**: Interactive web interface for real-time visualization

## FAQ & Troubleshooting

**Q: I'm getting import errors with MNE after installation**
- Make sure you've installed the EDF extras: `pip install -e ".[edf]"`
- Some systems require additional libraries like libxml2. See the MNE documentation for details.

**Q: The emotion signals don't match my expectations**
- Check your configuration file to ensure `dims` and `channels` match your data
- Verify the signal type (`kind`) is appropriate for your use case

**Q: How do I save simulation results?**
- Use the `--output` flag with the CLI or the `log_dir` parameter with the API
- Results are saved in NPZ format for data and PNG/PDF for figures

### Known Issues

- **Performance on large datasets**: When using datasets larger than 1GB, memory usage can spike significantly. We recommend using the streaming data loader for large files.
- **Windows compatibility**: Some visualization features may render differently on Windows. We're working to standardize the appearance across platforms.
- **Matplotlib backend issues**: In some environments, you may need to explicitly set the Matplotlib backend with `matplotlib.use('Agg')` before importing pyplot.

## Contributing

We welcome contributions! To contribute:

1. Open an Issue describing the problem or feature
2. Fork the repository
3. Create a new branch for your feature
4. Add your changes, ensuring tests pass
5. Submit a Pull Request

Please follow the existing code style (Black + Flake8) and add appropriate tests for new features.

For more details, see [CONTRIBUTING.md](issue.md).

## Changelog

### v0.1.0 (2025-05-30)
- Initial release
- Core simulation components
- Basic ML decoders
- Parameter sweep functionality
- Documentation and examples

For detailed release notes, see [CHANGELOG.md](issue.md).

## Citation

If you use this software in your research, please cite:

```bibtex
@article{Almahjoub2025,
  title={Real-Time Affective Co-Regulation via Closed-Loop EEG–TMS Brain-to-Brain Interfaces: Technical Design and Clinical Ethics},
  author={Almahjoub, Mohanad},
  journal={},
  year={2025},
  doi={10.xxxx/xxxxx}
}
```

Or for the software package:

```bibtex
@software{bbi_simulation2025,
  author = {Almahjoub, Mohanad},
  title = {Brain-to-Brain Interface Simulation Framework},
  year = {2025},
  url = {https://github.com/MohanadAlmahjoub/bbi_simulation},
  version = {0.1.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author and Contact

- **Author**: MOHANAD ALMAHJOUB
- **Email**: mohanad.almahjoub@std.ankaramedipol.edu.tr
- **GitHub**: [https://github.com/MohanadAlmahjoub/bbi_simulation](https://github.com/MohanadAlmahjoub/bbi_simulation)

For questions and inquiries, please open an [Issue](https://github.com/MohanadAlmahjoub/bbi_simulation/issues) on GitHub or contact via the email address above.

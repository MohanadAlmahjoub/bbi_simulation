#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for Brain-to-Brain Interface Simulation package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bbi_simulation",
    version="0.1.0",
    author="MOHANAD ALMAHJOUB",
    author_email="mohanad.almahjoub@std.ankaramedipol.edu.tr",
    description="A simulation framework for closed-loop, real-time affective brain-to-brain interfaces",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bbi_simulation",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.9, <3.11",
    install_requires=[
        "numpy>=1.20.0,<1.25.0",
        "scipy>=1.7.0,<1.11.0",
        "matplotlib>=3.4.0,<3.8.0",
        "scikit-learn>=1.0.0,<1.3.0",
        "pandas>=1.3.0,<2.1.0",
        "pyyaml>=6.0,<7.0",
        "seaborn>=0.11.0,<0.13.0",
        "pytest>=7.0.0,<8.0.0",
    ],
    extras_require={
        "edf": ["mne>=1.0.0,<1.5.0"],
        "dev": [
            "pytest>=7.0.0,<8.0.0",
            "pytest-cov>=4.0.0,<5.0.0",
            "black>=23.0.0,<24.0.0",
            "isort>=5.10.0,<6.0.0",
            "flake8>=6.0.0,<7.0.0",
        ],
        "notebook": ["jupyter>=1.0.0,<2.0.0", "ipywidgets>=8.0.0,<9.0.0"],
    },
    entry_points={
        "console_scripts": [
            "bbi-simulate=bbi_simulation.main:main",
            "bbi-sweep=bbi_simulation.main:run_sweep",
        ],
    },
    include_package_data=True,
    package_data={
        "bbi_simulation": ["config/*.yaml"],
    },
)

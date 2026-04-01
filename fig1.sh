#!/bin/bash
set -euo pipefail

n_repeats=10
methods="U,U+U,U+GL"

# Run recovery mode
echo "Starting recovery mode with $n_repeats repeats..."
PYTHONPATH=src python -m gl_lowpopart.experiments.fig1 --mode recovery --num_repeats "$n_repeats" --methods "$methods"

# Run completion mode
echo "Starting completion mode with $n_repeats repeats..."
PYTHONPATH=src python -m gl_lowpopart.experiments.fig1 --mode completion --num_repeats "$n_repeats" --methods "$methods"

# Run hard mode
echo "Starting hard mode with $n_repeats repeats..."
PYTHONPATH=src python -m gl_lowpopart.experiments.fig1 --mode hard --num_repeats "$n_repeats" --methods "$methods"

echo "Plotting results..."
PYTHONPATH=src python -m gl_lowpopart.plotting.fig1

echo "All experiments completed!"

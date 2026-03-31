#!/bin/bash
n_repeats=10

# echo "Starting hard mode with $n_repeats repeats..."
# PYTHONPATH=src python -m gl_lowpopart.experiments.fig1 --mode hard --num_repeats $n_repeats

# Run recovery mode
echo "Starting recovery mode with $n_repeats repeats..."
PYTHONPATH=src python -m gl_lowpopart.experiments.fig1 --mode recovery --num_repeats $n_repeats --methods U,U+U

# Run completion mode
echo "Starting completion mode with $n_repeats repeats..."
PYTHONPATH=src python -m gl_lowpopart.experiments.fig1 --mode completion --num_repeats $n_repeats --methods U,U+U

echo "Plotting results..."
PYTHONPATH=src python -m gl_lowpopart.plotting.fig1

echo "All experiments completed!"

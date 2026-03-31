#!/bin/bash

# Run recovery mode
echo "Starting recovery mode with 60 repeats..."
PYTHONPATH=src python -m gl_lowpopart.experiments.fig1 --mode recovery --num_repeats 60

# Run completion mode
echo "Starting completion mode with 60 repeats..."
PYTHONPATH=src python -m gl_lowpopart.experiments.fig1 --mode completion --num_repeats 60

echo "All experiments completed!"

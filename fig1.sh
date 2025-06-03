#!/bin/bash

# Run recovery mode
echo "Starting recovery mode with 60 repeats..."
python fig1.py --mode recovery --num_repeats 60

# Run completion mode
echo "Starting completion mode with 60 repeats..."
python fig1.py --mode completion --num_repeats 60

echo "All experiments completed!"

#!/bin/bash

# Activate uv environment if needed
# source .venv/bin/activate

# Run recovery experiment
PYTHONPATH=src python -m gl_lowpopart.experiments.fig2 --mode recovery

# Run completion experiment
PYTHONPATH=src python -m gl_lowpopart.experiments.fig2 --mode completion

# Optional: Combine the plots into a single figure
# You can use ImageMagick if installed:
# convert +append completion_results.png recovery_results.png fig2_combined.png 
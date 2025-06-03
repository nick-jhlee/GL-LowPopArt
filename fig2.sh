#!/bin/bash

# Activate conda environment if needed
# source activate your_env_name

# Run recovery experiment
python fig2.py --mode recovery

# Run completion experiment
python fig2.py --mode completion

# Optional: Combine the plots into a single figure
# You can use ImageMagick if installed:
# convert +append completion_results.png recovery_results.png fig2_combined.png 
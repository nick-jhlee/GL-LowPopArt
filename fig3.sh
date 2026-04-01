#!/bin/bash
set -euo pipefail

n_repeats=30
n_total=100000
n1_values="5000,10000,20000,30000,40000,50000,70000,90000"

# Run recovery mode
echo "Starting fig3 recovery mode with $n_repeats repeats (N_total=$n_total)..."
PYTHONPATH=src python -m gl_lowpopart.experiments.fig3 \
  --mode recovery \
  --num_repeats "$n_repeats" \
  --N_total "$n_total" \
  --N1_values "$n1_values"

# Run completion mode
echo "Starting fig3 completion mode with $n_repeats repeats (N_total=$n_total)..."
PYTHONPATH=src python -m gl_lowpopart.experiments.fig3 \
  --mode completion \
  --num_repeats "$n_repeats" \
  --N_total "$n_total" \
  --N1_values "$n1_values"

echo "Plotting fig3 results..."
PYTHONPATH=src python -m gl_lowpopart.plotting.fig3

echo "Fig3 experiments completed!"
#!/bin/bash
set -euo pipefail

n_repeats=10

# Run recovery mode
echo "Starting recovery mode with $n_repeats repeats..."
PYTHONPATH=src python -m gl_lowpopart.experiments.fig3 --mode recovery --num_repeats "$n_repeats"

# Run completion mode
echo "Starting completion mode with $n_repeats repeats..."
PYTHONPATH=src python -m gl_lowpopart.experiments.fig3 --mode completion --num_repeats "$n_repeats"

# Run hard mode
echo "Starting hard mode with $n_repeats repeats..."
PYTHONPATH=src python -m gl_lowpopart.experiments.fig3 --mode hard --num_repeats "$n_repeats"

echo "Plotting results..."
PYTHONPATH=src python -m gl_lowpopart.plotting.fig3

echo "All experiments completed!"

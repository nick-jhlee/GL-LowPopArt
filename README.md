Codes for the following paper: [_GL-LowPopArt: A Nearly Instance-Wise Minimax-Optimal Estimator for (Adaptive) Generalized Low-Rank Trace Regression_](https://openreview.net/forum?id=TyArXyYnvz).

# Before Running
You require a license of [Mosek](https://www.mosek.com) to run the codes.

# Install
Clone the repository and first run
```shell
$ conda env create -f environment.yml
```
to create a conda environment.

# Reproducing Figure 1
Run
```shell
$ bash fig1.sh && python fig1_plot.py
```

# Reproducing Figure 2
Run
```shell
$ bash fig2.sh && python fig2_plot.py
```

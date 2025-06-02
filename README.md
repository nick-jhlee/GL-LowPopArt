Codes for the following paper by [Junghyun Lee](https://nick-jhlee.github.io/), [Kyoungseok Jang](https://jajajang.github.io), [Kwang-Sung Jun](https://kwangsungjun.github.io), [Milan Vojnović](https://personal.lse.ac.uk/vojnovic/), and [Se-Young Yun](https://fbsqkd.github.io/):
- [_GL-LowPopArt: A Nearly Instance-Wise Minimax-Optimal Estimator for (Adaptive) Generalized Low-Rank Trace Regression_](https://openreview.net/forum?id=TyArXyYnvz) (ICML 2025 *Spotlight*),

If you plan to use this repository or cite our paper, please use the following bibtex format:

```latex
@InProceedings{lee2025gl-lowpopart,
  title = 	 {{GL-LowPopArt: A Nearly Instance-Wise Minimax-Optimal Estimator for (Adaptive) Generalized Low-Rank Trace Regression}},
  author =       {Lee, Junghyun and Jang, Kyoungseok and Jun, Kwang-Sung and Vojnovi\`{c}, Milan and Yun, Se-Young},
  booktitle = 	 {Proceedings of The 42nd International Conference on Machine Learning},
  pages = 	 {},
  year = 	 {2025},
  editor = 	 {},
  volume = 	 {},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--19 July},
  publisher =    {PMLR},
  pdf = 	 {},
  url = 	 {https://openreview.net/forum?id=TyArXyYnvz},
}
```

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
$ bash fig1.sh
```

# Reproducing Figure 2
Run
```shell
$ bash fig2.sh
```

# Reproducing Figure 3
Run
```shell
$ bash fig3.sh
```
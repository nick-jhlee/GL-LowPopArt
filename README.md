Codes for [_GL-LowPopArt: A Nearly Instance-Wise Minimax-Optimal Estimator for Generalized Low-Rank Trace Regression_](https://openreview.net/forum?id=mMxiIP3Gxu) (AISTATS 2026) by [Junghyun Lee](https://nick-jhlee.github.io/), [Kyoungseok Jang](https://jajajang.github.io/), [Kwang-Sung Jun](https://kwangsungjun.github.io/), [Milan Vojnović](https://personal.lse.ac.uk/vojnovic/), and [Se-Young Yun](https://fbsqkd.github.io/).

If you plan to use this repository or cite our paper, please use the following bibtex format:

```latex
@InProceedings{lee2026gl-lowpopart,
  title = 	 {{GL-LowPopArt: A Nearly Instance-Wise Minimax-Optimal Estimator for Generalized Low-Rank Trace Regression}},
  author =       {Lee, Junghyun and Jang, Kyoungseok and Jun, Kwang-Sung and Vojnovi\'{c}, Milan and Yun, Se-Young},
  booktitle = 	 {Proceedings of The 29th International Conference on Artificial Intelligence and Statistics},
  year = 	 {2026},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {02--05 May},
  publisher =    {PMLR},
  url = 	 {https://openreview.net/forum?id=mMxiIP3Gxu},
}
```



# Before Running
You require a license of [Mosek](https://www.mosek.com) to run the codes.

# Install
This repository uses `uv` for environment and dependency management.

1) Install `uv` (if not installed):
```shell
$ curl -LsSf https://astral.sh/uv/install.sh | sh
```

2) Create and activate a virtual environment:
```shell
$ uv venv --python 3.10
$ source .venv/bin/activate
```

3) Install dependencies:
```shell
$ uv sync
```

# Reproducing Figure 1
Run
```shell
$ bash fig1.sh
```
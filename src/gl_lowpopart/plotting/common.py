"""Shared plotting helpers."""

import matplotlib.pyplot as plt

from gl_lowpopart.config import figure_file, result_file


def load_pair(fig: str, model: str):
    import json

    with open(result_file(fig, "completion", model), "r") as f:
        completion = json.load(f)
    with open(result_file(fig, "recovery", model), "r") as f:
        recovery = json.load(f)
    return completion, recovery


def set_style():
    plt.rcParams.update(
        {
            "font.size": 40,
            "axes.labelsize": 44,
            "axes.titlesize": 46,
            "xtick.labelsize": 32,
            "ytick.labelsize": 32,
            "legend.fontsize": 42,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_outputs(fig_name: str, model: str):
    plt.savefig(figure_file(fig_name, model, "png"), dpi=300, bbox_inches="tight")
    plt.savefig(figure_file(fig_name, model, "pdf"), dpi=600, bbox_inches="tight")
    plt.close()


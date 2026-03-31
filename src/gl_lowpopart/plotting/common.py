"""Shared plotting helpers."""

import json
import os

import matplotlib.pyplot as plt

from gl_lowpopart.config import figure_file, result_file


def load_pair(fig: str, model: str):
    available = load_available_modes(fig, model)
    missing = [mode for mode in ("completion", "recovery", "hard") if mode not in available]
    if missing:
        missing_paths = [result_file(fig, mode, model) for mode in missing]
        raise FileNotFoundError(f"Missing required JSON outputs: {missing_paths}")
    return available["completion"], available["recovery"], available["hard"]


def load_available_modes(fig: str, model: str):
    available = {}
    for mode in ("completion", "recovery", "hard"):
        path = result_file(fig, mode, model)
        if os.path.exists(path):
            with open(path, "r") as f:
                available[mode] = json.load(f)
    if not available:
        paths = [result_file(fig, mode, model) for mode in ("completion", "recovery", "hard")]
        raise FileNotFoundError(f"No JSON outputs found. Expected one of: {paths}")
    return available


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


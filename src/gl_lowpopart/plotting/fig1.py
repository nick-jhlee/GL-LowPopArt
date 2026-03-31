"""Plot Figure 1 from JSON outputs."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gl_lowpopart.config import RESULTS_JSON_DIR
from gl_lowpopart.plotting.common import load_pair, save_outputs, set_style


def plot_data(ax, data, Ns, methods, styles, title, specific_Ns=None):
    results = {}
    for method, label in tqdm(methods.items(), desc="Methods"):
        means = data[method]["mean"]
        cis = data[method]["ci"]
        sorted_indices = np.argsort(Ns)
        sorted_Ns = np.array(Ns)[sorted_indices]
        sorted_means = np.array(means)[sorted_indices]
        sorted_cis = np.array(cis)[sorted_indices]
        if specific_Ns is not None:
            mask = np.isin(sorted_Ns, specific_Ns)
            sorted_Ns = sorted_Ns[mask]
            sorted_means = sorted_means[mask]
            sorted_cis = sorted_cis[mask]

        lower_cis = [ci[0] for ci in sorted_cis]
        upper_cis = [ci[1] for ci in sorted_cis]

        ax.errorbar(
            sorted_Ns,
            sorted_means,
            yerr=[sorted_means - np.array(lower_cis), np.array(upper_cis) - sorted_means],
            label=label,
            color=styles[method]["color"],
            linestyle=styles[method]["linestyle"],
            marker=styles[method]["marker"],
            capsize=5,
            linewidth=2,
            markersize=8,
        )
        results[method] = {"mean": sorted_means.tolist(), "lower_ci": lower_cis, "upper_ci": upper_cis}

    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel("Nuclear Norm Error")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.7)
    return results


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 1 results")
    parser.add_argument("--model", type=str, choices=["bernoulli", "poisson"], default="bernoulli")
    args = parser.parse_args()

    completion_data, recovery_data = load_pair("Fig1", args.model)
    completion_Ns = completion_data["metadata"]["Ns"]
    recovery_Ns = recovery_data["metadata"]["Ns"]

    methods = {
        "Stage I (no E-optimal)": "U",
        "Stage I (with E-optimal)": "E",
        "Stage I+II (no E, no GL)": "U+U",
        "Stage I+II (with E, no GL)": "E+U",
        "Stage I+II (no E, with GL)": "U+GL",
        "Stage I+II (with E, with GL)": "E+GL",
    }
    if "BMF" in completion_data and "BMF" in recovery_data:
        methods = {"BMF": "BMF-GD", **methods}

    styles = {
        "Stage I (no E-optimal)": {"color": "#0072B2", "linestyle": "--", "marker": "s"},
        "Stage I (with E-optimal)": {"color": "#0072B2", "linestyle": "-", "marker": "^"},
        "Stage I+II (no E, no GL)": {"color": "#E69F00", "linestyle": "--", "marker": "D"},
        "Stage I+II (with E, no GL)": {"color": "#E69F00", "linestyle": "-", "marker": "v"},
        "Stage I+II (no E, with GL)": {"color": "#009E73", "linestyle": "--", "marker": "<"},
        "Stage I+II (with E, with GL)": {"color": "#009E73", "linestyle": "-", "marker": ">"},
    }
    if "BMF" in methods:
        styles["BMF"] = {"color": "#000000", "linestyle": "-", "marker": "o"}

    set_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(56, 16))
    completion_results = plot_data(
        ax1, completion_data, completion_Ns, methods, styles, "Matrix Completion", specific_Ns=[10000, 20000, 30000, 40000, 50000]
    )
    recovery_results = plot_data(
        ax2, recovery_data, recovery_Ns, methods, styles, "Matrix Recovery", specific_Ns=[10000, 20000, 30000, 40000, 50000]
    )
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=len(methods), frameon=True)
    plt.tight_layout()
    save_outputs("fig1", args.model)

    suffix = "" if args.model == "bernoulli" else f"_{args.model}"
    with open(f"{RESULTS_JSON_DIR}/fig1_results{suffix}.json", "w") as f:
        json.dump(
            {
                "completion": {"Ns": completion_Ns, "methods": completion_results},
                "recovery": {"Ns": recovery_Ns, "methods": recovery_results},
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()


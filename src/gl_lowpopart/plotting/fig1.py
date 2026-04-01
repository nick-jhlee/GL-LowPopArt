"""Plot Figure 1 from JSON outputs."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gl_lowpopart.config import RESULTS_JSON_DIR
from gl_lowpopart.plotting.common import load_available_modes, save_outputs, set_style


def plot_data(ax, data, Ns, methods, styles, title, specific_Ns=None):
    results = {}
    for method, label in tqdm(methods.items(), desc="Methods"):
        means = data[method]["mean"]
        cis = data[method]["ci"]
        sorted_indices = np.argsort(Ns)
        sorted_Ns = np.array(Ns)[sorted_indices]
        sorted_means = np.array(means, dtype=float)[sorted_indices]
        sorted_cis = np.array(cis, dtype=float)[sorted_indices]
        if specific_Ns is not None:
            mask = np.isin(sorted_Ns, specific_Ns)
            # Fall back to all available Ns when the requested subset is too sparse.
            if np.count_nonzero(mask) >= 2:
                sorted_Ns = sorted_Ns[mask]
                sorted_means = sorted_means[mask]
                sorted_cis = sorted_cis[mask]

        lower_cis = sorted_cis[:, 0]
        upper_cis = sorted_cis[:, 1]

        # Ensure valid CI ordering and non-negative error bars for matplotlib.
        lower_cis = np.minimum(lower_cis, upper_cis)
        upper_cis = np.maximum(lower_cis, upper_cis)
        lower_cis = np.minimum(lower_cis, sorted_means)
        upper_cis = np.maximum(upper_cis, sorted_means)

        yerr_lower = np.maximum(0.0, sorted_means - lower_cis)
        yerr_upper = np.maximum(0.0, upper_cis - sorted_means)

        finite_mask = np.isfinite(sorted_Ns) & np.isfinite(sorted_means) & np.isfinite(yerr_lower) & np.isfinite(yerr_upper)
        sorted_Ns = sorted_Ns[finite_mask]
        sorted_means = sorted_means[finite_mask]
        lower_cis = lower_cis[finite_mask]
        upper_cis = upper_cis[finite_mask]
        yerr_lower = yerr_lower[finite_mask]
        yerr_upper = yerr_upper[finite_mask]

        ax.errorbar(
            sorted_Ns,
            sorted_means,
            yerr=[yerr_lower, yerr_upper],
            label=label,
            color=styles[method]["color"],
            linestyle=styles[method]["linestyle"],
            marker=styles[method]["marker"],
            capsize=5,
            linewidth=2,
            markersize=8,
        )
        results[method] = {"mean": sorted_means.tolist(), "lower_ci": lower_cis.tolist(), "upper_ci": upper_cis.tolist()}

    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel("Nuclear Norm Error")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.7)
    return results


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 1 results")
    parser.add_argument("--model", type=str, choices=["bernoulli", "poisson"], default="bernoulli")
    args = parser.parse_args()

    mode_data = load_available_modes("Fig1", args.model)
    mode_order = [mode for mode in ("completion", "recovery", "hard") if mode in mode_data]

    method_labels = {
        "Stage I (no E-optimal)": "U",
        "Stage I (with E-optimal)": "E",
        "Stage I+II (no E, no GL)": "U+U",
        "Stage I+II (with E, no GL)": "E+U",
        "Stage I+II (no E, with GL)": "U+GL",
        "Stage I+II (with E, with GL)": "E+GL",
    }
    if any("BMF" in mode_data[mode] for mode in mode_order):
        method_labels = {"BMF": "BMF-GD", **method_labels}

    styles = {
        "Stage I (no E-optimal)": {"color": "#0072B2", "linestyle": "--", "marker": "s"},
        "Stage I (with E-optimal)": {"color": "#0072B2", "linestyle": "-", "marker": "^"},
        "Stage I+II (no E, no GL)": {"color": "#E69F00", "linestyle": "--", "marker": "D"},
        "Stage I+II (with E, no GL)": {"color": "#E69F00", "linestyle": "-", "marker": "v"},
        "Stage I+II (no E, with GL)": {"color": "#009E73", "linestyle": "--", "marker": "<"},
        "Stage I+II (with E, with GL)": {"color": "#009E73", "linestyle": "-", "marker": ">"},
    }
    if "BMF" in method_labels:
        styles["BMF"] = {"color": "#000000", "linestyle": "-", "marker": "o"}

    set_style()
    fig, axes = plt.subplots(len(mode_order), 1, figsize=(28, 12 * len(mode_order)))
    if len(mode_order) == 1:
        axes = [axes]

    mode_titles = {
        "completion": "Matrix Completion",
        "recovery": "Matrix Recovery",
        "hard": "Hard Instance",
    }
    per_mode_results = {}
    legend_handles = {}
    for ax, mode in zip(axes, mode_order):
        data = mode_data[mode]
        Ns = data["metadata"]["Ns"]
        mode_methods = {method: label for method, label in method_labels.items() if method in data}
        if not mode_methods:
            raise ValueError(f"No methods available to plot for mode '{mode}'.")

        per_mode_results[mode] = {"Ns": Ns, "methods": plot_data(ax, data, Ns, mode_methods, styles, mode_titles[mode])}

        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_handles:
                legend_handles[label] = handle
        # , specific_Ns=[10000, 20000, 30000, 40000, 50000]

    if legend_handles:
        labels = list(legend_handles.keys())
        handles = [legend_handles[label] for label in labels]
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=len(labels), frameon=True)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save_outputs("fig1", args.model)

    suffix = "" if args.model == "bernoulli" else f"_{args.model}"
    with open(f"{RESULTS_JSON_DIR}/fig1_results{suffix}.json", "w") as f:
        json.dump(per_mode_results, f, indent=2)


if __name__ == "__main__":
    main()


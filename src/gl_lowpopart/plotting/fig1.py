"""Plot Figure 1 from JSON outputs."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from tqdm import tqdm

from gl_lowpopart.config import RESULTS_JSON_DIR
from gl_lowpopart.plotting.common import load_available_modes, save_outputs, set_style


def select_metric_entry(method_entry, metric):
    if metric in method_entry:
        return method_entry[metric]
    if metric == "nuc" and "mean" in method_entry:
        return method_entry
    raise KeyError(f"Missing '{metric}' metric data. Re-run Figure 1 experiments to generate it.")


def plot_data(ax, data, Ns, methods, styles, title, metric, specific_Ns=None):
    results = {}
    for method, label in tqdm(methods.items(), desc="Methods"):
        metric_entry = select_metric_entry(data[method], metric)
        means = metric_entry["mean"]
        cis = metric_entry["ci"]
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
    y_label = "Nuclear Norm Error" if metric == "nuc" else "Operator Norm Error"
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.7)
    return results


def add_inset(ax, data, Ns, methods, styles, metric, n_min=1000, n_max=10000):
    inset_ax = inset_axes(ax, width="40%", height="40%", loc="upper right", borderpad=1.2)
    subset = [N for N in sorted(set(Ns)) if n_min <= N <= n_max]
    if len(subset) < 2:
        return

    for method in methods:
        metric_entry = select_metric_entry(data[method], metric)
        means = np.asarray(metric_entry["mean"], dtype=float)
        cis = np.asarray(metric_entry["ci"], dtype=float)
        sorted_indices = np.argsort(Ns)
        sorted_Ns = np.array(Ns)[sorted_indices]
        sorted_means = means[sorted_indices]
        sorted_cis = cis[sorted_indices]

        mask = np.isin(sorted_Ns, subset)
        if np.count_nonzero(mask) < 2:
            continue

        sorted_Ns = sorted_Ns[mask]
        sorted_means = sorted_means[mask]
        sorted_cis = sorted_cis[mask]

        lower_cis = sorted_cis[:, 0]
        upper_cis = sorted_cis[:, 1]
        lower_cis = np.minimum(lower_cis, upper_cis)
        upper_cis = np.maximum(lower_cis, upper_cis)
        lower_cis = np.minimum(lower_cis, sorted_means)
        upper_cis = np.maximum(upper_cis, sorted_means)
        yerr_lower = np.maximum(0.0, sorted_means - lower_cis)
        yerr_upper = np.maximum(0.0, upper_cis - sorted_means)

        finite_mask = np.isfinite(sorted_Ns) & np.isfinite(sorted_means) & np.isfinite(yerr_lower) & np.isfinite(yerr_upper)
        sorted_Ns = sorted_Ns[finite_mask]
        sorted_means = sorted_means[finite_mask]
        yerr_lower = yerr_lower[finite_mask]
        yerr_upper = yerr_upper[finite_mask]

        if len(sorted_Ns) < 2:
            continue

        inset_ax.errorbar(
            sorted_Ns,
            sorted_means,
            yerr=[yerr_lower, yerr_upper],
            color=styles[method]["color"],
            linestyle=styles[method]["linestyle"],
            marker=styles[method]["marker"],
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label="_nolegend_",
        )

    inset_ax.set_xlim(n_min - 20, n_max + 20)
    inset_ax.set_xticks([n_min, (n_min + n_max) // 2, n_max])
    inset_ax.tick_params(axis="both", labelsize=12)
    inset_ax.grid(True, linestyle="--", alpha=0.5)
    mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.5", lw=1.0)


def render_metric(mode_data, mode_order, method_labels, styles, model, metric):
    fig, axes = plt.subplots(len(mode_order), 1, figsize=(28, 12 * len(mode_order)))
    if len(mode_order) == 1:
        axes = [axes]

    mode_titles = {
        "completion": "Matrix Completion",
        "recovery": "Matrix Recovery (Random)",
        "hard": "Matrix Recovery (Hard)",
    }
    per_mode_results = {}
    legend_handles = {}
    for ax, mode in zip(axes, mode_order):
        data = mode_data[mode]
        Ns = data["metadata"]["Ns"]
        mode_methods = {method: label for method, label in method_labels.items() if method in data}
        if not mode_methods:
            raise ValueError(f"No methods available to plot for mode '{mode}'.")

        per_mode_results[mode] = {
            "Ns": Ns,
            "methods": plot_data(ax, data, Ns, mode_methods, styles, mode_titles[mode], metric),
        }
        # add_inset(ax, data, Ns, mode_methods, styles, metric, n_min=100, n_max=1000)

        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in legend_handles:
                legend_handles[label] = handle

    if legend_handles:
        labels = list(legend_handles.keys())
        handles = [legend_handles[label] for label in labels]
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=len(labels), frameon=True)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    figure_stem = "fig1" if metric == "nuc" else "fig1-op"
    save_outputs(figure_stem, model)

    suffix = "" if model == "bernoulli" else f"_{model}"
    results_stem = "fig1_results" if metric == "nuc" else "fig1_results_op"
    with open(f"{RESULTS_JSON_DIR}/{results_stem}{suffix}.json", "w") as f:
        json.dump(per_mode_results, f, indent=2)


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
        "Stage I (no E-optimal)": {"color": "#0072B2", "linestyle": "-", "marker": "s"},
        "Stage I (with E-optimal)": {"color": "#0072B2", "linestyle": "-", "marker": "^"},
        "Stage I+II (no E, no GL)": {"color": "#E69F00", "linestyle": "-", "marker": "D"},
        "Stage I+II (with E, no GL)": {"color": "#E69F00", "linestyle": "-", "marker": "v"},
        "Stage I+II (no E, with GL)": {"color": "#009E73", "linestyle": "-", "marker": "<"},
        "Stage I+II (with E, with GL)": {"color": "#009E73", "linestyle": "-", "marker": ">"},
    }
    if "BMF" in method_labels:
        styles["BMF"] = {"color": "#000000", "linestyle": "-", "marker": "o"}

    set_style()
    for metric in ("nuc", "op"):
        render_metric(mode_data, mode_order, method_labels, styles, args.model, metric)


if __name__ == "__main__":
    main()


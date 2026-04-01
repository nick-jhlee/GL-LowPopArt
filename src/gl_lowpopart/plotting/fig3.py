"""Plot Figure 3 (fixed N, varying N1 ablation)."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np

from gl_lowpopart.config import RESULTS_JSON_DIR, result_file
from gl_lowpopart.plotting.common import save_outputs, set_style


def load_mode_data(fig, mode, model):
    with open(result_file(fig, mode, model), "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Plot Figure 3 ablation results")
    parser.add_argument("--model", type=str, choices=["bernoulli", "poisson"], default="bernoulli")
    parser.add_argument(
        "--modes",
        type=str,
        default="completion,recovery,hard",
        help="Comma-separated subset of completion,recovery,hard",
    )
    args = parser.parse_args()

    mode_order = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid_modes = [m for m in ("completion", "recovery", "hard") if m in mode_order]
    if not valid_modes:
        raise ValueError("No valid modes specified. Use any subset of: completion,recovery,hard")

    mode_data = {}
    for mode in valid_modes:
        mode_data[mode] = load_mode_data("Fig3", mode, args.model)

    styles = {
        "completion": {"color": "#0072B2", "marker": "o", "label": "Matrix Completion"},
        "recovery": {"color": "#E69F00", "marker": "s", "label": "Matrix Recovery"},
        "hard": {"color": "#009E73", "marker": "^", "label": "Hard Instance"},
    }

    set_style()
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    results_json = {}
    for mode in valid_modes:
        data = mode_data[mode]
        n1_values = np.array(data["metadata"]["N1_values"], dtype=float)
        n2_values = np.array(data["metadata"]["N2_values"], dtype=float)
        means = np.array(data["E+GL"]["mean"], dtype=float)
        cis = np.array(data["E+GL"]["ci"], dtype=float)
        lower = cis[:, 0]
        upper = cis[:, 1]

        style = styles[mode]
        ax.errorbar(
            n1_values,
            means,
            yerr=[means - lower, upper - means],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2,
            markersize=8,
            capsize=5,
            label=style["label"],
        )
        results_json[mode] = {
            "N1_values": n1_values.tolist(),
            "N2_values": n2_values.tolist(),
            "mean": means.tolist(),
            "lower_ci": lower.tolist(),
            "upper_ci": upper.tolist(),
        }

    ax.set_xlabel("Stage-I Budget N1")
    ax.set_ylabel("Nuclear Norm Error (lower is better)")
    ax.set_title("GL-LowPopArt Ablation: Fixed N, Varying N1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(frameon=True)
    plt.tight_layout()
    save_outputs("fig3", args.model)

    suffix = "" if args.model == "bernoulli" else f"_{args.model}"
    with open(f"{RESULTS_JSON_DIR}/fig3_results{suffix}.json", "w") as f:
        json.dump(results_json, f, indent=2)


if __name__ == "__main__":
    main()


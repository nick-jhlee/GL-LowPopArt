"""Figure 3 experiment runner: fixed total N, vary Stage-I budget N1."""

import argparse
import json
import logging
import multiprocessing as mp
from datetime import datetime

import numpy as np
from tqdm import tqdm

from gl_lowpopart.config import (
    DEFAULT_FIG3_N1_VALUES,
    DEFAULT_FIG3_N_TOTAL,
    DEFAULT_PARAMS,
    result_file,
    setup_logging,
)
from gl_lowpopart.core.optimization import GL_LowPopArt, nuc_norm_MLE
from gl_lowpopart.experiments.common import build_env, build_problem_instances
from gl_lowpopart.utils import mean_t_ci


def parse_n1_values(raw_n1_values, n_total):
    if raw_n1_values:
        n1_values = [int(x.strip()) for x in raw_n1_values.split(",") if x.strip()]
    else:
        # Paper-inspired sweep points for N=1e5.
        n1_values = list(DEFAULT_FIG3_N1_VALUES)
    n1_values = sorted(set(n1_values))
    valid = [n1 for n1 in n1_values if 0 < n1 < n_total]
    if not valid:
        raise ValueError(f"No valid N1 values. Each N1 must satisfy 0 < N1 < N_total ({n_total}).")
    return valid


def run_single_repetition(args):
    run_idx, n1, n_total, mode, model, d1, d2, r, k, delta, c_lambda, instance, stage1_solver = args
    arm_set, theta_star = instance
    env = build_env(arm_set, theta_star, model=model)

    n2 = n_total - n1
    nuc_coef = np.sqrt(c_lambda * np.log((d1 + d2) / delta) / n1)
    theta0, _, _ = nuc_norm_MLE(env, n1, d1, d2, nuc_coef, E_optimal=True, stage1_solver=stage1_solver)
    theta_hat = GL_LowPopArt(env, n2, d1, d2, delta, theta0, GL_optimal=True)
    error = float(np.linalg.norm(theta_hat - theta_star, "nuc"))
    return run_idx, error


def run_experiment(mode, model, d1, d2, r, k, num_repeats, n_total, n1_values, delta, c_lambda, seed=42, stage1_solver="fista"):
    np.random.seed(seed)
    problem_instances = build_problem_instances(mode, model, d1, d2, r, k, num_repeats, seed=seed)
    errors_all = []

    for n1 in tqdm(n1_values, desc="Stage-I budgets (N1)"):
        logging.info("Processing N1=%s (N2=%s)", n1, n_total - n1)
        args_list = [
            (run_idx, n1, n_total, mode, model, d1, d2, r, k, delta, c_lambda, problem_instances[run_idx], stage1_solver)
            for run_idx in range(num_repeats)
        ]
        errors_reps = [np.nan] * num_repeats
        try:
            num_cores = min(num_repeats, max(1, mp.cpu_count() - 1))
            with mp.Pool(processes=num_cores) as pool:
                results = list(tqdm(pool.imap(run_single_repetition, args_list), total=num_repeats, desc=f"Repeats N1={n1}", leave=False))
            for run_idx, error in results:
                errors_reps[run_idx] = error
        except Exception as exc:
            logging.error("Parallel processing failed for N1=%s: %s", n1, exc)
            for run_idx in tqdm(range(num_repeats), desc=f"Repeats N1={n1}", leave=False):
                idx, error = run_single_repetition(args_list[run_idx])
                errors_reps[idx] = error
        errors_all.append(errors_reps)
    return errors_all


def save_results(errors_all, n1_values, n_total, mode, model, params, logger):
    n2_values = [n_total - n1 for n1 in n1_values]
    results = {
        "E+GL": {
            "mean": np.mean(errors_all, axis=1).tolist(),
            "ci": [mean_t_ci(errors_all[i]) for i in range(len(errors_all))],
            "raw": {str(n1): errors_all[i] for i, n1 in enumerate(n1_values)},
        },
        "metadata": {
            "mode": mode,
            "model": model,
            "N_total": n_total,
            "N1_values": n1_values,
            "N2_values": n2_values,
            "params": params,
            "timestamp": datetime.now().isoformat(),
        },
    }
    with open(result_file("Fig3", mode, model), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", result_file("Fig3", mode, model))


def main():
    parser = argparse.ArgumentParser(description="Run Figure 3 ablation: vary N1 with fixed total N")
    parser.add_argument("--mode", type=str, choices=["completion", "recovery", "hard"], required=True)
    parser.add_argument("--model", type=str, choices=["bernoulli", "poisson"], default="bernoulli")
    parser.add_argument("--d1", type=int, default=DEFAULT_PARAMS["d1"])
    parser.add_argument("--d2", type=int, default=DEFAULT_PARAMS["d2"])
    parser.add_argument("--r", type=int, default=DEFAULT_PARAMS["r"])
    parser.add_argument("--num_repeats", type=int, default=DEFAULT_PARAMS["num_repeats"])
    parser.add_argument("--N_total", type=int, default=DEFAULT_FIG3_N_TOTAL)
    parser.add_argument("--N1_values", type=str, default=None, help="Comma-separated list, e.g. 5000,10000,20000")
    parser.add_argument("--stage1_solver", type=str, choices=["fista", "cvxpy"], default="fista")
    args = parser.parse_args()

    logger = setup_logging(args.mode, "fig3", model=args.model)
    n1_values = parse_n1_values(args.N1_values, args.N_total)

    params = DEFAULT_PARAMS.copy()
    params.update(
        {
            "model": args.model,
            "d1": args.d1,
            "d2": args.d2,
            "r": args.r,
            "num_repeats": args.num_repeats,
            "stage1_solver": args.stage1_solver,
            "N_total": args.N_total,
            "N1_values": n1_values,
        }
    )
    errors_all = run_experiment(
        args.mode,
        args.model,
        args.d1,
        args.d2,
        args.r,
        params["K"],
        params["num_repeats"],
        args.N_total,
        n1_values,
        params["delta"],
        params["c_lambda"],
        stage1_solver=args.stage1_solver,
    )
    save_results(errors_all, n1_values, args.N_total, args.mode, args.model, params, logger)


if __name__ == "__main__":
    main()


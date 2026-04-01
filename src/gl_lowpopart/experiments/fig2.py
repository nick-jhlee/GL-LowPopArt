"""Figure 2 experiment runner."""

import argparse
import json
import logging
import multiprocessing as mp
import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

from gl_lowpopart.config import DEFAULT_PARAMS, result_file, setup_logging
from gl_lowpopart.core.optimization import GL_LowPopArt, nuc_norm_MLE
from gl_lowpopart.experiments.common import build_env, build_problem_instances
from gl_lowpopart.utils import mean_t_ci


def run_single_repetition(args):
    run_idx, N, mode, model, d1, d2, r, K, delta, c_lambda, c_nu, Rmax, instance, stage1_solver = args
    arm_set, Theta_star = instance
    env = build_env(arm_set, Theta_star, model=model)

    error_uniform = None
    error_e_optimal = None
    error_zero = None
    error_random = None

    fig2_file = result_file("Fig2", mode, model, intermediate=True)
    if os.path.exists(fig2_file):
        try:
            with open(fig2_file, "r") as f:
                fig2_results = json.load(f)
            if str(N) in fig2_results["uniform"]["raw"]:
                error_uniform = fig2_results["uniform"]["raw"][str(N)][run_idx]
            if str(N) in fig2_results["e_optimal"]["raw"]:
                error_e_optimal = fig2_results["e_optimal"]["raw"][str(N)][run_idx]
            if str(N) in fig2_results["zero"]["raw"]:
                error_zero = fig2_results["zero"]["raw"][str(N)][run_idx]
            if str(N) in fig2_results["random"]["raw"]:
                error_random = fig2_results["random"]["raw"][str(N)][run_idx]
        except Exception:
            pass

    if error_uniform is None or error_e_optimal is None:
        fig1_file = result_file("Fig1", mode, model, intermediate=True)
        if os.path.exists(fig1_file):
            try:
                with open(fig1_file, "r") as f:
                    fig1_results = json.load(f)
                if error_uniform is None and str(N) in fig1_results["Stage I+II (no E, with GL)"]["raw"]:
                    error_uniform = fig1_results["Stage I+II (no E, with GL)"]["raw"][str(N)][run_idx]
                if error_e_optimal is None and str(N) in fig1_results["Stage I+II (with E, with GL)"]["raw"]:
                    error_e_optimal = fig1_results["Stage I+II (with E, with GL)"]["raw"][str(N)][run_idx]
            except Exception:
                pass

    N1 = N // 2
    N2 = N - N1
    nuc_coef = c_lambda * np.sqrt(8 * Rmax * np.log((d1 + d2) / delta) / N1)

    if error_uniform is None:
        Theta0_uniform, _, _ = nuc_norm_MLE(env, N1, d1, d2, nuc_coef, E_optimal=False, stage1_solver=stage1_solver)
        Theta_uniform = GL_LowPopArt(env, N2, d1, d2, delta, Theta0_uniform, c_nu)
        error_uniform = float(np.linalg.norm(Theta_uniform - Theta_star, "nuc"))

    if error_e_optimal is None:
        Theta0_e_optimal, _, _ = nuc_norm_MLE(env, N1, d1, d2, nuc_coef, E_optimal=True, stage1_solver=stage1_solver)
        Theta_e_optimal = GL_LowPopArt(env, N2, d1, d2, delta, Theta0_e_optimal, c_nu)
        error_e_optimal = float(np.linalg.norm(Theta_e_optimal - Theta_star, "nuc"))

    if error_zero is None:
        Theta0_zero = np.zeros((d1, d2))
        Theta_zero = GL_LowPopArt(env, N, d1, d2, delta, Theta0_zero, c_nu)
        error_zero = float(np.linalg.norm(Theta_zero - Theta_star, "nuc"))

    if error_random is None:
        Theta0_random = np.random.randn(d1, d2) * 0.1
        Theta_random = GL_LowPopArt(env, N, d1, d2, delta, Theta0_random, c_nu)
        error_random = float(np.linalg.norm(Theta_random - Theta_star, "nuc"))

    return {
        "error_uniform": error_uniform,
        "error_e_optimal": error_e_optimal,
        "error_zero": error_zero,
        "error_random": error_random,
    }


def run_experiment(mode, model, d1, d2, r, K, num_repeats, delta, Ns, c_lambda, c_nu, Rmax, seed=42, stage1_solver="fista"):
    np.random.seed(seed)
    problem_instances = build_problem_instances(mode, model, d1, d2, r, K, num_repeats, seed=seed)

    errors_uniform_all = []
    errors_e_optimal_all = []
    errors_zero_all = []
    errors_random_all = []

    pbar = tqdm(Ns, desc="Sample sizes")
    run_params = {
        "d1": d1,
        "d2": d2,
        "r": r,
        "K": K,
        "num_repeats": num_repeats,
        "delta": delta,
        "c_lambda": c_lambda,
        "c_nu": c_nu,
        "Rmax": Rmax,
        "stage1_solver": stage1_solver,
    }
    for N in pbar:
        pbar.set_description(f"Processing N={N}")
        logging.info("Processing N=%s", N)

        errors_uniform_reps = []
        errors_e_optimal_reps = []
        errors_zero_reps = []
        errors_random_reps = []

        results_path = result_file("Fig2", mode, model, intermediate=True)
        if os.path.exists(results_path):
            try:
                with open(results_path, "r") as f:
                    current_results = json.load(f)
                metadata = current_results.get("metadata", {})
                params_match = metadata.get("mode") == mode and metadata.get("model") == model and metadata.get("params") == run_params
                if params_match and len(current_results["uniform"]["mean"]) > Ns.index(N):
                    errors_uniform_reps = current_results["uniform"]["raw"][str(N)]
                    errors_e_optimal_reps = current_results["e_optimal"]["raw"][str(N)]
                    errors_zero_reps = current_results["zero"]["raw"][str(N)]
                    errors_random_reps = current_results["random"]["raw"][str(N)]
                    errors_uniform_all.append(errors_uniform_reps)
                    errors_e_optimal_all.append(errors_e_optimal_reps)
                    errors_zero_all.append(errors_zero_reps)
                    errors_random_all.append(errors_random_reps)
                    continue
            except Exception:
                pass

        args_list = [
            (run_idx, N, mode, model, d1, d2, r, K, delta, c_lambda, c_nu, Rmax, problem_instances[run_idx], stage1_solver)
            for run_idx in range(num_repeats)
        ]

        try:
            num_cores = min(num_repeats, max(1, mp.cpu_count() - 1))
            with mp.Pool(processes=num_cores) as pool:
                results = list(tqdm(pool.imap(run_single_repetition, args_list), total=num_repeats, desc=f"Repeats N={N}", leave=False))
            for result in results:
                errors_uniform_reps.append(result["error_uniform"])
                errors_e_optimal_reps.append(result["error_e_optimal"])
                errors_zero_reps.append(result["error_zero"])
                errors_random_reps.append(result["error_random"])
        except Exception as exc:
            logging.error("Parallel processing failed for N=%s: %s", N, exc)
            for run_idx in tqdm(range(num_repeats), desc=f"Repeats N={N}", leave=False):
                result = run_single_repetition(args_list[run_idx])
                errors_uniform_reps.append(result["error_uniform"])
                errors_e_optimal_reps.append(result["error_e_optimal"])
                errors_zero_reps.append(result["error_zero"])
                errors_random_reps.append(result["error_random"])

        errors_uniform_all.append(errors_uniform_reps)
        errors_e_optimal_all.append(errors_e_optimal_reps)
        errors_zero_all.append(errors_zero_reps)
        errors_random_all.append(errors_random_reps)

        current_results = {
            "uniform": {
                "mean": np.mean(errors_uniform_all, axis=1).tolist(),
                "raw": {str(x): errors_uniform_all[i] for i, x in enumerate(Ns[: len(errors_uniform_all)])},
            },
            "e_optimal": {
                "mean": np.mean(errors_e_optimal_all, axis=1).tolist(),
                "raw": {str(x): errors_e_optimal_all[i] for i, x in enumerate(Ns[: len(errors_e_optimal_all)])},
            },
            "zero": {
                "mean": np.mean(errors_zero_all, axis=1).tolist(),
                "raw": {str(x): errors_zero_all[i] for i, x in enumerate(Ns[: len(errors_zero_all)])},
            },
            "random": {
                "mean": np.mean(errors_random_all, axis=1).tolist(),
                "raw": {str(x): errors_random_all[i] for i, x in enumerate(Ns[: len(errors_random_all)])},
            },
            "metadata": {
                "mode": mode,
                "model": model,
                "Ns": Ns[: len(errors_uniform_all)],
                "params": {
                    "d1": d1,
                    "d2": d2,
                    "r": r,
                    "K": K,
                    "num_repeats": num_repeats,
                    "delta": delta,
                    "c_lambda": c_lambda,
                    "c_nu": c_nu,
                    "Rmax": Rmax,
                    "stage1_solver": stage1_solver,
                },
                "timestamp": datetime.now().isoformat(),
            },
        }
        with open(result_file("Fig2", mode, model, intermediate=True), "w") as f:
            json.dump(current_results, f, indent=2)

    return errors_uniform_all, errors_e_optimal_all, errors_zero_all, errors_random_all


def save_results(all_errors, Ns, mode, model, params, logger):
    errors_uniform_all, errors_e_optimal_all, errors_zero_all, errors_random_all = all_errors
    results = {
        "uniform": {
            "mean": np.mean(errors_uniform_all, axis=1).tolist(),
            "ci": [mean_t_ci(errors_uniform_all[i]) for i in range(len(errors_uniform_all))],
            "raw": {str(N): errors_uniform_all[i] for i, N in enumerate(Ns[: len(errors_uniform_all)])},
        },
        "e_optimal": {
            "mean": np.mean(errors_e_optimal_all, axis=1).tolist(),
            "ci": [mean_t_ci(errors_e_optimal_all[i]) for i in range(len(errors_e_optimal_all))],
            "raw": {str(N): errors_e_optimal_all[i] for i, N in enumerate(Ns[: len(errors_e_optimal_all)])},
        },
        "zero": {
            "mean": np.mean(errors_zero_all, axis=1).tolist(),
            "ci": [mean_t_ci(errors_zero_all[i]) for i in range(len(errors_zero_all))],
            "raw": {str(N): errors_zero_all[i] for i, N in enumerate(Ns[: len(errors_zero_all)])},
        },
        "random": {
            "mean": np.mean(errors_random_all, axis=1).tolist(),
            "ci": [mean_t_ci(errors_random_all[i]) for i in range(len(errors_random_all))],
            "raw": {str(N): errors_random_all[i] for i, N in enumerate(Ns[: len(errors_random_all)])},
        },
        "metadata": {"mode": mode, "model": model, "Ns": Ns, "params": params, "timestamp": datetime.now().isoformat()},
    }
    with open(result_file("Fig2", mode, model), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", result_file("Fig2", mode, model))


def main():
    parser = argparse.ArgumentParser(description="Run Figure 2 experiments")
    parser.add_argument("--mode", type=str, choices=["completion", "recovery"], required=True)
    parser.add_argument("--model", type=str, choices=["bernoulli", "poisson"], default="bernoulli")
    parser.add_argument("--d1", type=int, default=DEFAULT_PARAMS["d1"])
    parser.add_argument("--d2", type=int, default=DEFAULT_PARAMS["d2"])
    parser.add_argument("--r", type=int, default=DEFAULT_PARAMS["r"])
    parser.add_argument("--num_repeats", type=int, default=DEFAULT_PARAMS["num_repeats"])
    parser.add_argument("--stage1_solver", type=str, choices=["fista", "cvxpy"], default="fista")
    args = parser.parse_args()

    logger = setup_logging(args.mode, "fig2", model=args.model)
    params = DEFAULT_PARAMS.copy()
    params.update(
        {
            "model": args.model,
            "d1": args.d1,
            "d2": args.d2,
            "r": args.r,
            "num_repeats": args.num_repeats,
            "stage1_solver": args.stage1_solver,
        }
    )
    all_errors = run_experiment(
        args.mode,
        args.model,
        args.d1,
        args.d2,
        args.r,
        params["K"],
        params["num_repeats"],
        params["delta"],
        params["Ns"],
        params["c_lambda"],
        params["c_nu"],
        params["Rmax"],
        stage1_solver=args.stage1_solver,
    )
    save_results(all_errors, params["Ns"], args.mode, args.model, params, logger)


if __name__ == "__main__":
    main()


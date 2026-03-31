"""Figure 1 experiment runner."""

import argparse
import json
import logging
import multiprocessing as mp
import os
from datetime import datetime

import numpy as np
from tqdm import tqdm

from gl_lowpopart.config import DEFAULT_PARAMS, result_file, setup_logging
from gl_lowpopart.experiments.common import build_env, build_problem_instances, run_bmf, run_stage1, run_stage1_2
from gl_lowpopart.utils import studentized_double_bootstrap


def run_single_repetition(args):
    run_idx, N, mode, model, d1, d2, r, K, delta, c_lambda, c_nu, Rmax, instance = args
    arm_set, Theta_star = instance
    env = build_env(arm_set, Theta_star, model=model)

    nuc_coef = c_lambda * np.sqrt(8 * Rmax * np.log((d1 + d2) / delta) / N)
    error_stage1_no_e, X1, y1 = run_stage1(env, N, d1, d2, nuc_coef, False)
    error_stage1_with_e = run_stage1(env, N, d1, d2, nuc_coef, True)[0]
    error_bmf = run_bmf(env, d1, r, X1, y1) if model == "bernoulli" else np.nan

    N1 = N // 2
    N2 = N - N1
    nuc_coef = c_lambda * np.sqrt(8 * Rmax * np.log((d1 + d2) / delta) / N1)
    error_stage12_no_e_no_gl = run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, False, False)
    error_stage12_no_e_with_gl = run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, False, True)
    error_stage12_with_e_no_gl = run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, True, False)
    error_stage12_with_e_with_gl = run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, True, True)

    return {
        "error_bmf": error_bmf,
        "error_stage1_no_e": error_stage1_no_e,
        "error_stage1_with_e": error_stage1_with_e,
        "error_stage12_no_e_no_gl": error_stage12_no_e_no_gl,
        "error_stage12_no_e_with_gl": error_stage12_no_e_with_gl,
        "error_stage12_with_e_no_gl": error_stage12_with_e_no_gl,
        "error_stage12_with_e_with_gl": error_stage12_with_e_with_gl,
    }


def run_experiment(mode, model, d1, d2, r, K, num_repeats, delta, Ns, c_lambda, c_nu, Rmax, logger, seed=42):
    errors_bmf_all = []
    errors_stage1_no_e_all = []
    errors_stage1_with_e_all = []
    errors_stage12_no_e_no_gl_all = []
    errors_stage12_no_e_with_gl_all = []
    errors_stage12_with_e_no_gl_all = []
    errors_stage12_with_e_with_gl_all = []

    problem_instances = build_problem_instances(mode, model, d1, d2, r, K, num_repeats, seed=seed)

    pbar = tqdm(Ns, desc="Sample sizes")
    for N in pbar:
        pbar.set_description(f"Processing N={N}")
        logger.info("Processing N=%s", N)

        errors_bmf_reps = []
        errors_stage1_no_e_reps = []
        errors_stage1_with_e_reps = []
        errors_stage12_no_e_no_gl_reps = []
        errors_stage12_no_e_with_gl_reps = []
        errors_stage12_with_e_no_gl_reps = []
        errors_stage12_with_e_with_gl_reps = []

        results_file = result_file("Fig1", mode, model, intermediate=True)
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    current_results = json.load(f)
                key = "BMF" if "BMF" in current_results else "Stage I (no E-optimal)"
                if len(current_results[key]["mean"]) > Ns.index(N):
                    if "BMF" in current_results and "raw" in current_results["BMF"]:
                        errors_bmf_reps = current_results["BMF"]["raw"][str(N)]
                    else:
                        errors_bmf_reps = [np.nan] * num_repeats
                    errors_stage1_no_e_reps = current_results["Stage I (no E-optimal)"]["raw"][str(N)]
                    errors_stage1_with_e_reps = current_results["Stage I (with E-optimal)"]["raw"][str(N)]
                    errors_stage12_no_e_no_gl_reps = current_results["Stage I+II (no E, no GL)"]["raw"][str(N)]
                    errors_stage12_no_e_with_gl_reps = current_results["Stage I+II (no E, with GL)"]["raw"][str(N)]
                    errors_stage12_with_e_no_gl_reps = current_results["Stage I+II (with E, no GL)"]["raw"][str(N)]
                    errors_stage12_with_e_with_gl_reps = current_results["Stage I+II (with E, with GL)"]["raw"][str(N)]
                    errors_bmf_all.append(errors_bmf_reps)
                    errors_stage1_no_e_all.append(errors_stage1_no_e_reps)
                    errors_stage1_with_e_all.append(errors_stage1_with_e_reps)
                    errors_stage12_no_e_no_gl_all.append(errors_stage12_no_e_no_gl_reps)
                    errors_stage12_no_e_with_gl_all.append(errors_stage12_no_e_with_gl_reps)
                    errors_stage12_with_e_no_gl_all.append(errors_stage12_with_e_no_gl_reps)
                    errors_stage12_with_e_with_gl_all.append(errors_stage12_with_e_with_gl_reps)
                    continue
            except Exception as exc:
                logger.warning("Failed to load existing results: %s", exc)

        args_list = [
            (run_idx, N, mode, model, d1, d2, r, K, delta, c_lambda, c_nu, Rmax, problem_instances[run_idx])
            for run_idx in range(num_repeats)
        ]

        try:
            num_cores = min(num_repeats, max(1, mp.cpu_count() - 1))
            with mp.Pool(processes=num_cores) as pool:
                results = list(tqdm(pool.imap(run_single_repetition, args_list), total=num_repeats, desc=f"Repeats N={N}", leave=False))
            for result in results:
                errors_bmf_reps.append(result["error_bmf"])
                errors_stage1_no_e_reps.append(result["error_stage1_no_e"])
                errors_stage1_with_e_reps.append(result["error_stage1_with_e"])
                errors_stage12_no_e_no_gl_reps.append(result["error_stage12_no_e_no_gl"])
                errors_stage12_no_e_with_gl_reps.append(result["error_stage12_no_e_with_gl"])
                errors_stage12_with_e_no_gl_reps.append(result["error_stage12_with_e_no_gl"])
                errors_stage12_with_e_with_gl_reps.append(result["error_stage12_with_e_with_gl"])
        except Exception as exc:
            logger.error("Parallel processing failed for N=%s: %s", N, exc)
            for run_idx in tqdm(range(num_repeats), desc=f"Repeats N={N}", leave=False):
                result = run_single_repetition(args_list[run_idx])
                errors_bmf_reps.append(result["error_bmf"])
                errors_stage1_no_e_reps.append(result["error_stage1_no_e"])
                errors_stage1_with_e_reps.append(result["error_stage1_with_e"])
                errors_stage12_no_e_no_gl_reps.append(result["error_stage12_no_e_no_gl"])
                errors_stage12_no_e_with_gl_reps.append(result["error_stage12_no_e_with_gl"])
                errors_stage12_with_e_no_gl_reps.append(result["error_stage12_with_e_no_gl"])
                errors_stage12_with_e_with_gl_reps.append(result["error_stage12_with_e_with_gl"])

        errors_bmf_all.append(errors_bmf_reps)
        errors_stage1_no_e_all.append(errors_stage1_no_e_reps)
        errors_stage1_with_e_all.append(errors_stage1_with_e_reps)
        errors_stage12_no_e_no_gl_all.append(errors_stage12_no_e_no_gl_reps)
        errors_stage12_no_e_with_gl_all.append(errors_stage12_no_e_with_gl_reps)
        errors_stage12_with_e_no_gl_all.append(errors_stage12_with_e_no_gl_reps)
        errors_stage12_with_e_with_gl_all.append(errors_stage12_with_e_with_gl_reps)

        current_results = {
            "Stage I (no E-optimal)": {
                "mean": np.mean(errors_stage1_no_e_all, axis=1).tolist(),
                "raw": {str(x): errors_stage1_no_e_all[i] for i, x in enumerate(Ns[: len(errors_stage1_no_e_all)])},
            },
            "Stage I (with E-optimal)": {
                "mean": np.mean(errors_stage1_with_e_all, axis=1).tolist(),
                "raw": {str(x): errors_stage1_with_e_all[i] for i, x in enumerate(Ns[: len(errors_stage1_with_e_all)])},
            },
            "Stage I+II (no E, no GL)": {
                "mean": np.mean(errors_stage12_no_e_no_gl_all, axis=1).tolist(),
                "raw": {str(x): errors_stage12_no_e_no_gl_all[i] for i, x in enumerate(Ns[: len(errors_stage12_no_e_no_gl_all)])},
            },
            "Stage I+II (no E, with GL)": {
                "mean": np.mean(errors_stage12_no_e_with_gl_all, axis=1).tolist(),
                "raw": {str(x): errors_stage12_no_e_with_gl_all[i] for i, x in enumerate(Ns[: len(errors_stage12_no_e_with_gl_all)])},
            },
            "Stage I+II (with E, no GL)": {
                "mean": np.mean(errors_stage12_with_e_no_gl_all, axis=1).tolist(),
                "raw": {str(x): errors_stage12_with_e_no_gl_all[i] for i, x in enumerate(Ns[: len(errors_stage12_with_e_no_gl_all)])},
            },
            "Stage I+II (with E, with GL)": {
                "mean": np.mean(errors_stage12_with_e_with_gl_all, axis=1).tolist(),
                "raw": {str(x): errors_stage12_with_e_with_gl_all[i] for i, x in enumerate(Ns[: len(errors_stage12_with_e_with_gl_all)])},
            },
            "metadata": {
                "mode": mode,
                "model": model,
                "Ns": Ns[: len(errors_stage1_no_e_all)],
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
                },
                "timestamp": datetime.now().isoformat(),
            },
        }
        if model == "bernoulli":
            current_results["BMF"] = {
                "mean": np.mean(errors_bmf_all, axis=1).tolist(),
                "raw": {str(x): errors_bmf_all[i] for i, x in enumerate(Ns[: len(errors_bmf_all)])},
            }
        with open(result_file("Fig1", mode, model, intermediate=True), "w") as f:
            json.dump(current_results, f, indent=2)

    return (
        errors_bmf_all,
        errors_stage1_no_e_all,
        errors_stage1_with_e_all,
        errors_stage12_no_e_no_gl_all,
        errors_stage12_no_e_with_gl_all,
        errors_stage12_with_e_no_gl_all,
        errors_stage12_with_e_with_gl_all,
    )


def save_results(all_errors, Ns, mode, model, params, logger):
    (
        errors_bmf_all,
        errors_stage1_no_e_all,
        errors_stage1_with_e_all,
        errors_stage12_no_e_no_gl_all,
        errors_stage12_no_e_with_gl_all,
        errors_stage12_with_e_no_gl_all,
        errors_stage12_with_e_with_gl_all,
    ) = all_errors

    results = {
        "Stage I (no E-optimal)": {
            "mean": np.mean(errors_stage1_no_e_all, axis=1).tolist(),
            "ci": [studentized_double_bootstrap(errors_stage1_no_e_all[i]) for i in range(len(errors_stage1_no_e_all))],
            "raw": {str(N): errors_stage1_no_e_all[i] for i, N in enumerate(Ns[: len(errors_stage1_no_e_all)])},
        },
        "Stage I (with E-optimal)": {
            "mean": np.mean(errors_stage1_with_e_all, axis=1).tolist(),
            "ci": [studentized_double_bootstrap(errors_stage1_with_e_all[i]) for i in range(len(errors_stage1_with_e_all))],
            "raw": {str(N): errors_stage1_with_e_all[i] for i, N in enumerate(Ns[: len(errors_stage1_with_e_all)])},
        },
        "Stage I+II (no E, no GL)": {
            "mean": np.mean(errors_stage12_no_e_no_gl_all, axis=1).tolist(),
            "ci": [studentized_double_bootstrap(errors_stage12_no_e_no_gl_all[i]) for i in range(len(errors_stage12_no_e_no_gl_all))],
            "raw": {str(N): errors_stage12_no_e_no_gl_all[i] for i, N in enumerate(Ns[: len(errors_stage12_no_e_no_gl_all)])},
        },
        "Stage I+II (no E, with GL)": {
            "mean": np.mean(errors_stage12_no_e_with_gl_all, axis=1).tolist(),
            "ci": [studentized_double_bootstrap(errors_stage12_no_e_with_gl_all[i]) for i in range(len(errors_stage12_no_e_with_gl_all))],
            "raw": {str(N): errors_stage12_no_e_with_gl_all[i] for i, N in enumerate(Ns[: len(errors_stage12_no_e_with_gl_all)])},
        },
        "Stage I+II (with E, no GL)": {
            "mean": np.mean(errors_stage12_with_e_no_gl_all, axis=1).tolist(),
            "ci": [studentized_double_bootstrap(errors_stage12_with_e_no_gl_all[i]) for i in range(len(errors_stage12_with_e_no_gl_all))],
            "raw": {str(N): errors_stage12_with_e_no_gl_all[i] for i, N in enumerate(Ns[: len(errors_stage12_with_e_no_gl_all)])},
        },
        "Stage I+II (with E, with GL)": {
            "mean": np.mean(errors_stage12_with_e_with_gl_all, axis=1).tolist(),
            "ci": [studentized_double_bootstrap(errors_stage12_with_e_with_gl_all[i]) for i in range(len(errors_stage12_with_e_with_gl_all))],
            "raw": {str(N): errors_stage12_with_e_with_gl_all[i] for i, N in enumerate(Ns[: len(errors_stage12_with_e_with_gl_all)])},
        },
        "metadata": {"mode": mode, "model": model, "Ns": Ns, "params": params, "timestamp": datetime.now().isoformat()},
    }
    if model == "bernoulli":
        results["BMF"] = {
            "mean": np.mean(errors_bmf_all, axis=1).tolist(),
            "ci": [studentized_double_bootstrap(errors_bmf_all[i]) for i in range(len(errors_bmf_all))],
            "raw": {str(N): errors_bmf_all[i] for i, N in enumerate(Ns[: len(errors_bmf_all)])},
        }

    with open(result_file("Fig1", mode, model), "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", result_file("Fig1", mode, model))


def main():
    parser = argparse.ArgumentParser(description="Run Figure 1 experiments")
    parser.add_argument("--mode", type=str, choices=["completion", "recovery", "hard"], required=True)
    parser.add_argument("--model", type=str, choices=["bernoulli", "poisson"], default="bernoulli")
    parser.add_argument("--d1", type=int, default=DEFAULT_PARAMS["d1"])
    parser.add_argument("--d2", type=int, default=DEFAULT_PARAMS["d2"])
    parser.add_argument("--r", type=int, default=DEFAULT_PARAMS["r"])
    parser.add_argument("--num_repeats", type=int, default=DEFAULT_PARAMS["num_repeats"])
    args = parser.parse_args()

    logger = setup_logging(args.mode, "fig1", model=args.model)
    params = DEFAULT_PARAMS.copy()
    params.update({"model": args.model, "d1": args.d1, "d2": args.d2, "r": args.r, "num_repeats": args.num_repeats})

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
        logger,
    )
    save_results(all_errors, params["Ns"], args.mode, args.model, params, logger)


if __name__ == "__main__":
    main()


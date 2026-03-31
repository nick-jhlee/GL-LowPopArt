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

METRIC_SPECS = [
    ("error_stage1_no_e", "Stage I (no E-optimal)"),
    ("error_stage1_with_e", "Stage I (with E-optimal)"),
    ("error_stage12_no_e_no_gl", "Stage I+II (no E, no GL)"),
    ("error_stage12_no_e_with_gl", "Stage I+II (no E, with GL)"),
    ("error_stage12_with_e_no_gl", "Stage I+II (with E, no GL)"),
    ("error_stage12_with_e_with_gl", "Stage I+II (with E, with GL)"),
]
BERNOULLI_ONLY_METRIC = ("error_bmf", "BMF")
METHOD_ALIASES = {
    "U": "error_stage1_no_e",
    "E": "error_stage1_with_e",
    "U+U": "error_stage12_no_e_no_gl",
    "E+U": "error_stage12_with_e_no_gl",
    "U+GL": "error_stage12_no_e_with_gl",
    "E+GL": "error_stage12_with_e_with_gl",
    "BMF": "error_bmf",
}
METRIC_TO_ALIAS = {metric_key: alias for alias, metric_key in METHOD_ALIASES.items()}
VALID_METHOD_TOKENS = tuple(METHOD_ALIASES.keys())


def active_metric_specs(model, enabled_metric_keys=None):
    specs = METRIC_SPECS + [BERNOULLI_ONLY_METRIC] if model == "bernoulli" else METRIC_SPECS
    if enabled_metric_keys is None:
        return specs
    enabled_set = set(enabled_metric_keys)
    return [spec for spec in specs if spec[0] in enabled_set]


def empty_metric_store(model, enabled_metric_keys=None):
    return {metric_key: [] for metric_key, _ in active_metric_specs(model, enabled_metric_keys)}


def parse_enabled_metric_keys(model, methods_arg):
    default_keys = [metric_key for metric_key, _ in active_metric_specs(model)]
    if not methods_arg:
        return default_keys

    selected_tokens = [token.strip() for token in methods_arg.split(",") if token.strip()]
    unknown_tokens = [token for token in selected_tokens if token not in METHOD_ALIASES]
    if unknown_tokens:
        raise ValueError(f"Unknown methods: {unknown_tokens}. Valid choices: {list(VALID_METHOD_TOKENS)}")
    if model != "bernoulli" and "BMF" in selected_tokens:
        raise ValueError("Method 'BMF' is only available for bernoulli model.")

    selected_keys = {METHOD_ALIASES[token] for token in selected_tokens}
    ordered_keys = [metric_key for metric_key in default_keys if metric_key in selected_keys]
    if not ordered_keys:
        raise ValueError("No valid methods selected.")
    return ordered_keys


def append_metrics(target_store, source_store):
    for metric_key in target_store:
        target_store[metric_key].append(source_store[metric_key])


def build_results_payload(metric_store, Ns, model, metadata, include_ci=False, enabled_metric_keys=None):
    results = {}
    for metric_key, metric_label in active_metric_specs(model, enabled_metric_keys):
        metric_data = metric_store[metric_key]
        means = [float(np.mean(values)) if len(values) > 0 else np.nan for values in metric_data]
        entry = {
            "mean": means,
            "raw": {str(N): metric_data[i] for i, N in enumerate(Ns[: len(metric_data)])},
        }
        if include_ci:
            entry["ci"] = [studentized_double_bootstrap(metric_data[i]) for i in range(len(metric_data))]
        results[metric_label] = entry
    results["metadata"] = metadata
    return results


def run_single_repetition(args):
    run_idx, N, mode, model, d1, d2, r, K, delta, c_lambda, c_nu, Rmax, instance, enabled_metric_keys = args
    arm_set, Theta_star = instance
    env = build_env(arm_set, Theta_star, model=model)
    enabled_set = set(enabled_metric_keys)

    nuc_coef = np.sqrt(c_lambda * np.log((d1 + d2) / delta) / N)
    need_stage1_no_e = "error_stage1_no_e" in enabled_set or "error_bmf" in enabled_set
    if need_stage1_no_e:
        error_stage1_no_e, X1, y1 = run_stage1(env, N, d1, d2, nuc_coef, False)
    else:
        error_stage1_no_e, X1, y1 = np.nan, None, None

    error_stage1_with_e = run_stage1(env, N, d1, d2, nuc_coef, True)[0] if "error_stage1_with_e" in enabled_set else np.nan
    error_bmf = run_bmf(env, d1, r, X1, y1) if "error_bmf" in enabled_set else np.nan

    N1 = N // 4
    N2 = N - N1
    nuc_coef = np.sqrt(c_lambda * np.log((d1 + d2) / delta) / N1)
    stage12_flags = {
        "error_stage12_no_e_no_gl": (False, False),
        "error_stage12_no_e_with_gl": (False, True),
        "error_stage12_with_e_no_gl": (True, False),
        "error_stage12_with_e_with_gl": (True, True),
    }
    stage12_results = {}
    for metric_key, flags in stage12_flags.items():
        if metric_key in enabled_set:
            use_e, use_gl = flags
            stage12_results[metric_key] = run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, use_e, use_gl)
        else:
            stage12_results[metric_key] = np.nan

    results = {
        "error_bmf": error_bmf,
        "error_stage1_no_e": error_stage1_no_e,
        "error_stage1_with_e": error_stage1_with_e,
        **stage12_results,
    }
    return {metric_key: results[metric_key] for metric_key in enabled_metric_keys}


def run_experiment(mode, model, d1, d2, r, K, num_repeats, delta, Ns, c_lambda, c_nu, Rmax, logger, enabled_metric_keys, seed=42):
    metric_store_all = empty_metric_store(model, enabled_metric_keys)

    problem_instances = build_problem_instances(mode, model, d1, d2, r, K, num_repeats, seed=seed)

    pbar = tqdm(Ns, desc="Sample sizes")
    for N in pbar:
        pbar.set_description(f"Processing N={N}")
        logger.info("Processing N=%s", N)

        metric_store_reps = empty_metric_store(model, enabled_metric_keys)

        results_file = result_file("Fig1", mode, model, intermediate=True)
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    current_results = json.load(f)
                key = "BMF" if "BMF" in current_results else "Stage I (no E-optimal)"
                if len(current_results[key]["mean"]) > Ns.index(N):
                    for metric_key, metric_label in active_metric_specs(model, enabled_metric_keys):
                        if metric_label in current_results and "raw" in current_results[metric_label]:
                            metric_store_reps[metric_key] = current_results[metric_label]["raw"][str(N)]
                        else:
                            metric_store_reps[metric_key] = [np.nan] * num_repeats
                    append_metrics(metric_store_all, metric_store_reps)
                    continue
            except Exception as exc:
                logger.warning("Failed to load existing results: %s", exc)

        args_list = [
            (run_idx, N, mode, model, d1, d2, r, K, delta, c_lambda, c_nu, Rmax, problem_instances[run_idx], enabled_metric_keys)
            for run_idx in range(num_repeats)
        ]

        try:
            num_cores = min(num_repeats, max(1, mp.cpu_count() - 1))
            with mp.Pool(processes=num_cores) as pool:
                results = list(tqdm(pool.imap(run_single_repetition, args_list), total=num_repeats, desc=f"Repeats N={N}", leave=False))
            for result in results:
                for metric_key in metric_store_reps:
                    metric_store_reps[metric_key].append(result[metric_key])
        except Exception as exc:
            logger.error("Parallel processing failed for N=%s: %s", N, exc)
            for run_idx in tqdm(range(num_repeats), desc=f"Repeats N={N}", leave=False):
                result = run_single_repetition(args_list[run_idx])
                for metric_key in metric_store_reps:
                    metric_store_reps[metric_key].append(result[metric_key])

        append_metrics(metric_store_all, metric_store_reps)

        first_metric_key = enabled_metric_keys[0]
        current_results = build_results_payload(
            metric_store_all,
            Ns,
            model,
            {
                "mode": mode,
                "model": model,
                "Ns": Ns[: len(metric_store_all[first_metric_key])],
                "methods": [METRIC_TO_ALIAS[metric_key] for metric_key in enabled_metric_keys],
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
            include_ci=False,
            enabled_metric_keys=enabled_metric_keys,
        )
        with open(result_file("Fig1", mode, model, intermediate=True), "w") as f:
            json.dump(current_results, f, indent=2)

    return metric_store_all


def save_results(all_errors, Ns, mode, model, params, logger, enabled_metric_keys):
    results = build_results_payload(
        all_errors,
        Ns,
        model,
        {
            "mode": mode,
            "model": model,
            "Ns": Ns,
            "methods": [METRIC_TO_ALIAS[metric_key] for metric_key in enabled_metric_keys],
            "params": params,
            "timestamp": datetime.now().isoformat(),
        },
        include_ci=True,
        enabled_metric_keys=enabled_metric_keys,
    )

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
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help=f"Comma-separated method aliases from {list(VALID_METHOD_TOKENS)} (e.g., U,U+U)",
    )
    args = parser.parse_args()

    logger = setup_logging(args.mode, "fig1", model=args.model)
    enabled_metric_keys = parse_enabled_metric_keys(args.model, args.methods)
    params = DEFAULT_PARAMS.copy()
    params.update(
        {
            "model": args.model,
            "d1": args.d1,
            "d2": args.d2,
            "r": args.r,
            "num_repeats": args.num_repeats,
            "methods": [METRIC_TO_ALIAS[metric_key] for metric_key in enabled_metric_keys],
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
        logger,
        enabled_metric_keys,
    )
    save_results(all_errors, params["Ns"], args.mode, args.model, params, logger, enabled_metric_keys)


if __name__ == "__main__":
    main()


"""Configuration and artifact path helpers."""

import logging
import os
from datetime import datetime

DEFAULT_PARAMS = {
    "K": 150,
    "delta": 0.001,
    "Ns": [10000, 20000, 30000, 40000, 50000],
    "c_lambda": 1,
    "c_nu": 1,
    "Rmax": 1 / 4,
    "d1": 3,
    "d2": 3,
    "r": 1,
    "num_repeats": 60,
}

RESULTS_DIR = "results"
RESULTS_JSON_DIR = os.path.join(RESULTS_DIR, "json")
RESULTS_FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
PROBLEM_INSTANCES_DIR = os.path.join(RESULTS_DIR, "problem_instances")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

for directory in [
    RESULTS_DIR,
    RESULTS_JSON_DIR,
    RESULTS_FIGURES_DIR,
    PROBLEM_INSTANCES_DIR,
    LOGS_DIR,
]:
    os.makedirs(directory, exist_ok=True)


def result_file(fig: str, mode: str, model: str, intermediate: bool = False) -> str:
    stem = f"{fig}_{mode}" if model == "bernoulli" else f"{fig}_{mode}_{model}"
    if intermediate:
        stem = f"{stem}_intermediate"
    return os.path.join(RESULTS_JSON_DIR, f"{stem}.json")


def figure_file(fig_name: str, model: str, ext: str) -> str:
    suffix = "" if model == "bernoulli" else f"_{model}"
    return os.path.join(RESULTS_FIGURES_DIR, f"{fig_name}{suffix}.{ext}")


def setup_logging(mode: str, fig: str = "fig1", model: str = "bernoulli") -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "" if model == "bernoulli" else f"_{model}"
    log_file = os.path.join(LOGS_DIR, f"{fig}_{mode}{suffix}_{timestamp}.log")

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    logging.info("Starting new experiment: fig=%s, mode=%s, model=%s", fig, mode, model)
    return root_logger


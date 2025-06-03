# Standard library imports
import argparse
import json
import os
import warnings
from datetime import datetime
from math import floor
from typing import Dict, List, Tuple, Any

# Suppress CVXPY warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='cvxpy')

# Third-party imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

# Local imports
from burer_monteiro import *
from config import DEFAULT_PARAMS, PROBLEM_INSTANCES_DIR, RESULTS_DIR, setup_logging
from gl_lowpopart import *
from problems import *
from utils import studentized_double_bootstrap

def save_problem_instance(arm_set: List[np.ndarray], Theta_star: np.ndarray, mode: str, 
                         run_idx: int, save_dir: str = 'problem_instances', filename: str = None) -> None:
    """
    Save problem instance (arm set and Theta_star) to HDF5 file.
    
    Args:
        arm_set: List of arm matrices
        Theta_star: Target matrix
        mode: Experiment mode ('completion', 'recovery', or 'hard')
        run_idx: Run index
        save_dir: Directory to save the instance
        filename: Optional filename to save the instance
    """
    try:
        os.makedirs(save_dir, exist_ok=True)
        if filename:
            filepath = os.path.join(save_dir, filename)
        else:
            filepath = os.path.join(save_dir, f'{mode}_run{run_idx}_instance.h5')
        with h5py.File(filepath, 'w') as f:
            # Save arm set
            arm_set_array = np.array([arm.flatten('F') for arm in arm_set])
            f.create_dataset('arm_set', data=arm_set_array)
            # Save Theta_star
            f.create_dataset('Theta_star', data=Theta_star)
            # Save metadata
            f.attrs['mode'] = mode
            f.attrs['run_idx'] = run_idx
            f.attrs['timestamp'] = datetime.now().isoformat()
    except Exception as e:
        raise RuntimeError(f"Failed to save problem instance: {str(e)}")

def load_problem_instance(mode: str, run_idx: int, 
                         save_dir: str = 'problem_instances', filename: str = None) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Load problem instance from HDF5 file.
    
    Args:
        mode: Experiment mode ('completion', 'recovery', or 'hard')
        run_idx: Run index
        save_dir: Directory containing the instance file
        filename: Optional filename to load the instance
        
    Returns:
        Tuple of (arm_set, Theta_star)
    """
    try:
        if filename:
            filepath = os.path.join(save_dir, filename)
        else:
            filepath = os.path.join(save_dir, f'{mode}_run{run_idx}_instance.h5')
        with h5py.File(filepath, 'r') as f:
            arm_set_array = f['arm_set'][:]
            Theta_star = f['Theta_star'][:]
            # Reshape arm set back to original format
            arm_set = [arm.reshape(Theta_star.shape, order='F') for arm in arm_set_array]
        return arm_set, Theta_star
    except Exception as e:
        raise RuntimeError(f"Failed to load problem instance: {str(e)}")

def run_bmf(env: Any, d1: int, r: int, X1: np.ndarray, y1: np.ndarray) -> float:
    """
    Run BMF evaluation.
    
    Args:
        env: Environment instance
        d1: First dimension
        r: Rank
        X1: Feature matrix
        y1: Response vector
        
    Returns:
        Nuclear norm error
    """
    X_bmf, y_bmf = X1, y1
    Theta_BMF = Burer_Monteiro(d1, r, X_bmf, y_bmf)
    return np.linalg.norm(Theta_BMF - env.Theta_star, 'nuc')

def run_stage1(env: Any, N: int, d1: int, d2: int, nuc_coef: float, 
               e_optimal: bool) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Run Stage I (Nuclear norm MLE) evaluation.
    
    Args:
        env: Environment instance
        N: Sample size
        d1: First dimension
        d2: Second dimension
        nuc_coef: Nuclear norm coefficient
        e_optimal: Whether to use E-optimal design
        
    Returns:
        Tuple of (nuclear norm error, feature matrix, response vector)
    """
    Theta0, X1, y1 = nuc_norm_MLE(env, N, d1, d2, nuc_coef, E_optimal=e_optimal)
    return np.linalg.norm(Theta0 - env.Theta_star, 'nuc'), X1, y1

def run_stage1_2(env: Any, N1: int, N2: int, d1: int, d2: int, nuc_coef: float,
                 c_nu: float, delta: float, e_optimal: bool, gl_optimal: bool) -> float:
    """
    Run Stage I + II evaluation.
    
    Args:
        env: Environment instance
        N1: First stage sample size
        N2: Second stage sample size
        d1: First dimension
        d2: Second dimension
        nuc_coef: Nuclear norm coefficient
        c_nu: Nu coefficient
        delta: Confidence parameter
        e_optimal: Whether to use E-optimal design
        gl_optimal: Whether to use GL-optimal design
        
    Returns:
        Nuclear norm error
    """
    # Stage I
    Theta0, _, _ = nuc_norm_MLE(env, N1, d1, d2, nuc_coef, E_optimal=e_optimal)
    # Stage II
    Theta = GL_LowPopArt(env, N2, d1, d2, delta, Theta0, c_nu, GL_optimal=gl_optimal)
    return np.linalg.norm(Theta - env.Theta_star, 'nuc')

def run_single_repetition(args: Tuple[int, int, str, int, int, int, int, float, float, float, float, Tuple[List[np.ndarray], np.ndarray]]) -> Dict[str, float]:
    """
    Run a single repetition of the experiment.
    
    Args:
        args: Tuple containing (run_idx, N, mode, d1, d2, r, K, delta, c_lambda, c_nu, Rmax, problem_instance)
        
    Returns:
        Dictionary containing errors for all algorithms
    """
    run_idx, N, mode, d1, d2, r, K, delta, c_lambda, c_nu, Rmax, (arm_set, Theta_star) = args
    
    env = OneBitCompletion(arm_set, Theta_star)
    env.Theta_star = Theta_star

    # Run all algorithms
    
    # Stage I
    nuc_coef = c_lambda * np.sqrt(8 * Rmax * np.log((d1 + d2) / delta) / N)
    error_stage1_no_e, X1, y1 = run_stage1(env, N, d1, d2, nuc_coef, False)
    error_stage1_with_e = run_stage1(env, N, d1, d2, nuc_coef, True)[0]
    
    # BMF
    error_bmf = run_bmf(env, d1, r, X1, y1)
    
    # Stage I+II
    # N1 = 2 * floor(np.sqrt(N))
    N1 = N // 2
    N2 = N - N1
    nuc_coef = c_lambda * np.sqrt(8 * Rmax * np.log((d1 + d2) / delta) / N1)
    error_stage12_no_e_no_gl = run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, False, False)
    error_stage12_no_e_with_gl = run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, False, True)
    error_stage12_with_e_no_gl = run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, True, False)
    error_stage12_with_e_with_gl = run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, True, True)
    
    return {
        'error_bmf': error_bmf,
        'error_stage1_no_e': error_stage1_no_e,
        'error_stage1_with_e': error_stage1_with_e,
        'error_stage12_no_e_no_gl': error_stage12_no_e_no_gl,
        'error_stage12_no_e_with_gl': error_stage12_no_e_with_gl,
        'error_stage12_with_e_no_gl': error_stage12_with_e_no_gl,
        'error_stage12_with_e_with_gl': error_stage12_with_e_with_gl
    }

def run_experiment(mode: str, d1: int, d2: int, r: int, K: int, num_repeats: int,
                  delta: float, Ns: List[int], c_lambda: float, c_nu: float,
                  Rmax: float, logger: logging.Logger, seed: int = 42) -> Tuple[List[List[float]], ...]:
    """
    Run the main experiment comparing all algorithms.
    
    Args:
        mode: Experiment mode ('completion', 'recovery', or 'hard')
            - 'completion': Standard matrix completion with single-entry observations
            - 'recovery': Matrix recovery with random measurements
            - 'hard': Hard instance from Jang et al. (2024)
        d1: First dimension
        d2: Second dimension
        r: Rank
        K: Number of arms
        num_repeats: Number of experiment repeats
        delta: Confidence parameter
        Ns: List of sample sizes
        c_lambda: Scaling for lambda in Stage I
        c_nu: Scaling for nu in Stage II
        Rmax: Maximum reward
        logger: Logger instance
        seed: Random seed for environment generation only
    """
    # Initialize error arrays for all algorithms
    errors_bmf_all = []
    errors_stage1_no_e_all = []
    errors_stage1_with_e_all = []
    errors_stage12_no_e_no_gl_all = []
    errors_stage12_no_e_with_gl_all = []
    errors_stage12_with_e_no_gl_all = []
    errors_stage12_with_e_with_gl_all = []

    # First, generate or load all problem instances with fixed seed
    problem_instances = []
    rng = np.random.RandomState(seed)
    
    # Generate arm set once for all runs
    instance_file = f'{mode}_arm_set.h5'
    if os.path.exists(os.path.join(PROBLEM_INSTANCES_DIR, instance_file)):
        arm_set, _ = load_problem_instance(mode, 0, PROBLEM_INSTANCES_DIR, filename=instance_file)
    else:
        arm_set = generate_arm_set(d1, d2, K, mode=mode, rng=rng)
        # Save arm set separately
        save_problem_instance(arm_set, np.zeros((d1, d2)), mode, 0, PROBLEM_INSTANCES_DIR, filename=instance_file)
    
    # Generate Theta_star for each run
    for run_idx in range(num_repeats):
        instance_file = f'{mode}_run{run_idx}_theta.h5'
        if os.path.exists(os.path.join(PROBLEM_INSTANCES_DIR, instance_file)):
            _, Theta_star = load_problem_instance(mode, run_idx, PROBLEM_INSTANCES_DIR, filename=instance_file)
        else:
            # Use the seeded RNG for environment generation only
            Theta_star = generate_Theta_star(d1, d2, r, rng=rng)
            # Save only Theta_star
            save_problem_instance([], Theta_star, mode, run_idx, PROBLEM_INSTANCES_DIR, filename=instance_file)
        problem_instances.append((arm_set, Theta_star))

    # Create progress bar for sample sizes
    pbar = tqdm(Ns, desc="Sample sizes")
    for N in pbar:
        pbar.set_description(f"Processing N={N}")
        logger.info(f"Processing N={N}")
        
        # Initialize arrays for this N
        errors_bmf_reps = []
        errors_stage1_no_e_reps = []
        errors_stage1_with_e_reps = []
        errors_stage12_no_e_no_gl_reps = []
        errors_stage12_no_e_with_gl_reps = []
        errors_stage12_with_e_no_gl_reps = []
        errors_stage12_with_e_with_gl_reps = []

        # Check if results exist for this N
        results_file = f'{RESULTS_DIR}/Fig1_{mode}_intermediate.json'
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    current_results = json.load(f)
                    if len(current_results['BMF']['mean']) > Ns.index(N):
                        logger.info(f"Loading existing results for N={N}")
                        # Handle both old and new result formats
                        if 'raw' in current_results['BMF']:
                            # New format with raw data
                            errors_bmf_reps = current_results['BMF']['raw'][str(N)]
                            errors_stage1_no_e_reps = current_results['Stage I (no E-optimal)']['raw'][str(N)]
                            errors_stage1_with_e_reps = current_results['Stage I (with E-optimal)']['raw'][str(N)]
                            errors_stage12_no_e_no_gl_reps = current_results['Stage I+II (no E, no GL)']['raw'][str(N)]
                            errors_stage12_no_e_with_gl_reps = current_results['Stage I+II (no E, with GL)']['raw'][str(N)]
                            errors_stage12_with_e_no_gl_reps = current_results['Stage I+II (with E, no GL)']['raw'][str(N)]
                            errors_stage12_with_e_with_gl_reps = current_results['Stage I+II (with E, with GL)']['raw'][str(N)]
                        else:
                            # Old format - we'll need to run the experiment again
                            logger.info("Found old format results - will run experiment again")
                            continue
                        
                        # Store results for this N
                        errors_bmf_all.append(errors_bmf_reps)
                        errors_stage1_no_e_all.append(errors_stage1_no_e_reps)
                        errors_stage1_with_e_all.append(errors_stage1_with_e_reps)
                        errors_stage12_no_e_no_gl_all.append(errors_stage12_no_e_no_gl_reps)
                        errors_stage12_no_e_with_gl_all.append(errors_stage12_no_e_with_gl_reps)
                        errors_stage12_with_e_no_gl_all.append(errors_stage12_with_e_no_gl_reps)
                        errors_stage12_with_e_with_gl_all.append(errors_stage12_with_e_with_gl_reps)
                        continue
            except Exception as e:
                logger.warning(f"Failed to load existing results: {str(e)}")
                logger.info("Will run experiment again")

        # Prepare arguments for parallel processing
        args_list = [(run_idx, N, mode, d1, d2, r, K, delta, c_lambda, c_nu, Rmax, problem_instances[run_idx]) 
                    for run_idx in range(num_repeats)]

        # Run repeats in parallel
        try:
            # Use number of CPU cores minus 1 to leave one core free for system tasks
            # num_cores = min(num_repeats, max(1, mp.cpu_count() - 1))  # Use as many cores as repeats
            num_cores = 50
            # num_cores = 16   # use only 2 cores
            logger.info(f"Running {num_repeats} repeats in parallel using {num_cores} cores")
            
            with mp.Pool(processes=num_cores) as pool:
                results = list(tqdm(
                    pool.imap(run_single_repetition, args_list),
                    total=num_repeats,
                    desc=f"Repeats for N={N}",
                    leave=False
                ))

            # Process results
            for result in results:
                errors_bmf_reps.append(result['error_bmf'])
                errors_stage1_no_e_reps.append(result['error_stage1_no_e'])
                errors_stage1_with_e_reps.append(result['error_stage1_with_e'])
                errors_stage12_no_e_no_gl_reps.append(result['error_stage12_no_e_no_gl'])
                errors_stage12_no_e_with_gl_reps.append(result['error_stage12_no_e_with_gl'])
                errors_stage12_with_e_no_gl_reps.append(result['error_stage12_with_e_no_gl'])
                errors_stage12_with_e_with_gl_reps.append(result['error_stage12_with_e_with_gl'])

        except Exception as e:
            logger.error(f"Error in parallel processing for N={N}: {str(e)}")
            # Fall back to sequential processing if parallel processing fails
            logger.info("Falling back to sequential processing")
            for run_idx in tqdm(range(num_repeats), desc=f"Repeats for N={N}", leave=False):
                try:
                    result = run_single_repetition((run_idx, N, mode, d1, d2, r, K, delta, c_lambda, c_nu, Rmax, problem_instances[run_idx]))
                    errors_bmf_reps.append(result['error_bmf'])
                    errors_stage1_no_e_reps.append(result['error_stage1_no_e'])
                    errors_stage1_with_e_reps.append(result['error_stage1_with_e'])
                    errors_stage12_no_e_no_gl_reps.append(result['error_stage12_no_e_no_gl'])
                    errors_stage12_no_e_with_gl_reps.append(result['error_stage12_no_e_with_gl'])
                    errors_stage12_with_e_no_gl_reps.append(result['error_stage12_with_e_no_gl'])
                    errors_stage12_with_e_with_gl_reps.append(result['error_stage12_with_e_with_gl'])
                except Exception as e:
                    logger.error(f"Error in run {run_idx} for N={N}: {str(e)}")
                    continue

        # Store results for this N
        errors_bmf_all.append(errors_bmf_reps)
        errors_stage1_no_e_all.append(errors_stage1_no_e_reps)
        errors_stage1_with_e_all.append(errors_stage1_with_e_reps)
        errors_stage12_no_e_no_gl_all.append(errors_stage12_no_e_no_gl_reps)
        errors_stage12_no_e_with_gl_all.append(errors_stage12_no_e_with_gl_reps)
        errors_stage12_with_e_no_gl_all.append(errors_stage12_with_e_no_gl_reps)
        errors_stage12_with_e_with_gl_all.append(errors_stage12_with_e_with_gl_reps)

        # Log detailed results for this N
        logger.info(f"N={N} completed with {num_repeats} repeats")
        logger.info(f"BMF mean error: {np.mean(errors_bmf_reps):.4f}")
        logger.info(f"Stage I (no E) mean error: {np.mean(errors_stage1_no_e_reps):.4f}")
        logger.info(f"Stage I (with E) mean error: {np.mean(errors_stage1_with_e_reps):.4f}")
        logger.info(f"Stage I+II (no E, no GL) mean error: {np.mean(errors_stage12_no_e_no_gl_reps):.4f}")
        logger.info(f"Stage I+II (no E, with GL) mean error: {np.mean(errors_stage12_no_e_with_gl_reps):.4f}")
        logger.info(f"Stage I+II (with E, no GL) mean error: {np.mean(errors_stage12_with_e_no_gl_reps):.4f}")
        logger.info(f"Stage I+II (with E, with GL) mean error: {np.mean(errors_stage12_with_e_with_gl_reps):.4f}")

        # Save intermediate results
        try:
            current_results = {
                'BMF': {
                    'mean': np.mean(errors_bmf_all, axis=1).tolist(),
                    'raw': {str(N): errors_bmf_all[i] for i, N in enumerate(Ns[:len(errors_bmf_all)])}
                },
                'Stage I (no E-optimal)': {
                    'mean': np.mean(errors_stage1_no_e_all, axis=1).tolist(),
                    'raw': {str(N): errors_stage1_no_e_all[i] for i, N in enumerate(Ns[:len(errors_stage1_no_e_all)])}
                },
                'Stage I (with E-optimal)': {
                    'mean': np.mean(errors_stage1_with_e_all, axis=1).tolist(),
                    'raw': {str(N): errors_stage1_with_e_all[i] for i, N in enumerate(Ns[:len(errors_stage1_with_e_all)])}
                },
                'Stage I+II (no E, no GL)': {
                    'mean': np.mean(errors_stage12_no_e_no_gl_all, axis=1).tolist(),
                    'raw': {str(N): errors_stage12_no_e_no_gl_all[i] for i, N in enumerate(Ns[:len(errors_stage12_no_e_no_gl_all)])}
                },
                'Stage I+II (no E, with GL)': {
                    'mean': np.mean(errors_stage12_no_e_with_gl_all, axis=1).tolist(),
                    'raw': {str(N): errors_stage12_no_e_with_gl_all[i] for i, N in enumerate(Ns[:len(errors_stage12_no_e_with_gl_all)])}
                },
                'Stage I+II (with E, no GL)': {
                    'mean': np.mean(errors_stage12_with_e_no_gl_all, axis=1).tolist(),
                    'raw': {str(N): errors_stage12_with_e_no_gl_all[i] for i, N in enumerate(Ns[:len(errors_stage12_with_e_no_gl_all)])}
                },
                'Stage I+II (with E, with GL)': {
                    'mean': np.mean(errors_stage12_with_e_with_gl_all, axis=1).tolist(),
                    'raw': {str(N): errors_stage12_with_e_with_gl_all[i] for i, N in enumerate(Ns[:len(errors_stage12_with_e_with_gl_all)])}
                },
                'metadata': {
                    'mode': mode,
                    'Ns': Ns[:len(errors_bmf_all)],  # Only include Ns that have been processed
                    'params': {
                        'd1': d1,
                        'd2': d2,
                        'r': r,
                        'K': K,
                        'num_repeats': num_repeats,
                        'delta': delta,
                        'c_lambda': c_lambda,
                        'c_nu': c_nu,
                        'Rmax': Rmax
                    },
                    'timestamp': datetime.now().isoformat()
                }
            }

            # Save intermediate results to JSON
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(f'{RESULTS_DIR}/Fig1_{mode}_intermediate.json', 'w') as f:
                json.dump(current_results, f, indent=2)
            logger.info(f"Intermediate results saved to {RESULTS_DIR}/Fig1_{mode}_intermediate.json")
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {str(e)}")

    return (errors_bmf_all, errors_stage1_no_e_all, errors_stage1_with_e_all,
            errors_stage12_no_e_no_gl_all, errors_stage12_no_e_with_gl_all,
            errors_stage12_with_e_no_gl_all, errors_stage12_with_e_with_gl_all)

def save_results(all_errors: Tuple[List[List[float]], ...], Ns: List[int], 
                mode: str, params: Dict[str, Any], logger: logging.Logger) -> None:
    """
    Save results to JSON file.
    
    Args:
        all_errors: Tuple of error arrays for all algorithms
        Ns: List of sample sizes
        mode: Experiment mode ('completion' or 'recovery')
        params: Dictionary of experiment parameters
        logger: Logger instance
    """
    # Unpack errors
    (errors_bmf_all, errors_stage1_no_e_all, errors_stage1_with_e_all,
     errors_stage12_no_e_no_gl_all, errors_stage12_no_e_with_gl_all,
     errors_stage12_with_e_no_gl_all, errors_stage12_with_e_with_gl_all) = all_errors

    # Compute means and bootstrapped CIs for each algorithm
    logger.info("Computing bootstrapped confidence intervals...")
    results = {
        'BMF': {
            'mean': np.mean(errors_bmf_all, axis=1).tolist(),
            'ci': [studentized_double_bootstrap(errors_bmf_all[i]) for i in range(len(errors_bmf_all))],
            'raw': {str(N): errors_bmf_all[i] for i, N in enumerate(Ns[:len(errors_bmf_all)])}
        },
        'Stage I (no E-optimal)': {
            'mean': np.mean(errors_stage1_no_e_all, axis=1).tolist(),
            'ci': [studentized_double_bootstrap(errors_stage1_no_e_all[i]) for i in range(len(errors_stage1_no_e_all))],
            'raw': {str(N): errors_stage1_no_e_all[i] for i, N in enumerate(Ns[:len(errors_stage1_no_e_all)])}
        },
        'Stage I (with E-optimal)': {
            'mean': np.mean(errors_stage1_with_e_all, axis=1).tolist(),
            'ci': [studentized_double_bootstrap(errors_stage1_with_e_all[i]) for i in range(len(errors_stage1_with_e_all))],
            'raw': {str(N): errors_stage1_with_e_all[i] for i, N in enumerate(Ns[:len(errors_stage1_with_e_all)])}
        },
        'Stage I+II (no E, no GL)': {
            'mean': np.mean(errors_stage12_no_e_no_gl_all, axis=1).tolist(),
            'ci': [studentized_double_bootstrap(errors_stage12_no_e_no_gl_all[i]) for i in range(len(errors_stage12_no_e_no_gl_all))],
            'raw': {str(N): errors_stage12_no_e_no_gl_all[i] for i, N in enumerate(Ns[:len(errors_stage12_no_e_no_gl_all)])}
        },
        'Stage I+II (no E, with GL)': {
            'mean': np.mean(errors_stage12_no_e_with_gl_all, axis=1).tolist(),
            'ci': [studentized_double_bootstrap(errors_stage12_no_e_with_gl_all[i]) for i in range(len(errors_stage12_no_e_with_gl_all))],
            'raw': {str(N): errors_stage12_no_e_with_gl_all[i] for i, N in enumerate(Ns[:len(errors_stage12_no_e_with_gl_all)])}
        },
        'Stage I+II (with E, no GL)': {
            'mean': np.mean(errors_stage12_with_e_no_gl_all, axis=1).tolist(),
            'ci': [studentized_double_bootstrap(errors_stage12_with_e_no_gl_all[i]) for i in range(len(errors_stage12_with_e_no_gl_all))],
            'raw': {str(N): errors_stage12_with_e_no_gl_all[i] for i, N in enumerate(Ns[:len(errors_stage12_with_e_no_gl_all)])}
        },
        'Stage I+II (with E, with GL)': {
            'mean': np.mean(errors_stage12_with_e_with_gl_all, axis=1).tolist(),
            'ci': [studentized_double_bootstrap(errors_stage12_with_e_with_gl_all[i]) for i in range(len(errors_stage12_with_e_with_gl_all))],
            'raw': {str(N): errors_stage12_with_e_with_gl_all[i] for i, N in enumerate(Ns[:len(errors_stage12_with_e_with_gl_all)])}
        }
    }

    # Add metadata
    results['metadata'] = {
        'mode': mode,
        'Ns': Ns,
        'params': params,
        'timestamp': datetime.now().isoformat()
    }

    try:
        # Save results to JSON
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(f'{RESULTS_DIR}/Fig1_{mode}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {RESULTS_DIR}/Fig1_{mode}.json")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")

def main() -> None:
    """Main entry point for the experiment."""
    parser = argparse.ArgumentParser(description='Run 1-bit matrix experiments')
    parser.add_argument('--mode', type=str, choices=['completion', 'recovery', 'hard'], required=True,
                      help='Mode of operation: completion or recovery or hard')
    parser.add_argument('--d1', type=int, default=DEFAULT_PARAMS['d1'], help='First dimension')
    parser.add_argument('--d2', type=int, default=DEFAULT_PARAMS['d2'], help='Second dimension')
    parser.add_argument('--r', type=int, default=DEFAULT_PARAMS['r'], help='Rank')
    parser.add_argument('--num_repeats', type=int, default=DEFAULT_PARAMS['num_repeats'],
                      help='Number of experiment repeats')
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.mode, 'fig1')
    logger.info(f"Starting experiment with parameters: {args}")

    # Use default parameters from config
    params = DEFAULT_PARAMS.copy()
    # Update with command line arguments
    params.update({
        'd1': args.d1,
        'd2': args.d2,
        'r': args.r,
        'num_repeats': args.num_repeats
    })

    try:
        # Run experiment with all algorithms
        all_errors = run_experiment(
            args.mode, args.d1, args.d2, args.r, params['K'], params['num_repeats'],
            params['delta'], params['Ns'], params['c_lambda'], params['c_nu'], params['Rmax'],
            logger
        )
        
        # Save results and generate plot
        save_results(all_errors, params['Ns'], args.mode, params, logger)
        logger.info("Experiment completed successfully")
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == '__main__':
    main() 
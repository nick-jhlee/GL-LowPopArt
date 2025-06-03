from problems import *
from gl_lowpopart import *
from burer_monteiro import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from config import DEFAULT_PARAMS, setup_logging, PROBLEM_INSTANCES_DIR, RESULTS_DIR
import os
import multiprocessing as mp
import json
from datetime import datetime
import logging
import argparse
from math import floor

def run_single_repetition(args: tuple) -> dict:
    """Run a single repetition of the experiment."""
    run_idx, N, mode, d1, d2, r, K, delta, c_lambda, c_nu, Rmax, (arm_set, Theta_star) = args
    
    # Initialize all errors as None
    error_uniform = None
    error_e_optimal = None
    error_zero = None
    error_random = None
    
    try:
        # First check Fig2 intermediate results
        fig2_file = f'{RESULTS_DIR}/Fig2_{mode}_intermediate.json'
        if os.path.exists(fig2_file):
            try:
                with open(fig2_file, 'r') as f:
                    fig2_results = json.load(f)
                if str(N) in fig2_results['uniform']['raw']:
                    error_uniform = fig2_results['uniform']['raw'][str(N)][run_idx]
                if str(N) in fig2_results['e_optimal']['raw']:
                    error_e_optimal = fig2_results['e_optimal']['raw'][str(N)][run_idx]
                if str(N) in fig2_results['zero']['raw']:
                    error_zero = fig2_results['zero']['raw'][str(N)][run_idx]
                if str(N) in fig2_results['random']['raw']:
                    error_random = fig2_results['random']['raw'][str(N)][run_idx]
            except Exception as e:
                logging.warning(f"Failed to load Fig2 intermediate results: {str(e)}")
        
        # If MLE results not found in Fig2, check Fig1
        if error_uniform is None or error_e_optimal is None:
            fig1_file = f'{RESULTS_DIR}/Fig1_{mode}_intermediate.json'
            if os.path.exists(fig1_file):
                try:
                    with open(fig1_file, 'r') as f:
                        fig1_results = json.load(f)
                    if error_uniform is None and str(N) in fig1_results['Stage I+II (no E, with GL)']['raw']:
                        error_uniform = fig1_results['Stage I+II (no E, with GL)']['raw'][str(N)][run_idx]
                    if error_e_optimal is None and str(N) in fig1_results['Stage I+II (with E, with GL)']['raw']:
                        error_e_optimal = fig1_results['Stage I+II (with E, with GL)']['raw'][str(N)][run_idx]
                except Exception as e:
                    logging.warning(f"Failed to load Fig1 intermediate results: {str(e)}")
        
        env = OneBitCompletion(arm_set, Theta_star)
        env.Theta_star = Theta_star

        # N1 = 2 * floor(np.sqrt(N))
        # N2 = N - N1
        N1 = N // 2
        N2 = N - N1
        nuc_coef = c_lambda * np.sqrt(8 * Rmax * np.log((d1 + d2) / delta) / N1)            
        # Uniform-based MLE with GL
        if error_uniform is None:
            try:
                Theta0_uniform, _, _ = nuc_norm_MLE(env, N1, d1, d2, nuc_coef, E_optimal=False)
                Theta_uniform = GL_LowPopArt(env, N2, d1, d2, delta, Theta0_uniform, c_nu)
                error_uniform = float(np.linalg.norm(Theta_uniform - Theta_star, 'nuc'))
                logging.info(f"Computed uniform error for run {run_idx}, N={N}: {error_uniform}")
            except Exception as e:
                logging.error(f"Error computing uniform error for run {run_idx}, N={N}: {str(e)}")
                logging.error(f"Error type: {type(e)}")
                import traceback
                logging.error(traceback.format_exc())
        
        # E-optimal design-based MLE with GL
        if error_e_optimal is None:
            try:
                Theta0_e_optimal, _, _ = nuc_norm_MLE(env, N1, d1, d2, nuc_coef, E_optimal=True)
                Theta_e_optimal = GL_LowPopArt(env, N2, d1, d2, delta, Theta0_e_optimal, c_nu)
                error_e_optimal = float(np.linalg.norm(Theta_e_optimal - Theta_star, 'nuc'))
                logging.info(f"Computed e_optimal error for run {run_idx}, N={N}: {error_e_optimal}")
            except Exception as e:
                logging.error(f"Error computing e_optimal error for run {run_idx}, N={N}: {str(e)}")
                logging.error(f"Error type: {type(e)}")
                import traceback
                logging.error(traceback.format_exc())
        
        # Zero initialization if not found in Fig2
        if error_zero is None:
            try:
                Theta0_zero = np.zeros((d1, d2))
                Theta_zero = GL_LowPopArt(env, N, d1, d2, delta, Theta0_zero, c_nu)
                error_zero = float(np.linalg.norm(Theta_zero - Theta_star, 'nuc'))
                logging.info(f"Computed zero error for run {run_idx}, N={N}: {error_zero}")
            except Exception as e:
                logging.error(f"Error computing zero error for run {run_idx}, N={N}: {str(e)}")
                logging.error(f"Error type: {type(e)}")
                import traceback
                logging.error(traceback.format_exc())
        
        # Random initialization if not found in Fig2
        if error_random is None:
            try:
                Theta0_random = np.random.randn(d1, d2) * 0.1
                Theta_random = GL_LowPopArt(env, N, d1, d2, delta, Theta0_random, c_nu)
                error_random = float(np.linalg.norm(Theta_random - Theta_star, 'nuc'))
                logging.info(f"Computed random error for run {run_idx}, N={N}: {error_random}")
            except Exception as e:
                logging.error(f"Error computing random error for run {run_idx}, N={N}: {str(e)}")
                logging.error(f"Error type: {type(e)}")
                import traceback
                logging.error(traceback.format_exc())
        
        return {
            'error_uniform': error_uniform,
            'error_e_optimal': error_e_optimal,
            'error_zero': error_zero,
            'error_random': error_random
        }
    except Exception as e:
        logging.error(f"Error in run {run_idx} for N={N}: {str(e)}")
        logging.error(f"Error type: {type(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return {
            'error_uniform': None,
            'error_e_optimal': None,
            'error_zero': None,
            'error_random': None
        }

def run_experiment(mode: str, d1: int, d2: int, r: int, K: int, num_repeats: int,
                  delta: float, Ns: list, c_lambda: float, c_nu: float,
                  Rmax: float, seed: int = 42) -> tuple:
    """Run the experiment for a given mode."""
    # Set random seed
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    
    # Generate problem instances
    problem_instances = []
    
    # Generate arm set once for all runs
    instance_file = f'{mode}_arm_set.h5'
    if os.path.exists(os.path.join(PROBLEM_INSTANCES_DIR, instance_file)):
        arm_set, _ = load_problem_instance(mode, 0, PROBLEM_INSTANCES_DIR)
    else:
        arm_set = generate_arm_set(d1, d2, K, mode=mode, rng=rng)
        # Save arm set separately
        save_problem_instance(arm_set, np.zeros((d1, d2)), mode, 0, PROBLEM_INSTANCES_DIR)
    
    # Generate Theta_star for each run
    for run_idx in range(num_repeats):
        instance_file = f'{mode}_run{run_idx}_theta.h5'
        if os.path.exists(os.path.join(PROBLEM_INSTANCES_DIR, instance_file)):
            _, Theta_star = load_problem_instance(mode, run_idx, PROBLEM_INSTANCES_DIR)
        else:
            # Use the seeded RNG for environment generation only
            Theta_star = generate_Theta_star(d1, d2, r, rng=rng)
            # Save only Theta_star
            save_problem_instance([], Theta_star, mode, run_idx, PROBLEM_INSTANCES_DIR)
        problem_instances.append((arm_set, Theta_star))
    
    # Initialize arrays for results
    errors_uniform_all = []
    errors_e_optimal_all = []
    errors_zero_all = []
    errors_random_all = []
    
    # Create progress bar for sample sizes
    pbar = tqdm(Ns, desc="Sample sizes")
    for N in pbar:
        pbar.set_description(f"Processing N={N}")
        logging.info(f"Processing N={N}")
        
        # Initialize arrays for this N
        errors_uniform_reps = []
        errors_e_optimal_reps = []
        errors_zero_reps = []
        errors_random_reps = []
        
        # Check if results exist for this N
        results_file = f'{RESULTS_DIR}/Fig2_{mode}_intermediate.json'
        if os.path.exists(results_file):
            try:
                with open(results_file, 'r') as f:
                    current_results = json.load(f)
                    if len(current_results['uniform']['mean']) > Ns.index(N):
                        logging.info(f"Loading existing results for N={N}")
                        # Handle both old and new result formats
                        if 'raw' in current_results['uniform']:
                            # New format with raw data
                            errors_uniform_reps = current_results['uniform']['raw'][str(N)]
                            errors_e_optimal_reps = current_results['e_optimal']['raw'][str(N)]
                            errors_zero_reps = current_results['zero']['raw'][str(N)]
                            errors_random_reps = current_results['random']['raw'][str(N)]
                        else:
                            # Old format - we'll need to run the experiment again
                            logging.info("Found old format results - will run experiment again")
                            continue
                        
                        # Store results for this N
                        errors_uniform_all.append(errors_uniform_reps)
                        errors_e_optimal_all.append(errors_e_optimal_reps)
                        errors_zero_all.append(errors_zero_reps)
                        errors_random_all.append(errors_random_reps)
                        continue
            except Exception as e:
                logging.warning(f"Failed to load existing results: {str(e)}")
                logging.info("Will run experiment again")
        
        # Prepare arguments for parallel processing
        args_list = [(run_idx, N, mode, d1, d2, r, K, delta, c_lambda, c_nu, Rmax, problem_instances[run_idx]) 
                    for run_idx in range(num_repeats)]
        
        # Run repeats in parallel
        try:
            num_cores = 40
            logging.info(f"Running {num_repeats} repeats in parallel using {num_cores} cores")
            
            with mp.Pool(processes=num_cores) as pool:
                results = list(tqdm(
                    pool.imap(run_single_repetition, args_list),
                    total=num_repeats,
                    desc=f"Repeats for N={N}",
                    leave=False
                ))
            
            # Process results
            for result in results:
                try:
                    # Only log if all errors are None
                    if all(v is None for v in result.values()):
                        logging.warning(f"All errors None in run {run_idx}")
                    
                    errors_uniform_reps.append(result['error_uniform'])
                    errors_e_optimal_reps.append(result['error_e_optimal'])
                    errors_zero_reps.append(result['error_zero'])
                    errors_random_reps.append(result['error_random'])
                except Exception as e:
                    logging.error(f"Run {run_idx} failed: {str(e)}")
                    continue
            
        except Exception as e:
            logging.error(f"Parallel processing failed for N={N}: {str(e)}")
            logging.info("Switching to sequential processing")
            for run_idx in tqdm(range(num_repeats), desc=f"Repeats for N={N}", leave=False):
                try:
                    result = run_single_repetition((run_idx, N, mode, d1, d2, r, K, delta, c_lambda, c_nu, Rmax, problem_instances[run_idx]))
                    if all(v is None for v in result.values()):
                        logging.warning(f"All errors None in run {run_idx}")
                    
                    errors_uniform_reps.append(result['error_uniform'])
                    errors_e_optimal_reps.append(result['error_e_optimal'])
                    errors_zero_reps.append(result['error_zero'])
                    errors_random_reps.append(result['error_random'])
                except Exception as e:
                    logging.error(f"Run {run_idx} failed: {str(e)}")
                    continue
        
        # Store results for this N
        try:
            errors_uniform_all.append(errors_uniform_reps)
            errors_e_optimal_all.append(errors_e_optimal_reps)
            errors_zero_all.append(errors_zero_reps)
            errors_random_all.append(errors_random_reps)
        except Exception as e:
            logging.error(f"Failed to store results for N={N}: {str(e)}")
            raise
        
        # Log summary for this N
        try:
            logging.info(f"N={N} completed: Uniform={np.mean(errors_uniform_reps):.4f}, E-opt={np.mean(errors_e_optimal_reps):.4f}, Zero={np.mean(errors_zero_reps):.4f}, Random={np.mean(errors_random_reps):.4f}")
        except Exception as e:
            logging.error(f"Failed to calculate means for N={N}: {str(e)}")
            raise
        
        # Save intermediate results
        try:
            current_results = {
                'uniform': {
                    'mean': np.mean(errors_uniform_all, axis=1).tolist(),
                    'raw': {str(N): errors_uniform_all[i] for i, N in enumerate(Ns[:len(errors_uniform_all)])}
                },
                'e_optimal': {
                    'mean': np.mean(errors_e_optimal_all, axis=1).tolist(),
                    'raw': {str(N): errors_e_optimal_all[i] for i, N in enumerate(Ns[:len(errors_e_optimal_all)])}
                },
                'zero': {
                    'mean': np.mean(errors_zero_all, axis=1).tolist(),
                    'raw': {str(N): errors_zero_all[i] for i, N in enumerate(Ns[:len(errors_zero_all)])}
                },
                'random': {
                    'mean': np.mean(errors_random_all, axis=1).tolist(),
                    'raw': {str(N): errors_random_all[i] for i, N in enumerate(Ns[:len(errors_random_all)])}
                },
                'metadata': {
                    'mode': mode,
                    'Ns': Ns[:len(errors_uniform_all)],  # Only include Ns that have been processed
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
            with open(f'{RESULTS_DIR}/Fig2_{mode}_intermediate.json', 'w') as f:
                json.dump(current_results, f, indent=2)
            
            # Log the saved results
            logging.info(f"\nSaved intermediate results for N={N}:")
            logging.info(f"Uniform initialization:")
            logging.info(f"  - Current N mean: {np.mean(errors_uniform_reps):.4f}")
            logging.info(f"  - All Ns means: {[f'{m:.4f}' for m in current_results['uniform']['mean']]}")
            logging.info(f"E-optimal initialization:")
            logging.info(f"  - Current N mean: {np.mean(errors_e_optimal_reps):.4f}")
            logging.info(f"  - All Ns means: {[f'{m:.4f}' for m in current_results['e_optimal']['mean']]}")
            logging.info(f"Zero initialization:")
            logging.info(f"  - Current N mean: {np.mean(errors_zero_reps):.4f}")
            logging.info(f"  - All Ns means: {[f'{m:.4f}' for m in current_results['zero']['mean']]}")
            logging.info(f"Random initialization:")
            logging.info(f"  - Current N mean: {np.mean(errors_random_reps):.4f}")
            logging.info(f"  - All Ns means: {[f'{m:.4f}' for m in current_results['random']['mean']]}")
            logging.info(f"Results saved to {RESULTS_DIR}/Fig2_{mode}_intermediate.json")
        except Exception as e:
            logging.error(f"Failed to save intermediate results: {str(e)}")
            logging.error(f"Error details: {str(e.__class__.__name__)}")
            import traceback
            logging.error(traceback.format_exc())
    
    return (errors_uniform_all, errors_e_optimal_all, errors_zero_all, errors_random_all)

def save_results(all_errors: tuple, Ns: list, mode: str, params: dict, logger: logging.Logger) -> None:
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
    errors_uniform_all, errors_e_optimal_all, errors_zero_all, errors_random_all = all_errors

    # Compute means and bootstrapped CIs for each algorithm
    logger.info("Computing bootstrapped confidence intervals...")
    results = {
        'uniform': {
            'mean': np.mean(errors_uniform_all, axis=1).tolist(),
            'ci': [studentized_double_bootstrap(errors_uniform_all[i]) for i in range(len(errors_uniform_all))],
            'raw': {str(N): errors_uniform_all[i] for i, N in enumerate(Ns[:len(errors_uniform_all)])}
        },
        'e_optimal': {
            'mean': np.mean(errors_e_optimal_all, axis=1).tolist(),
            'ci': [studentized_double_bootstrap(errors_e_optimal_all[i]) for i in range(len(errors_e_optimal_all))],
            'raw': {str(N): errors_e_optimal_all[i] for i, N in enumerate(Ns[:len(errors_e_optimal_all)])}
        },
        'zero': {
            'mean': np.mean(errors_zero_all, axis=1).tolist(),
            'ci': [studentized_double_bootstrap(errors_zero_all[i]) for i in range(len(errors_zero_all))],
            'raw': {str(N): errors_zero_all[i] for i, N in enumerate(Ns[:len(errors_zero_all)])}
        },
        'random': {
            'mean': np.mean(errors_random_all, axis=1).tolist(),
            'ci': [studentized_double_bootstrap(errors_random_all[i]) for i in range(len(errors_random_all))],
            'raw': {str(N): errors_random_all[i] for i, N in enumerate(Ns[:len(errors_random_all)])}
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
        with open(f'{RESULTS_DIR}/Fig2_{mode}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {RESULTS_DIR}/Fig2_{mode}.json")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")

def main() -> None:
    """Main entry point for the experiment."""
    parser = argparse.ArgumentParser(description='Run initialization comparison experiments')
    parser.add_argument('--mode', type=str, choices=['completion', 'recovery'], required=True,
                      help='Mode of operation: completion or recovery')
    parser.add_argument('--d1', type=int, default=DEFAULT_PARAMS['d1'], help='First dimension')
    parser.add_argument('--d2', type=int, default=DEFAULT_PARAMS['d2'], help='Second dimension')
    parser.add_argument('--r', type=int, default=DEFAULT_PARAMS['r'], help='Rank')
    parser.add_argument('--num_repeats', type=int, default=DEFAULT_PARAMS['num_repeats'],
                      help='Number of experiment repeats')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.mode, 'fig2')
    logger.info(f"Starting experiment with parameters: {args}")
    
    # Use default parameters from config
    params = DEFAULT_PARAMS.copy()
    params.update({
        'd1': args.d1,
        'd2': args.d2,
        'r': args.r,
        'num_repeats': args.num_repeats
    })
    
    try:
        # Run experiment
        all_errors = run_experiment(
            args.mode, args.d1, args.d2, args.r, params['K'], params['num_repeats'],
            params['delta'], params['Ns'], params['c_lambda'], params['c_nu'], params['Rmax']
        )
        
        # Save results
        save_results(all_errors, params['Ns'], args.mode, params, logger)
        logger.info("Experiment completed successfully")
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        raise

if __name__ == '__main__':
    main() 
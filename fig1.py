# Standard library imports
import argparse
import json
import os
from datetime import datetime
from math import floor

# Third-party imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Local imports
from burer_monteiro import *
from config import DEFAULT_PARAMS, PROBLEM_INSTANCES_DIR, RESULTS_DIR, setup_logging
from gl_lowpopart import *
from problems import *

def save_problem_instance(arm_set, Theta_star, mode, N, run_idx, save_dir='problem_instances'):
    """Save problem instance (arm set and Theta_star) to HDF5 file"""
    os.makedirs(save_dir, exist_ok=True)
    filename = f'{save_dir}/{mode}_N{N}_run{run_idx}_instance.h5'
    with h5py.File(filename, 'w') as f:
        # Save arm set
        arm_set_array = np.array([arm.flatten('F') for arm in arm_set])
        f.create_dataset('arm_set', data=arm_set_array)
        # Save Theta_star
        f.create_dataset('Theta_star', data=Theta_star)
        # Save metadata
        f.attrs['mode'] = mode
        f.attrs['N'] = N
        f.attrs['run_idx'] = run_idx
        f.attrs['timestamp'] = datetime.now().isoformat()

def load_problem_instance(mode, N, run_idx, save_dir='problem_instances'):
    """Load problem instance from HDF5 file"""
    filename = f'{save_dir}/{mode}_N{N}_run{run_idx}_instance.h5'
    with h5py.File(filename, 'r') as f:
        arm_set_array = f['arm_set'][:]
        Theta_star = f['Theta_star'][:]
        # Reshape arm set back to original format
        arm_set = [arm.reshape(Theta_star.shape, order='F') for arm in arm_set_array]
    return arm_set, Theta_star

def run_bmf(env, d1, r, X1, y1):
    """Run BMF evaluation"""
    X_bmf, y_bmf = X1, y1
    Theta_BMF = Burer_Monteiro(d1, r, X_bmf, y_bmf)
    return np.linalg.norm(Theta_BMF - env.Theta_star, 'nuc')

def run_stage1(env, N, d1, d2, nuc_coef, e_optimal):
    """Run Stage I (Nuclear norm MLE) evaluation"""
    Theta0, X1, y1 = nuc_norm_MLE(env, N, d1, d2, nuc_coef, E_optimal=e_optimal)
    return np.linalg.norm(Theta0 - env.Theta_star, 'nuc'), X1, y1

def run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, e_optimal, gl_optimal):
    """Run Stage I + II evaluation"""
    # Stage I
    Theta0, _, _ = nuc_norm_MLE(env, N1, d1, d2, nuc_coef, E_optimal=e_optimal)
    # Stage II
    Theta = GL_LowPopArt(env, N2, d1, d2, delta, Theta0, c_nu, GL_optimal=gl_optimal)
    return np.linalg.norm(Theta - env.Theta_star, 'nuc')

def run_experiment(mode, d1, d2, r, K, num_repeats, delta, Ns, c_lambda, c_nu, Rmax, logger):
    """
    Run the main experiment comparing all algorithms
    """
    # Initialize error arrays for all algorithms
    errors_bmf_all = []
    errors_stage1_no_e_all = []
    errors_stage1_with_e_all = []
    errors_stage12_no_e_no_gl_all = []
    errors_stage12_no_e_with_gl_all = []
    errors_stage12_with_e_no_gl_all = []
    errors_stage12_with_e_with_gl_all = []

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
            with open(results_file, 'r') as f:
                current_results = json.load(f)
                if len(current_results['BMF']['mean']) > Ns.index(N):
                    logger.info(f"Loading existing results for N={N}")
                    errors_bmf_reps = current_results['BMF']['raw'][str(N)]
                    errors_stage1_no_e_reps = current_results['Stage I (no E-optimal)']['raw'][str(N)]
                    errors_stage1_with_e_reps = current_results['Stage I (with E-optimal)']['raw'][str(N)]
                    errors_stage12_no_e_no_gl_reps = current_results['Stage I+II (no E, no GL)']['raw'][str(N)]
                    errors_stage12_no_e_with_gl_reps = current_results['Stage I+II (no E, with GL)']['raw'][str(N)]
                    errors_stage12_with_e_no_gl_reps = current_results['Stage I+II (with E, no GL)']['raw'][str(N)]
                    errors_stage12_with_e_with_gl_reps = current_results['Stage I+II (with E, with GL)']['raw'][str(N)]
                    
                    # Store results for this N
                    errors_bmf_all.append(errors_bmf_reps)
                    errors_stage1_no_e_all.append(errors_stage1_no_e_reps)
                    errors_stage1_with_e_all.append(errors_stage1_with_e_reps)
                    errors_stage12_no_e_no_gl_all.append(errors_stage12_no_e_no_gl_reps)
                    errors_stage12_no_e_with_gl_all.append(errors_stage12_no_e_with_gl_reps)
                    errors_stage12_with_e_no_gl_all.append(errors_stage12_with_e_no_gl_reps)
                    errors_stage12_with_e_with_gl_all.append(errors_stage12_with_e_with_gl_reps)
                    continue

        # Run repeats
        for run_idx in tqdm(range(num_repeats), desc=f"Repeats for N={N}", leave=False):
            # Check if problem instance exists
            instance_file = f'{PROBLEM_INSTANCES_DIR}/{mode}_N{N}_run{run_idx}_instance.h5'
            if os.path.exists(instance_file):
                logger.info(f"Loading existing problem instance for N={N}, run={run_idx}")
                arm_set, Theta_star = load_problem_instance(mode, N, run_idx, PROBLEM_INSTANCES_DIR)
            else:
                # Generate a new problem instance
                Theta_star = generate_Theta_star(d1, d2, r)
                arm_set = generate_arm_set(d1, d2, K, mode=mode)
                # Save the problem instance
                save_problem_instance(arm_set, Theta_star, mode, N, run_idx, PROBLEM_INSTANCES_DIR)
            
            env = OneBitCompletion(arm_set, Theta_star)
            env.Theta_star = Theta_star

            # Run Stage I
            nuc_coef = c_lambda * np.sqrt(8 * Rmax * np.log((d1 + d2) / delta) / N)
            error_stage1_no_e, X1, y1 = run_stage1(env, N, d1, d2, nuc_coef, False)
            error_stage1_with_e = run_stage1(env, N, d1, d2, nuc_coef, True)[0]
            errors_stage1_no_e_reps.append(error_stage1_no_e)
            errors_stage1_with_e_reps.append(error_stage1_with_e)

            # Run BMF
            error_bmf = run_bmf(env, d1, r, X1, y1)
            errors_bmf_reps.append(error_bmf)

            # Run Stage I+II combinations
            N1 = 2 * floor(np.sqrt(N))
            N2 = N - N1
            nuc_coef1 = c_lambda * np.sqrt(8 * Rmax * np.log((d1 + d2) / delta) / N1)
            error_stage12_no_e_no_gl = run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, False, False)
            error_stage12_no_e_with_gl = run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, False, True)
            error_stage12_with_e_no_gl = run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, True, False)
            error_stage12_with_e_with_gl = run_stage1_2(env, N1, N2, d1, d2, nuc_coef, c_nu, delta, True, True)
            
            errors_stage12_no_e_no_gl_reps.append(error_stage12_no_e_no_gl)
            errors_stage12_no_e_with_gl_reps.append(error_stage12_no_e_with_gl)
            errors_stage12_with_e_no_gl_reps.append(error_stage12_with_e_no_gl)
            errors_stage12_with_e_with_gl_reps.append(error_stage12_with_e_with_gl)

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
        logger.info(f"BMF mean error: {np.mean(errors_bmf_reps):.4f} ± {np.std(errors_bmf_reps):.4f}")
        logger.info(f"Stage I (no E) mean error: {np.mean(errors_stage1_no_e_reps):.4f} ± {np.std(errors_stage1_no_e_reps):.4f}")
        logger.info(f"Stage I (with E) mean error: {np.mean(errors_stage1_with_e_reps):.4f} ± {np.std(errors_stage1_with_e_reps):.4f}")
        logger.info(f"Stage I+II (no E, no GL) mean error: {np.mean(errors_stage12_no_e_no_gl_reps):.4f} ± {np.std(errors_stage12_no_e_no_gl_reps):.4f}")
        logger.info(f"Stage I+II (no E, with GL) mean error: {np.mean(errors_stage12_no_e_with_gl_reps):.4f} ± {np.std(errors_stage12_no_e_with_gl_reps):.4f}")
        logger.info(f"Stage I+II (with E, no GL) mean error: {np.mean(errors_stage12_with_e_no_gl_reps):.4f} ± {np.std(errors_stage12_with_e_no_gl_reps):.4f}")
        logger.info(f"Stage I+II (with E, with GL) mean error: {np.mean(errors_stage12_with_e_with_gl_reps):.4f} ± {np.std(errors_stage12_with_e_with_gl_reps):.4f}")

        # Save intermediate results
        current_results = {
            'BMF': {
                'mean': np.mean(errors_bmf_all, axis=1).tolist(),
                'std': np.std(errors_bmf_all, axis=1).tolist(),
                'raw': {str(N): errors_bmf_reps for N in Ns[:len(errors_bmf_all)]}
            },
            'Stage I (no E-optimal)': {
                'mean': np.mean(errors_stage1_no_e_all, axis=1).tolist(),
                'std': np.std(errors_stage1_no_e_all, axis=1).tolist(),
                'raw': {str(N): errors_stage1_no_e_reps for N in Ns[:len(errors_stage1_no_e_all)]}
            },
            'Stage I (with E-optimal)': {
                'mean': np.mean(errors_stage1_with_e_all, axis=1).tolist(),
                'std': np.std(errors_stage1_with_e_all, axis=1).tolist(),
                'raw': {str(N): errors_stage1_with_e_reps for N in Ns[:len(errors_stage1_with_e_all)]}
            },
            'Stage I+II (no E, no GL)': {
                'mean': np.mean(errors_stage12_no_e_no_gl_all, axis=1).tolist(),
                'std': np.std(errors_stage12_no_e_no_gl_all, axis=1).tolist(),
                'raw': {str(N): errors_stage12_no_e_no_gl_reps for N in Ns[:len(errors_stage12_no_e_no_gl_all)]}
            },
            'Stage I+II (no E, with GL)': {
                'mean': np.mean(errors_stage12_no_e_with_gl_all, axis=1).tolist(),
                'std': np.std(errors_stage12_no_e_with_gl_all, axis=1).tolist(),
                'raw': {str(N): errors_stage12_no_e_with_gl_reps for N in Ns[:len(errors_stage12_no_e_with_gl_all)]}
            },
            'Stage I+II (with E, no GL)': {
                'mean': np.mean(errors_stage12_with_e_no_gl_all, axis=1).tolist(),
                'std': np.std(errors_stage12_with_e_no_gl_all, axis=1).tolist(),
                'raw': {str(N): errors_stage12_with_e_no_gl_reps for N in Ns[:len(errors_stage12_with_e_no_gl_all)]}
            },
            'Stage I+II (with E, with GL)': {
                'mean': np.mean(errors_stage12_with_e_with_gl_all, axis=1).tolist(),
                'std': np.std(errors_stage12_with_e_with_gl_all, axis=1).tolist(),
                'raw': {str(N): errors_stage12_with_e_with_gl_reps for N in Ns[:len(errors_stage12_with_e_with_gl_all)]}
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

    return (errors_bmf_all, errors_stage1_no_e_all, errors_stage1_with_e_all,
            errors_stage12_no_e_no_gl_all, errors_stage12_no_e_with_gl_all,
            errors_stage12_with_e_no_gl_all, errors_stage12_with_e_with_gl_all)

def save_results(all_errors, Ns, mode, params, logger):
    """Save results and generate plot"""
    # Unpack errors
    (errors_bmf_all, errors_stage1_no_e_all, errors_stage1_with_e_all,
     errors_stage12_no_e_no_gl_all, errors_stage12_no_e_with_gl_all,
     errors_stage12_with_e_no_gl_all, errors_stage12_with_e_with_gl_all) = all_errors

    # Compute mean and standard deviation for each algorithm
    results = {
        'BMF': {
            'mean': np.mean(errors_bmf_all, axis=1).tolist(),
            'std': np.std(errors_bmf_all, axis=1).tolist()
        },
        'Stage I (no E-optimal)': {
            'mean': np.mean(errors_stage1_no_e_all, axis=1).tolist(),
            'std': np.std(errors_stage1_no_e_all, axis=1).tolist()
        },
        'Stage I (with E-optimal)': {
            'mean': np.mean(errors_stage1_with_e_all, axis=1).tolist(),
            'std': np.std(errors_stage1_with_e_all, axis=1).tolist()
        },
        'Stage I+II (no E, no GL)': {
            'mean': np.mean(errors_stage12_no_e_no_gl_all, axis=1).tolist(),
            'std': np.std(errors_stage12_no_e_no_gl_all, axis=1).tolist()
        },
        'Stage I+II (no E, with GL)': {
            'mean': np.mean(errors_stage12_no_e_with_gl_all, axis=1).tolist(),
            'std': np.std(errors_stage12_no_e_with_gl_all, axis=1).tolist()
        },
        'Stage I+II (with E, no GL)': {
            'mean': np.mean(errors_stage12_with_e_no_gl_all, axis=1).tolist(),
            'std': np.std(errors_stage12_with_e_no_gl_all, axis=1).tolist()
        },
        'Stage I+II (with E, with GL)': {
            'mean': np.mean(errors_stage12_with_e_with_gl_all, axis=1).tolist(),
            'std': np.std(errors_stage12_with_e_with_gl_all, axis=1).tolist()
        }
    }

    # Add metadata
    results['metadata'] = {
        'mode': mode,
        'Ns': Ns,
        'params': params,
        'timestamp': datetime.now().isoformat()
    }

    # Save results to JSON
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f'{RESULTS_DIR}/Fig1_{mode}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {RESULTS_DIR}/Fig1_{mode}.json")

    # Plot with error bars
    plt.figure(1)
    # plt.errorbar(Ns, results['BMF']['mean'], yerr=results['BMF']['std'], 
    #             fmt='o-', label='BMF', color='black', capsize=5)
    plt.errorbar(Ns, results['Stage I (no E-optimal)']['mean'], 
                yerr=results['Stage I (no E-optimal)']['std'], 
                fmt='o-', label='Stage I (no E-optimal)', color='blue', capsize=5)
    plt.errorbar(Ns, results['Stage I (with E-optimal)']['mean'], 
                yerr=results['Stage I (with E-optimal)']['std'], 
                fmt='o-', label='Stage I (with E-optimal)', color='red', capsize=5)
    plt.errorbar(Ns, results['Stage I+II (no E, no GL)']['mean'], 
                yerr=results['Stage I+II (no E, no GL)']['std'], 
                fmt='o-', label='Stage I+II (no E, no GL)', color='green', capsize=5)
    plt.errorbar(Ns, results['Stage I+II (no E, with GL)']['mean'], 
                yerr=results['Stage I+II (no E, with GL)']['std'], 
                fmt='o-', label='Stage I+II (no E, with GL)', color='purple', capsize=5)
    plt.errorbar(Ns, results['Stage I+II (with E, no GL)']['mean'], 
                yerr=results['Stage I+II (with E, no GL)']['std'], 
                fmt='o-', label='Stage I+II (with E, no GL)', color='orange', capsize=5)
    plt.errorbar(Ns, results['Stage I+II (with E, with GL)']['mean'], 
                yerr=results['Stage I+II (with E, with GL)']['std'], 
                fmt='o-', label='Stage I+II (with E, with GL)', color='brown', capsize=5)
    
    plt.xlabel('N')
    plt.ylabel('Nuclear norm error')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(f'1-Bit {mode.capitalize()} of {params["d1"]}x{params["d2"]} Rank-{params["r"]} Matrix')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/Fig1_{mode}.png', dpi=300, bbox_inches='tight')
    logger.info(f"Plot saved to {RESULTS_DIR}/Fig1_{mode}.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Run 1-bit matrix experiments')
    parser.add_argument('--mode', type=str, choices=['completion', 'recovery'], required=True,
                      help='Mode of operation: completion or recovery')
    parser.add_argument('--d1', type=int, default=DEFAULT_PARAMS['d1'], help='First dimension')
    parser.add_argument('--d2', type=int, default=DEFAULT_PARAMS['d2'], help='Second dimension')
    parser.add_argument('--r', type=int, default=DEFAULT_PARAMS['r'], help='Rank')
    parser.add_argument('--num_repeats', type=int, default=DEFAULT_PARAMS['num_repeats'],
                      help='Number of experiment repeats')
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.mode)
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

    # Run experiment with all algorithms
    all_errors = run_experiment(
        args.mode, args.d1, args.d2, args.r, params['K'], params['num_repeats'],
        params['delta'], params['Ns'], params['c_lambda'], params['c_nu'], params['Rmax'],
        logger
    )
    
    # Save results and generate plot
    save_results(all_errors, params['Ns'], args.mode, params, logger)
    logger.info("Experiment completed successfully")

if __name__ == '__main__':
    main() 
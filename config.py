"""
Configuration file for GL-LowPopArt experiments
"""
import os
import logging
from datetime import datetime

# Default parameters for experiments
DEFAULT_PARAMS = {
    'K': 150,                    # Number of arms
    'delta': 0.001,              # Confidence parameter
    # 'Ns': [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000],  # Sample sizes
    # 'Ns': [10000, 20000, 30000, 40000, 50000] + [11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000],  # Sample sizes
    'Ns': [10000, 20000, 30000, 40000, 50000],  # Sample sizes
    # 'Ns': [60000, 70000, 80000, 90000, 100000],  # Sample sizes
    'c_lambda': 1,               # Scaling for lambda in Stage I nuc-regularized MLE
    'c_nu': 1,                   # Scaling for nu in Stage II Catoni
    'Rmax': 1/4,                 # Maximum reward
    'd1': 3,                     # First dimension
    'd2': 3,                     # Second dimension
    'r': 1,                      # Rank
    'num_repeats': 60             # Number of experiment repeats
}

# File paths
PROBLEM_INSTANCES_DIR = 'problem_instances'
RESULTS_DIR = 'results'
LOGS_DIR = 'logs'

# Create directories if they don't exist
for directory in [PROBLEM_INSTANCES_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

def setup_logging(mode: str, fig: str = 'fig1') -> logging.Logger:
    """Set up logging configuration for the experiment
    
    Args:
        mode: The mode of operation ('completion' or 'recovery')
        fig: The figure number ('fig1' or 'fig2')
    """
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(LOGS_DIR, f'{fig}_{mode}_{timestamp}.log')
    
    # Remove any existing handlers from the root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configure root logger
    root_logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger.addHandler(file_handler)
    
    # Log experiment start
    logging.info(f"Starting new experiment: fig={fig}, mode={mode}")
    
    return root_logger 
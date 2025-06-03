import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Read both JSON files
with open('results/Fig2_completion.json', 'r') as f:
    completion_data = json.load(f)
with open('results/Fig2_recovery.json', 'r') as f:
    recovery_data = json.load(f)

# Extract metadata
completion_Ns = completion_data['metadata']['Ns']
recovery_Ns = recovery_data['metadata']['Ns']

# Define methods and their labels
methods = {
    'uniform': 'U+GL',
    'e_optimal': 'E+GL',
    'zero': '0+GL',
    'random': 'Rand+GL'
}

# Set font sizes
plt.rcParams.update({
    'font.size': 40,
    'axes.labelsize': 44,
    'axes.titlesize': 46,
    'xtick.labelsize': 32,
    'ytick.labelsize': 32,
    'legend.fontsize': 42
})

# Create figure with four subplots
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(56, 32))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(56, 16))

# Define colors and line styles for methods (colorblind-friendly palette)
styles = {
    'e_optimal': {'color': '#0072B2', 'linestyle': '-', 'marker': '^'},  # Blue
    'uniform': {'color': '#0072B2', 'linestyle': '--', 'marker': 's'},   # Blue dashed
    'zero': {'color': '#E69F00', 'linestyle': '-', 'marker': 'o'},       # Orange
    'random': {'color': '#009E73', 'linestyle': '-', 'marker': 'D'}      # Green
}

def plot_data(ax, data, Ns, title, N_range=None, specific_Ns=None):
    results = {}
    
    for method, label in tqdm(methods.items(), desc="Methods"):
        # Get pre-computed means and CIs from the JSON file
        means = data[method]['mean']
        cis = data[method]['ci']
        
        # Sort data points by N
        sorted_indices = np.argsort(Ns)
        sorted_Ns = np.array(Ns)[sorted_indices]
        sorted_means = np.array(means)[sorted_indices]
        sorted_cis = np.array(cis)[sorted_indices]
        
        # Filter by N range if specified
        if N_range is not None:
            mask = (sorted_Ns >= N_range[0]) & (sorted_Ns <= N_range[1])
            sorted_Ns = sorted_Ns[mask]
            sorted_means = sorted_means[mask]
            sorted_cis = sorted_cis[mask]
            
        # Filter by specific Ns if specified
        if specific_Ns is not None:
            mask = np.isin(sorted_Ns, specific_Ns)
            sorted_Ns = sorted_Ns[mask]
            sorted_means = sorted_means[mask]
            sorted_cis = sorted_cis[mask]
        
        # Extract lower and upper CIs
        lower_cis = [ci[0] for ci in sorted_cis]
        upper_cis = [ci[1] for ci in sorted_cis]
        
        # Plot with confidence intervals
        ax.errorbar(sorted_Ns, sorted_means, 
                   yerr=[sorted_means - np.array(lower_cis), 
                        np.array(upper_cis) - sorted_means],
                   label=label, 
                   color=styles[method]['color'],
                   linestyle=styles[method]['linestyle'],
                   marker=styles[method]['marker'],
                   capsize=5, 
                   linewidth=2, 
                   markersize=8)
        
        # Store results
        results[method] = {
            'mean': sorted_means.tolist(),
            'lower_ci': lower_cis,
            'upper_ci': upper_cis
        }
    
    ax.set_xlabel('Sample Size (N)', fontsize=44)
    ax.set_ylabel('Nuclear Norm Error', fontsize=44)
    ax.set_title(title, fontsize=46)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=32)
    
    return results

# Plot completion data (top row)
completion_results1 = plot_data(ax1, completion_data, completion_Ns, 'Matrix Completion', 
                              specific_Ns=[10000, 20000, 30000, 40000, 50000])
# completion_results2 = plot_data(ax3, completion_data, completion_Ns, 'Matrix Completion (N=10000-30000)', 
#                               N_range=(10000, 30000))

# Plot recovery data (bottom row)
recovery_results1 = plot_data(ax2, recovery_data, recovery_Ns, 'Matrix Recovery', 
                            specific_Ns=[10000, 20000, 30000, 40000, 50000])
# recovery_results2 = plot_data(ax4, recovery_data, recovery_Ns, 'Matrix Recovery (N=10000-30000)', 
#                             N_range=(10000, 30000))

# Create a single legend for all subplots
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), 
          ncol=4, frameon=True, fontsize=42)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('results/fig2.png', dpi=300, bbox_inches='tight')
plt.savefig('results/fig2.pdf', dpi=600, bbox_inches='tight')
plt.close()

# Save final statistics
final_results = {
    'completion': {
        'Ns': completion_Ns,
        'methods': completion_results1
    },
    'recovery': {
        'Ns': recovery_Ns,
        'methods': recovery_results1
    }
}

with open('results/fig2_results.json', 'w') as f:
    json.dump(final_results, f, indent=2) 
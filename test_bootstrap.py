import numpy as np
from utils import studentized_double_bootstrap

def test_bootstrap():
    # Create a simple dataset with known mean and variance
    np.random.seed(42)
    data = np.random.normal(loc=1.0, scale=0.5, size=1000)
    true_mean = 1.0
    
    # Compute confidence intervals
    ci_lower, ci_upper = studentized_double_bootstrap(data, n_boot=1000, n_boot2=500, alpha=0.05)
    
    # Print results
    print(f"True mean: {true_mean:.4f}")
    print(f"Sample mean: {np.mean(data):.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"CI width: {ci_upper - ci_lower:.4f}")
    
    # Check if true mean is within CI
    assert ci_lower <= true_mean <= ci_upper, "True mean should be within confidence interval"
    print("\nTest passed: True mean is within confidence interval")

if __name__ == "__main__":
    test_bootstrap() 
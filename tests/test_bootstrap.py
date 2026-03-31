import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gl_lowpopart.utils import studentized_double_bootstrap


def test_bootstrap():
    np.random.seed(42)
    data = np.random.normal(loc=1.0, scale=0.5, size=1000)
    true_mean = 1.0
    ci_lower, ci_upper = studentized_double_bootstrap(data, n_boot=1000, n_boot2=500, alpha=0.05)
    assert ci_lower <= true_mean <= ci_upper


if __name__ == "__main__":
    test_bootstrap()


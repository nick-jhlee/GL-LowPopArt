"""Thin wrapper for package Figure 1 runner."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from gl_lowpopart.experiments.fig1 import main


if __name__ == "__main__":
    main()
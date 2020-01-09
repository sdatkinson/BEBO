# File: synthetic.py
# File Created: Sunday, 3rd November 2019 11:38:53 am
# Author: Steven Atkinson (212726320@ge.com)

"""
Another synthetic function
"""

import numpy as np

from .base import WithFunction


class Synthetic(WithFunction):
    """
    Domain: [0, 1]
    """
    def __init__(self):
        super().__init__()

        self.real_dimensions = 1
        self.general_dimensions = 1
        self.num_types = [1000]
        self._fidelity_matrix = None

        self._cache_fidelity_params()

    def _cache_fidelity_params(self):
        # Initialize and cache:
        rng_state = np.random.get_state()
        np.random.seed(42)
        self._fidelity_matrix = np.random.rand(1000, 2)
        # ...And resume previous state
        np.random.set_state(rng_state)

    def _call(self, x, i):
        a, b = self._fidelity_matrix[i[0]]

        # Original in terms of x_scaled...
        y = a + 4.0 * x - 4.0
        return 0.1 * y ** 4 - y ** 2 + (2.0 + b) * np.sin(2.0 * y)

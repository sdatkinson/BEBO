# File: forrester.py
# File Created: Sunday, 3rd November 2019 10:21:48 am
# Author: Steven Atkinson (212726320@ge.com)

import numpy as np

from .base import WithFunction


class Forrester(WithFunction):
    def __init__(self):
        super().__init__()

        self.real_dimensions = 1
        self.general_dimensions = 1
        self.num_types = [1000]
        self._fidelity_matrix = None

        self._cache_fidelity_params()

    def _call(self, x, i):
        a, b, c = self._fidelity_matrix[i[0]]
        f_hi = (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0)
        return a * f_hi + b * (x - 0.5) + c

    def _cache_fidelity_params(self):
        # Initialize and cache:
        rng_state = np.random.get_state()
        np.random.seed(42)
        self._fidelity_matrix = np.concatenate(
            (
                np.array([[1.0, 0.0, 0.0], [0.5, 10.0, -5.0]]),
                np.array([[0.0, 0.0, -5.0]]) + np.array([[1.0, 10.0, 5.0]]) * \
                    np.random.rand(998, 3)
            )
        )
        # ...And resume previous state
        np.random.set_state(rng_state)    
    
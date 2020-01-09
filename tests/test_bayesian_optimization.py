# File: test_bayesian_optimization.py
# File Created: Tuesday, 5th November 2019 10:00:04 am
# Author: Steven Atkinson (212726320@ge.com)

import os
import sys

import numpy as np
import torch
import pytest

base_path = os.path.join(os.path.dirname(__file__), "..")

if not base_path in sys.path:
    sys.path.append(base_path)

from src import bayesian_optimization

torch_dtype = torch.double


class MockBase(bayesian_optimization.Base):
    pass


class TestBase(object):
    pass


class _TBase(object):
    """
    Set up reusable parts of the testing
    """
    @staticmethod
    def _mock_train_function():
        pass

    @staticmethod
    def _mock_append_function():
        pass

    @staticmethod
    def _mock_predict_function(x, diag=True):
        n = x.shape[0]
        mean = x @ torch.ones(2, 1, dtype=torch_dtype)
        if diag:
            var = torch.ones(n, 1, dtype=torch_dtype)
            return mean, var
        else:
            cov = 0.99 * torch.ones(n, n) + 0.01 * torch.eye(n, dtype=torch_dtype)
            return mean, cov


class TestStaticDataset(_TBase):
    @classmethod
    def setup_class(cls):
        cls.x_all = np.array([[1.0, 2.0], [20.0, 30.0]])
        cls.y_all = np.array([[3.0], [4.0]])

    def test_init(self):
        bo = bayesian_optimization.StaticDataset(
            self.x_all, 
            self.y_all, 
            self._mock_train_function, 
            self._mock_predict_function,
            self._mock_append_function
        )
    
    def test_get_p_best(self):
        """
        Note: because the prediction function is linear with a strong slope in 
        the (+, +) direction and the covariance is comparatively small, we 
        expect that the first input in x_all will be predicted as lower 
        basically all the time.

        So, p_best should probably be exactly [1, 0].
        A snapshot test (2019-11-05) shows this to be the case.
        """
        bo = bayesian_optimization.StaticDataset(
            self.x_all, 
            self.y_all, 
            self._mock_train_function, 
            self._mock_predict_function,
            self._mock_append_function
        )

        # RNG seeds to ensure that this test replicates
        np.random.seed(0)  
        torch.manual_seed(0)
        x = bo.x_all
        p_best = bo._get_p_best(x)
        assert isinstance(p_best, np.ndarray)
        assert p_best.ndim == 1
        assert p_best.size == x.shape[0]

        # Super rare that this wouldn't be the case (and shouldn't happen at all
        # since I seeded the RNG and checked).
        assert p_best[0] == 1.0, "p(best, 0) = %f? (should be 1.0)" % p_best[0]
        assert p_best[1] == 0.0, "p(best, 1) = %f? (should be 0.0)" % p_best[1]

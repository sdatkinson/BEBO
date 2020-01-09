# File: bayesian_optimization.py
# File Created: Friday, 1st November 2019 3:13:22 pm
# Author: Steven Atkinson (212726320@ge.com)

"""
Bayesian optimization with latent variable Gaussian processes
"""

import abc
from typing import Callable, Tuple

import numpy as np
from scipy.optimize import minimize
import torch.distributions
import matplotlib.pyplot as plt

from gptorch.util import TensorType, torch_dtype, as_tensor
from gptorch.functions import cholesky


def _expected_improvement(mean, std, best, xi=0.0, mode="max"):
    """
    Compute the expected improvement of a Gaussian with provided mean & std 
    deviation over a current best (lower is better).

    From https://krasserm.github.io/2018/03/21/bayesian-optimization
    Accessed 2019-11-02.

    :param mean: Mean of the Gaussian (scalar)
    :type mean: torch.Tensor
    :param std: Standard deviation of the Gaussian (scalar)
    :type std: torch.Tensor
    :param best: Current best
    :type best: torch.Tensor
    :param xi: Hyperparameter that can be increased to favor exploration.  
        0=true EI.
    :type xi: torch.Tensor  
    :param mode: Whether higher ("max") or lower ("min") is better.
    :type mode: str  

    :return: (torch.Tensor) expected improvement
    """

    if mode != "max" and mode != "min":
        raise ValueError('Require mode="max" or "min"')
    if mode == "min":
        return -_expected_improvement(-mean, std, -best, xi=-xi)
    
    best = as_tensor(best)
    z = (mean - best - xi) / std
    p = torch.distributions.Normal(torch.zeros(1), torch.ones(1))
    cdf, pdf = p.cdf(z), p.log_prob(z).exp()

    return (mean - best - xi) * cdf + std * pdf


class Base(abc.ABC):
    """
    Base BO class.  From this we either do BO with a function that we can 
    evaluate or a finite set of pre-computed data that we can pick from.
    """
    def __init__(self, dim, train_function, predict_function, append_function):
        """
        :param train_function: callable like `train_function()` to call before each time
            You want to add find a new point.
        :type train_function: callable with no args and no kwargs
        :param predict_function: Usage: `predict_function(x)`, where x is a 
            torch.DoubleTensor.
            Assume that it predicts a Gaussian.
            Returns two torch.DoubleTensors.  
            First is predictive mean. 
            Second is predictive variance.
        :type predict_function: Callable[torch.DoubleTensor]
        :param append_function: Append a dtatum to the model's training data
        :type append_function: Callable
        """

        self._dim = dim
        self.train_function = train_function
        self.predict_function = predict_function
        self.append_function = append_function

        # x: design points that BO tried.
        # y: Their objective values
        self.x, self.y = [], []
        # A list of callbacks to perform after training and before selecting the
        # next point. 
        # Each entry should be a dict with "func", "args", and "kwargs" fields.
        # Examples:
        # * Looking at the metamodel's current predictive function
        # * (StaticDataset) Assessing which available point is likely to be the 
        #   best.
        self._pre_selection_callbacks = []

    @property
    def dim(self):
        return self._dim

    def register_pre_selection_callback(self, func, *args, **kwargs):
        self._pre_selection_callbacks.append(
            {"func": func, "args": args, "kwargs": kwargs}
        )
    
    def add_points(self, n, verbose=False, train_callback=None):
        """
        Use the provided metamodel to propose and evaluate new points.

        :param n: How many points to find and add
        :type n: int
        :param verbose: Whether to print out the points that are decided on
        :type verbose: bool
        :param train_callback: A function to call after training is done
        :type train_callback: callable
        """

        for i in range(n):
            self.train_function()
            for callback in self._pre_selection_callbacks:
                callback["func"](*callback["args"], **callback["kwargs"])
            x_new, y_new = self._get_next_point()
            self.append_function(x_new, y_new)
            self.x.append(x_new)
            self.y.append(y_new)
            if verbose:
                print(
                    "Point %i / %i: %s, val=%.6e" % 
                    (i + 1, n, str(self.x[-1]), self.y[-1])
                )

    @abc.abstractmethod
    def _get_next_point(self):
        """
        Decide on the next x and get its corresponding y.

        :return: (np.ndarray, float) x_new, y_new
        """
        raise NotImplementedError("Implement getting next point")


class WithFunction(Base):
    """
    BO class.  Tries to minimize some black-box function.
    """
    def __init__(
        self, 
        x_bounds: np.ndarray,
        train_function: Callable[[], None], 
        predict_function: Callable[[TensorType], Tuple[TensorType, TensorType]], 
        append_function: Callable[[np.ndarray, np.ndarray], None],
        eval_function: Callable[[np.ndarray], float]):
        """
        :param x_bounds: hybercuboidal boundaries of the design space.  
            Shape = [D x 2].
            Each row is a dimension.  
            First column is lower bound. 
            Second column is upper bound.
        :type x_bounds: np.ndarray
        :param eval_function: Function to evaluate the "expensive" function at 
            some input location.  Must return a float.
        """

        super().__init__(
            x_bounds.shape[0], train_function, predict_function, append_function
        )
        self.x_bounds = x_bounds
        self.eval_function = eval_function

    def _get_next_point(self, n_restarts=20, disp=False):
        """
        Restart several times and take the best point found

        :return: (np.ndarray) 1D new design point
        """

        x, y = [], []
        for _ in range(n_restarts):
            xi, yi = self._find_next_point_single_restart(disp=disp)
            x.append(xi)
            y.append(yi)
        i = np.argmin(y)
        
        x_new = x[i]
        y_new = self.eval_function(x_new)

        return x_new, y_new
        
    def _find_next_point_single_restart(self, disp=False):
        """
        Use the surrogate's prediction function to find out where the next 
        design point should go.

        :return: (np.ndarray, float) x, prediction
        """

        x = self._new_point().flatten()
       
        def func_and_grad(x):
            x = TensorType(np.atleast_2d(x))
            x.requires_grad_(True)
            m, v = self.predict_function(x)
            s = v.sqrt()

            if not self.y:  # No current data: use mean ("everything is an improvement")
                f = m
            else:
                f = _expected_improvement(m, s, min(self.y), mode="min")

            if f.requires_grad:
                f.backward()
                g = x.grad.detach().cpu().numpy().flatten()
            else:
                g = 0.0 * x.detach().cpu().numpy().flatten()
            f = f.detach().cpu().item()
            
            return f, g

        options = dict(disp=disp, maxiter=100)
        result = minimize(
            func_and_grad, x, method="L-BFGS-B", bounds=self.x_bounds, options=options,
            jac=True
        )
        
        return result['x'], result['fun']

    def _new_point(self):
        xb = self.x_bounds.T
        return xb[0] + (xb[1] - xb[0]) * np.random.rand(self.dim)


class StaticDataset(Base):
    """
    BO case where we don't have a simulator; we just have a finite dataset of 
    pre-computed points.
    We'll just pretend that we haven't evaluated the points until we pick them.
    
    Other differences:
    * We don't need x_bounds since our feasible set is now just a finite set of 
      points given by x_all.
    * 


    We'll still keep the x's and y's that we "evaluate", but we'll also want to 
    keep track of our metamodel's belief about which of the possibilities in the
    dataset seems to look the best.
    """

    def __init__(self, x_all, y_all, train_function, predict_function, append_function):
        super().__init__(
            x_all.shape[1], train_function, predict_function, append_function
        )

        self.x_all, self.y_all = x_all, y_all
        self._added = np.array([False] * x_all.shape[0])
        self.register_pre_selection_callback(self._evaluate_dataset)
        # List of arrays, one entry per BO iteration.
        # The array is 1D, length=number of data in x_all/y_all.
        # Each entry is the probability that the coresponding datum is the best 
        # one in the set.
        self.p_best = []

    def _get_next_point(self, verbose=False):
        x_test = np.array(
            [xi for i, xi in enumerate(self.x_all) if not self._added[i]]
        )
        y_test = np.array(
            [yi for i, yi in enumerate(self.y_all) if not self._added[i]]
        )
        indices = np.array(
            [i for i in range(self.x_all.shape[0]) if not self._added[i]]
        )
        p_best = self._get_p_best(x_test)
        i = indices[np.argmax(p_best)]
        if verbose:
            print("Select point %i" % i)
        x_new, y_new = self.x_all[i], self.y_all[i][0]
        self._added[i] = True

        return x_new, y_new

    def _get_p_best(self, x_test, y_test=None, n_samples=100000, show=False):
        """
        Out of the inputs in x_test, determine for each input the probability 
        that its y would be the best (lowest)
        """

        with torch.no_grad():
            # Need the FULL predictive distribution!
            m, c = self.predict_function(TensorType(x_test), diag=False)
            assert m.shape[1] == 1, 'How to quantify "best" for multi-output?'
            
            lc = cholesky(c)
            epsilon = torch.randn(n_samples, *m.shape, dtype=torch_dtype)
            samples = (m[None, :, :] + lc[None, :, :] @ epsilon).cpu().numpy()

        i_best = np.argmin(samples, axis=1)

        p_best = np.array(
            [np.sum(i_best == i) for i in range(self.x_all.shape[0])]
        ) / n_samples

        if show:
            self._show_p_best_analysis(x_test, y_test, m, c, samples, p_best)

        return p_best

    def _show_p_best_analysis(self, x_test, y_test, m, c, samples, p_best):
        """
        Helper function to visualize the p(best) calculations
        """

        m = m.numpy().flatten()
        unc = 2.0 * np.sqrt(c.diag().numpy().flatten())
        x_plot = np.arange(m.size)

        # Plot 1: dimension plot
        plt.figure()
        plt.errorbar(np.arange(m.size), m, unc, linestyle="none", marker="o")
        for i in range(m.size):
            plt.text(i, m[i] - unc[i], "%.02f" % p_best[i], ha="center", va="top")
        # Plot some samples:
        n_samples = min(samples.shape[0], 10)
        for i in range(n_samples):
            plt.plot(x_plot, samples[i].flatten(), color="C2", alpha=0.3)
        if y_test is not None:
            plt.plot(
                np.arange(m.size), 
                y_test.flatten(), 
                linestyle="none", 
                marker="o", 
                color="C1"
            )
        plt.xlabel("Datum")
        plt.ylabel("Prediction")

        # Plot 2: prediction accuracy
        if y_test is not None:
            plt.figure()
            plt.errorbar(y_test.flatten(), m, unc, marker=".", linestyle="none")
            plt.plot(plt.xlim(), plt.xlim(), linestyle="--", color="C1")
            plt.xlabel("Targets")
            plt.ylabel("Predictions")

        plt.show()

    def _evaluate_dataset(self, n_samples=10000):
        """
        Using the metamodel, assess the probability of each datum in 
        (x_all, y_all) being the best one that we've got.

        :param n_samples: How many times to sample the predictive distribution to
        """

        self.p_best.append(self._get_p_best(self.x_all))

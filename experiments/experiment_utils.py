# File: experiment_utils.py
# File Created: Wednesday, 6th November 2019 9:37:33 am
# Author: Steven Atkinson (212726320@ge.com)

"""
Utilities specifically for these experiments that aren't necessarily intrinsic 
to the methodology, e.g.

* Getting systems
* Design of experiments
* Organizing legacy data
* Initializing and pre-training metamodels (before BO / the only training for 
  regression)
"""

import os
import sys

import numpy as np
from gptorch.util import TensorType
from gptorch.kernels import Rbf
import gptorch
from pyro.distributions import Normal
import torch

base_path = os.path.join(os.path.dirname(__file__), "..")
if base_path not in sys.path:
    sys.path.append(base_path)

from src import systems, BayesianGP
from src.models import EGP, SafeGPR as GPR
from src.embedders import DeterministicEmbedder, GaussianEmbedder


def get_x_bounds():
    """
    All problems where we have a design set to explore with a forward function
    are normalized to be a unit hypercube. [0, 1]^dx
    """
    return 0.0, 1.0


def doe(n, d):
    """
    Basic random experimental design
    """
    x_min, x_max = get_x_bounds()
    return x_min + (x_max - x_min) * np.random.rand(n, d)


def get_system(system_type, task_0):
    if system_type == "forrester":
        return systems.Forrester()
    elif system_type == "synthetic":
        return systems.Synthetic()
    elif system_type == "static":
        return systems.Static(task_0=task_0)
    else:
        raise ValueError("Unrecognized system type %s" % system_type)


def get_legacy_data(system, args):
    if system.has_function:
        return _get_legacy_data_with_function(system, args)
    else:
        return _get_legacy_data_static(system, args)


def initialize_model(xr, xg, y, model_type):
    """
    Initialize the metamodel to be used for BO.
    """

    if model_type == "BEGP":
        return EGP(xr, xg, y)
    elif model_type == "EGP":
        return EGP(xr, xg, y, embedder_type=DeterministicEmbedder)
    elif model_type == "GP":
        x = TensorType(np.zeros((0, xr.shape[1])))
        y = TensorType(np.zeros((0, 1)))
        return GPR(
            x, y, Rbf(x.shape[1], ARD=True), likelihood=gptorch.likelihoods.Gaussian(variance=0.001)
        )
    elif model_type == "BGP":
        return _initialize_bayesian_gp(xr, xg, y)
    raise ValueError("Unexpected model type %s" % model_type)


# Training


def pre_train(model, model_type):
    """
    Pre-train the model to get it in a good state before starting BO.
    Also used by the embedding models (not vanilla GP) for regression.
    """

    if model_type == "BEGP" or model_type == "EGP":
        model.train()
        model.likelihood.variance.requires_grad_(False)
        model.optimize(method="L-BFGS-B", max_iter=15)
        train_function_egp(model, max_iter=200, learning_rate=0.05)
        model.likelihood.variance.requires_grad_(True)
        train_function_egp(model, max_iter=300, learning_rate=0.05)
    elif model_type == "EGP":
        model.train()
        model.optimize(method="L-BFGS-B", max_iter=500)


def train_function_gptorch(model, method="Adam", learning_rate=None, max_iter=100):
    """
    gptorch training method used by vanilla GP both for regression and as the 
    intermediate training step for all methods during BO.
    """
    model.requires_grad_(True)
    model.optimize(method=method, learning_rate=learning_rate, max_iter=max_iter)
    model.requires_grad_(False)


def train_function_egp(model, max_iter=100, learning_rate=None):
    model.train()
    train_function_gptorch(
        model, method="Adam", learning_rate=learning_rate, max_iter=max_iter
    )


# Prediction


def predict_function_begp(model, system, xr, diag=True, n_samples=32):
    """
    Bayesian embedded Gaussian process model predictions.

    Take samples of the predictive density; each sample uses different samples 
    of the embedded latents.

    :param xr: Real-valued inputs
    :type xr: torch.Tensor
    :param n_samples: How many times to repeat each prediction
    :type n_samples: int

    :return: (torch.Tensor, torch.Tensor) mean & variance of the mixture of Gaussians
    """

    model.eval()
    n_test = xr.shape[0]
    xg = np.array([["0"] * system.general_dimensions] * n_test)
    mvs = [model.predict_y(xr, xg, diag=diag) for _ in range(n_samples)]

    # Outputs can be either numpy or torch, which have *slightly* different APIs.
    # Handle that here. (Need correct stack op & correct kwarg for taking means & 
    # variances over a dimension of the arrays)
    if isinstance(mvs[0][0], np.ndarray):
        stack = np.stack
        dim_kwarg = dict(axis=0)
        transpose_3d = lambda x: np.transpose(x, axes=[0, 2, 1])
    else:
        stack = torch.stack
        dim_kwarg = dict(dim=0)
        transpose_3d = lambda x: torch.transpose(x, 1, 2)

    # Cubes of predictions [S x N x DY]
    m3 = stack([mvsi[0] for mvsi in mvs])
    v3 = stack([mvsi[1] for mvsi in mvs])

    # Mixtures-of-Gaussians moments
    m = m3.mean(**dim_kwarg)
    if diag:
        v = m3.var(**dim_kwarg) + v3.mean(**dim_kwarg)
    else:
        if n_samples == 1:
            v = v3[0]
        else:
            # Unbiased cov:
            cov_mean = ((m3 - m) @ transpose_3d(m3 - m)).sum(**dim_kwarg) / \
                (m.shape[0] - 1)
            mean_cov = v3.mean(**dim_kwarg)
            v = cov_mean + mean_cov

    return m, v


def predict_function_egp(model, system, xr, diag=True):
    model.eval()
    n_test = xr.shape[0]
    xg = np.array([["0"] * system.general_dimensions] * n_test)
    return model.predict_y(xr, xg, diag=diag)


# Private functions


def _get_legacy_data_with_function(system, args):
    assert system.general_dimensions == 1, "1 general dimension for now."

    xr, xg, y = [], [], []
    # 0 is held-out system
    for i in range(1, args.num_legacy + 1):
        xr_i = doe(args.data_per_legacy, system.real_dimensions)
        xg_i = np.array([str(i)] * args.data_per_legacy)[:, np.newaxis]
        y_i = system(xr_i, [i])

        xr.append(xr_i)
        xg.append(xg_i)
        y.append(y_i)
    
    return np.concatenate(xr), np.concatenate(xg), np.concatenate(y)


def _get_legacy_data_static(system, args):
    if args.data_per_legacy != -1:
        raise ValueError(
            "Static dataset problems use all legacy data. " + 
            "Set --data-pet-legacy = -1 to show that you understand this."
        )

    if system.general_dimensions > 1:
        raise NotImplementedError(
            "More than 1 general dimension not supported yet."
        )

    num_legacy = args.num_legacy if args.num_legacy >= 0 else system.num_types[0] - 1

    if num_legacy > system.num_types[0]- 1:
        raise ValueError(
            "Asked for %i legacy instances, but only %i are available" %
            (num_legacy, system.num_types[0] - 1)
        )

    xr, xg, y = [], [], []
    # 0 is held-out system
    for i in range(1, num_legacy + 1):
        xr_i, y_i = system.get_data([i])
        xg_i = np.array([[str(i)] * system.general_dimensions] * xr_i.shape[0])

        xr.append(xr_i)
        xg.append(xg_i)
        y.append(y_i)
    
    return np.concatenate(xr), np.concatenate(xg), np.concatenate(y)


def _initialize_bayesian_gp(xr, xg, y):
    """
    Initialize a Bayesian GP where we use the elgacy data to form priors on the
    model's (constant) mean function * kernel parameters.

    Get Gaussians for priors based on empirical moments of models trained on
    legacy task.  For now, weight each legacy task equally
    """

    assert xg.shape[1] == 1  # Not sure how this might work otherwise

    # 1) Train models on legacy tasks to get data to form priors:
    legacy_parameters = {
        "scales": [],
        "variances": [],
        "jitters": [],
        "biases": []
    }
    for xgi in np.unique(xg):
        if xgi == "0": continue
        i = np.where(xg == xgi)[0]
        xi, yi = xr[i], y[i]
        kern = Rbf(xi.shape[1], ARD=True)
        model = GPR(
            xi, 
            yi, 
            Rbf(xi.shape[1], ARD=True), 
            mean_function=gptorch.mean_functions.Constant(1),
            likelihood=gptorch.likelihoods.Gaussian(variance=0.001)
        )
        # We don't need a good model--just a ballpark of reasonable 
        # hyperparameters so we can get a prior.
        model.optimize(method="Adam", max_iter=200)
        # ASSUMPTION: gptorch using ExpTransform as BayesianGP will!
        # Valid for at least gptorch version <= 0.3.
        # Get raw, torch param values
        legacy_parameters["scales"].append(model.kernel.length_scales.data)
        legacy_parameters["variances"].append(model.kernel.variance.data)
        legacy_parameters["jitters"].append(model.likelihood.variance.data)
        legacy_parameters["biases"].append(model.mean_function.val.data)

    for key in legacy_parameters.keys():
        legacy_parameters[key] = torch.stack(legacy_parameters[key])
    
    # 2) Initialize the Bayesian GP:
    i = np.where(xg == "0")[0]
    model = BayesianGP(xr[i], y[i].flatten())
    model.raw_scales_prior = Normal(
        legacy_parameters["scales"].mean(dim=0), legacy_parameters["scales"].std(dim=0)
    )
    model.raw_variance_prior = Normal(
        legacy_parameters["variances"].mean(dim=0), 
        legacy_parameters["variances"].std(dim=0)
    )
    model.raw_jitter_prior = Normal(
        legacy_parameters["jitters"].mean(dim=0), 
        legacy_parameters["jitters"].std(dim=0)
    )
    model.bias_prior = Normal(
        legacy_parameters["biases"].mean(dim=0), legacy_parameters["biases"].std(dim=0)
    )

    return model
    
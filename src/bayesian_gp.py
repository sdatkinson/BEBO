# File: bayesian_gp.py
# File Created: Thursday, 7th November 2019 9:55:27 am
# Author: Steven Atkinson (212726320@ge.com)

"""
Simple Bayesian Gaussian process

Example usage:
>>> model = BayesianGP(x, y)
>>> model.raw_scales_prior = Normal(mean_scales, std_scales)  # Optional
>>> model.fit()
>>> mf, vf = model.predict_f(x_test)
>>> my, cy = model.predict_y(x_test, diag=False)
"""

from functools import partial

import numpy as np
from pyro import sample
from pyro.distributions import Normal, MultivariateNormal, Delta
from pyro.distributions.transforms import ExpTransform
from pyro.infer import Predictive
from pyro.infer.mcmc import MCMC, NUTS
import torch

torch.set_num_threads(1)

TensorType = torch.DoubleTensor
torch_dtype = torch.double

zeros = partial(torch.zeros, dtype=torch_dtype)
ones = partial(torch.ones, dtype=torch_dtype)
eye = partial(torch.eye, dtype=torch_dtype)

_trtrs = lambda b, a: torch.triangular_solve(b, a, upper=False)[0]


def _squared_distance(x1, x2):
    """
    Compute squared distance matrix.

    :param x1: [N1 x D]
    :type x1: torch.Tensor
    :param x2: [N2 x D]
    :type x2: torch.Tensor

    :return: [N1 x N2] squared distance matrix
    """

    r2 = (
        torch.sum(x1 ** 2, dim=1, keepdim=True)
        - 2.0 * x1 @ x2.t()
        + torch.sum(x2 ** 2, dim=1, keepdim=True).t()
    )
    r2 = r2 - (torch.clamp(r2, max=0.0)).detach()

    return r2


def _rbf(x1, x2, scales, variance):
    x1, x2 = x1 / scales, x2 / scales
    r2 = _squared_distance(x1, x2)

    return variance * torch.exp(-r2)


def _rbf_diag(x1, variance):
    return variance + zeros(x1.shape[0])


def _jitchol(x):
    """
    Cholesky with jitter backup
    """
    try:
        return torch.cholesky(x)
    except RuntimeError:
        factor = x.diag().mean()
        for i in range(10):
            jitter = factor * 2.0 ** (-9 + i)
            try:
                return torch.cholesky(x + jitter * eye(x.shape[0]))
            except RuntimeError:
                pass
        else:
            raise RuntimeError("Cholesky failed")


def _input_as_tensor(func):
    def wrapped(obj, x_test, diag, with_jitter):
        from_numpy = isinstance(x_test, np.ndarray)
        if from_numpy:
            x_test = TensorType(x_test)
        mean, cov = func(obj, x_test, diag, with_jitter)
        if from_numpy:
            mean, cov = mean.detach().cpu().numpy(), cov.detach().cpu().numpy()
        return mean, cov

    return wrapped


class BayesianGP(object):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        """
        :param x: [N x D]
        :param y: [N]
        """

        x, y = TensorType(x), TensorType(y)
        assert x.ndimension() == 2
        assert y.ndimension() == 1
        assert x.shape[0] == y.numel()

        self.x = x
        self.y = y
        self.n_samples = 32
        self._xform = ExpTransform()

        # Length scales for the kernel
        self.raw_scales_prior = Normal(zeros(self.dx), ones(self.dx))
        # Kernel variance
        self.raw_variance_prior = Normal(zeros(1), ones(1))
        # Jitter, aka Gaussian likelihood's variance
        self.raw_jitter_prior = Normal(-3.0 + zeros(1), ones(1))
        # For the constant ("bias") mean function
        self.bias_prior = Normal(zeros(1), ones(1))

        self._mcmc = None

    @property
    def dx(self):
        """
        Input dimension
        """
        return self.x.shape[1]

    @property
    def n(self):
        """
        Number of data
        """
        return self.y.numel()

    def fit(self):
        mcmc_kernel = NUTS(self._prior_model)
        self._mcmc = MCMC(mcmc_kernel, num_samples=self.n_samples, warmup_steps=128)
        self._mcmc.run()

    def predict_f(self, x_test, diag=True):
        return self._predict(x_test, diag, False)

    def predict_y(self, x_test, diag=True):
        return self._predict(x_test, diag, True)

    def append_data(self, x_new, y_new):
        """
        Add new input-output pair(s) to the model

        :param x_new: inputs
        :type x_new: np.ndarray
        :param y_new: outputs
        :type y_new: np.ndarray
        """

        self.x = torch.cat((self.x, TensorType(np.atleast_2d(x_new))))
        self.y = torch.cat((self.y, TensorType(y_new.flatten())))

    def _prior_model(self):
        scales, variance, jitter, bias = self._get_samples()
        if self.n > 0:
            kyy = _rbf(self.x, self.x, scales, variance) + jitter * eye(self.n)
            try:
                ckyy = _jitchol(kyy)
                sample(
                    "output",
                    MultivariateNormal(bias + zeros(self.n), scale_tril=ckyy),
                    obs=self.y,
                )
            except RuntimeError:  # Cholesky fails?
                # "No chance"
                sample("output", Delta(zeros(1)), obs=ones(1))

    def _posterior_model(self, x_test, diag, with_jitter):
        """
        Return means & (co)variance samples.
        """

        assert self.n > 0, "Need at least one training datum for posterior"

        scales, variance, jitter, bias = self._get_samples()
        kyy = _rbf(self.x, self.x, scales, variance) + jitter * eye(self.n)
        ckyy = _jitchol(kyy)
        kys = _rbf(self.x, x_test, scales, variance)

        alpha = _trtrs(kys, ckyy)
        beta = _trtrs(self.y[:, None] - bias, ckyy)

        mean = (alpha.t() @ beta).flatten() + bias
        if diag:
            kss = _rbf_diag(x_test, variance)
            cov = kss - torch.sum(alpha ** 2, dim=0)
            if with_jitter:
                cov = cov + jitter
            # Guard against numerically-negative variances?
            cov = cov - (torch.clamp(cov, max=0.0)).detach()
        else:
            kss = _rbf(x_test, x_test, scales, variance)
            cov = kss - alpha.t() @ alpha
            if with_jitter:
                cov = cov + jitter * eye(*cov.shape)
            # Numerically-negativs variances?...

        sample("mean", Delta(mean))
        sample("cov", Delta(cov))

    def _posterior_model_no_data(self, x_test, diag, with_jitter):
        """
        When the conditioning set is empty
        """

        scales, variance, jitter, bias = self._get_samples()
        if diag:
            cov = _rbf_diag(x_test, variance)
            if with_jitter:
                cov = cov + jitter
        else:
            cov = _rbf(x_test, x_test, scales, variance)
            if with_jitter:
                cov = cov + jitter * eye(x_test.shape[0])
        mean = torch.zeros(x_test.shape[0]) + bias

        sample("mean", Delta(mean))
        sample("cov", Delta(cov))

    def _get_samples(self):
        scales = self._xform(sample("raw_scales", self.raw_scales_prior))
        variance = self._xform(sample("raw_variance", self.raw_variance_prior))
        jitter = self._xform(sample("raw_jitter", self.raw_jitter_prior))
        bias = sample("bias", self.bias_prior)

        return scales, variance, jitter, bias

    @_input_as_tensor
    def _predict(self, x_test: TensorType, diag, with_jitter):
        """
        Return predictive mean [N* x 1] and either predictive variance [N* x 1]
        or covariance [N* x N*]

        :return: (TensorType, TensorType) mean & (co)variance
        """

        model = self._posterior_model if self.n > 0 else self._posterior_model_no_data
        samples = Predictive(model, self._mcmc.get_samples()).get_samples(
            x_test, diag, with_jitter
        )

        means, covs = samples["mean"], samples["cov"]

        mean = means.mean(dim=0)
        # Law of total (co)variance:
        if diag:
            cov = means.var(dim=0) + covs.mean(dim=0)
        else:
            d_mean = (means - mean)[:, :, None]
            cov_of_means = (d_mean @ torch.transpose(d_mean, 1, 2)).sum(dim=0) / (
                means.shape[0] - 1
            )
            mean_of_covs = covs.mean(dim=0)
            cov = cov_of_means + mean_of_covs

        # Make sure the shapes are right:
        if len(mean.shape) == 1:
            mean = mean[:, None]
        if len(cov.shape) == 1:
            cov = cov[:, None]

        return mean, cov

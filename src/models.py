# File: model.py
# File Created: Friday, 23rd August 2019 8:23:41 am
# Author: Steven Atkinson (212726320@ge.com)

"""
"Embedding" models

You can provide both (traditional) real-valued inputs as well as
"""

import abc
from typing import List

import numpy as np
import torch
from gptorch.model import Param
from gptorch.models import GPR, SVGP
from gptorch import kernels, likelihoods
from gptorch.mean_functions import Constant
from gptorch.models.base import input_as_tensor
from gptorch.util import TensorType, torch_dtype

from .embedders import Embedder, DeterministicEmbedder, GaussianEmbedder


def _init_kernel_and_likelihood(x: torch.Tensor, y: torch.Tensor):
    """
    Reasonable initial values for kernels and likelihoods
    """

    kernel = kernels.Rbf(x.shape[1], ARD=True)
    x_range = x.max(dim=0)[0] - x.min(dim=0)[0]
    x_range[np.where(x_range == 0.0)[0]] = 1.0
    kernel.length_scales.data = torch.log(0.2 * x_range)
    kernel.variance.data[0] = torch.log(y.var())

    likelihood = likelihoods.Gaussian(variance=0.001 * y.var())

    return kernel, likelihood


def _embed(embedders, xg, mode=None):
    """
    Function that's external to the embedding class so that you can call it 
    before the torch.nn.Module __init__()
    """

    if mode is not None:
        for e in embedders:
            e.mode = mode

    return torch.cat([e(xg_j) for e, xg_j in zip(embedders, xg.T)], dim=1)


def _embed_before_predicting(f):
    """
    Decorator to call embedders before a predict function
    """

    @input_as_tensor  # Ensures xr is a torch Tensor
    def wrapped(obj, xr_test, xg_test, *args, **kwargs):
        
        # Single call to embedder so taht they embed the train & test general 
        # inputs coherently:
        xg_cat = np.concatenate([obj.xg, xg_test])
        xg_embed_all = obj._embed(xg_cat)
        xg_embed_train = xg_embed_all[:obj.num_data]
        xg_embed_test = xg_embed_all[obj.num_data:]

        x_all_train = torch.cat([obj.xr, xg_embed_train], dim=1)
        x_all_test = torch.cat([xr_test, xg_embed_test], dim=1)

        # Continue with the usual prediction function.
        return f(obj, x_all_test, x=x_all_train, *args, **kwargs)

    return wrapped


def _make_embedding_gp_class(cls):
    class C(cls):
        """
        Embedding GP class
        """
        def __init__(self, xr: np.ndarray, xg: np.ndarray, y: np.ndarray, d_xi=None,
                embedder_type=GaussianEmbedder, **kwargs):
            """
            :param xg: "General" training inputs--will be embedded
            :param xr: (Real-valued) training inputs--don't embed
            :param y: Training outputs
            :param d_xi: List of latent dimensions for each embedder. 
                If None, then use number of unique instances minus 1.

            """

            xr, y = torch.Tensor(xr), torch.Tensor(y)
            # initialize the embedders:
            latent_dimensions = self._get_latent_dimensions(xg, d_xi)
            embedders = [embedder_type(ld) for ld in latent_dimensions]
            # Temporary x to initialize the base kernel
            # .detach() so that it's a valid leaf as self.X
            x_temp = torch.cat([xr, _embed(embedders, xg, mode="posterior")], dim=1).\
                detach()
            kernel, likelihood = _init_kernel_and_likelihood(x_temp, y)

            super(cls, self).__init__(x_temp, y, kernel,
                mean_function=Constant(y.shape[1]),
                likelihood=likelihood, **kwargs)

            self.embedders = torch.nn.ModuleList(embedders)
            self.xr = xr
            self.xg = xg
            self.samples_per_loss = 1

        def train(self, mode=True):
            """
            Set model in training mode
            Main reason: unseen embedding inputs are initialized randomly to break 
            symmetry.
            """
            super(cls, self).train(mode=mode)
            for e in self.embedders:
                e.unseen_policy = "random"

        def eval(self):
            """
            Set model in "evaluation" (test-time) mode
            Unseen embedding inputs are initialized to the prior
            """
            super(cls, self).eval()
            for e in self.embedders:
                e.unseen_policy = "prior"

        def loss(self, xr=None, xg=None, y=None, n_samples=None):
            """
            Loss function for variational inference, i.e. the negative ELBO
            """

            xg = xg if xg is not None else self.xg
            xr = xr if xr is not None else self.xr
            y = y if y is not None else self.Y
            s = n_samples if n_samples is not None else self.samples_per_loss

            prediction_elbo = torch.stack([self._prediction_elbo(xr, xg, y) 
                for _ in range(s)]).mean()

            kl = self._kl_general_inputs(xg)
            elbo = prediction_elbo - kl

            return -elbo

        def _embed(self, xg, mode=None):
            return _embed(self.embedders, xg, mode=mode)

        @staticmethod
        def _get_latent_dimensions(xg, d_xi):
            if d_xi is None:
                return [max(1, np.unique(xg_j).size - 1) for xg_j in xg.T]
            else:
                if isinstance(d_xi, int):
                    d_xi = [d_xi]
                if len(d_xi) != xg.shape[1]:
                    raise ValueError(
                        "Gave %i latent dimensions in d_xi, " % len(d_xi) +
                        "but there are %i different general input sets to be embedded."
                        % xg.shape[1]
                    )
                return d_xi

        def _prediction_elbo(self, xr, xg, y):
            """
            Evidence lower bound based on a single sample from the embedding modules.
            This doesn't include the KL term

            1) Embed general inputs.
            2) Use the usual GP loss.
            """

            xg_embed = self._embed(xg, mode="posterior")
            x_all = torch.cat([xr, xg_embed], dim=1)

            # Losses are NEGATIVE ELBOs!  Take their negation here!
            return -super().loss(x_all, y)

        def _kl_general_inputs(self, xg):
            """
            KL divergence on the embedding distirbutions KL(q||p) 
            (for variational inference)

            Do it analytically because we have Gaussians.

            There are N' KLs to compute, where N' is the number of unique entries in
            xg 
            """

            # Assume that we've already embedded every entry in xg so we don't need to 
            # worry about KeyErrors

            # Stack over the different embedders/general input spaces
            return torch.stack([
                self._kl_general_inputs_single_dimension(e, xg_j)
                for xg_j, e in zip(xg.T, self.embedders)
            ]).sum() 

        def _kl_general_inputs_single_dimension(self, embedder, xg):
            """
            KL divergence term for a single embedder input dimension.
            Pick implementation based on the type of Embedder 
            (Deterministic or probabilistic).

            :return: TensorType
            """

            return {
                DeterministicEmbedder: self._kl_xg_deterministic,
                GaussianEmbedder: self._kl_xg_gaussian
            }[type(embedder)](embedder, xg)

        def _kl_xg_deterministic(self, e, xg):
            """
            Fill-in regularization ("KL") term for Deterministic embedder.
            Since the posterior is a point estimate, use negative log-prob since
            lower is better (thinking of KL)
            
            :param e: The embedder
            :type e: Embedder
            :param xg: general inputs (a single dimension)
            :type xg: np.ndarray, 1D

            :return: TensorType(0D?)
            """

            # Stack over the unique inputs for that input space and the 
            # associated embedder.
            return -torch.stack([
                torch.distributions.Normal(torch.zeros(e.d_out), torch.ones(e.d_out)).\
                    log_prob(e.loc[xg_ij])
                for xg_ij in Embedder.clean_inputs(np.unique(xg))
            ]).sum()

        def _kl_xg_gaussian(self, e, xg):
            """

            :param e: The embedder
            :type e: Embedder
            :param xg: general inputs (a single dimension)
            :type xg: np.ndarray, 1D

            :return: TensorType(0D?)
            """

            # Stack over the unique inputs for that input space and the 
            # associated embedder.
            return torch.stack([
                torch.distributions.kl.kl_divergence(
                    torch.distributions.Normal(
                        e.loc[xg_ij], e.scale[xg_ij].transform()
                    ),
                    torch.distributions.Normal(
                        torch.zeros(e.d_out), torch.ones(e.d_out)
                    )
                )
                for xg_ij in Embedder.clean_inputs(np.unique(xg))
            ]).sum()

        # prediction functions are conditioned on a sample of the embedding modules'
        # posteriors.
        # (See the decorator.)

        @_embed_before_predicting
        def predict_f(self, *args, **kwargs):
            return super(cls, self).predict_f(*args, **kwargs)

        @_embed_before_predicting
        def predict_y(self, *args, **kwargs):
            return super(cls, self).predict_y(*args, **kwargs)

    return C


EGP = _make_embedding_gp_class(GPR)


class SafeGPR(GPR):
    """
    Extensions to GPR model to make it "safe" when there is no training data.
    """
    def compute_loss(self, *args, **kwargs):
        if self.num_data == 0:
            loss = TensorType([0.0])
            loss.requires_grad_(True)
            return loss
        else:
            return super().loss(*args, **kwargs)

    def optimize(self, *args, **kwargs):
        if self.num_data == 0: return
        super().optimize(*args, **kwargs)

    def _predict(self, x_new, diag=True, x=None):
        if self.num_data > 0:
            return super()._predict(x_new, diag=diag, x=x)
        # No data?  Use the prior.
        mean = self.mean_function(x_new)
        cov = self.kernel.Kdiag(x_new) if diag else self.kernel.K(x_new)

        return mean, cov

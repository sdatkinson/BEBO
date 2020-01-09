# File: embedders.py
# File Created: Sunday, 3rd November 2019 11:51:21 am
# Author: Steven Atkinson (212726320@ge.com)

"""
Embedding layers
"""

import abc

import numpy as np
import torch
from torch.distributions.transforms import ExpTransform
from gptorch.model import Param
from gptorch.util import torch_dtype


class Embedder(torch.nn.Module):
    """
    Embedding layer.
    Maps one-dimensional categorical variables to latents with real value dimensions.
    To embed a multi-dimensional input space, just make a different embedder for each
    input dimension.
    """

    def __init__(self, d_out):
        super().__init__()
        self._d_out = d_out

    def forward(self, x):
        # Clean inputs?
        x = Embedder.clean_inputs(x)

        return self._embed(x)

    @property
    def d_out(self):
        return self._d_out

    @staticmethod
    def clean_inputs(x):
        return [str(xi).replace(".", "dot") for xi in x]

    @abc.abstractmethod
    def _embed(self, x):
        raise NotImplementedError()


class DeterministicEmbedder(Embedder):
    """
    Deterministic embedding
    """
    def __init__(self, d_out):
        super().__init__(d_out)
        self._loc = torch.nn.ParameterDict()
        self._unseen_policy = "random"

    @property
    def loc(self):
        """
        Posterior mean dictionary
        """
        return self._loc

    @property
    def unseen_policy(self):
        return self._unseen_policy

    @unseen_policy.setter
    def unseen_policy(self, val):
        if not val in ["prior", "random"]:
            raise ValueError("Unhandled unseen policy %s" % val)
        self._unseen_policy = val

    def _embed(self, x):
        return torch.stack([self._embed_one(xi) for xi in x])

    def _embed_one(self, x):
        if x not in self._loc:
            if self.unseen_policy == "random":
                loc = torch.randn(self.d_out, dtype=torch_dtype)
            elif self.unseen_policy == "prior":
                loc = torch.zeros(self.d_out, dtype=torch_dtype)
            self._loc[x] = torch.nn.Parameter(loc)
        return self.loc[x]


class GaussianEmbedder(Embedder):
    """
    Embedding layer.
    Maps one-dimensional categorical variables to multivariate distributions in 
    Euclidean space.

    Stochastic embedding to Gaussian distributions
    """

    def __init__(self, d_out):
        super().__init__(d_out)
        # Key is an input;
        # Value is a location (mean) / scale (std) for a Gaussian.
        self._loc = torch.nn.ParameterDict()
        self._scale = torch.nn.ParameterDict()
        self._num_terms = 2
        # How we are currently doing embeddings: according to the prior or posterior
        self.mode = "prior"
        # When True, embed calls go randomly to samples from the output 
        # distribution.
        # When False, we embed to the mode of the distribution
        self.random = True
        # What to do with previously-unseen inputs ("random" or "prior")
        self.unseen_policy = "random"

    @property
    def loc(self):
        """
        Posterior mean dictionary
        """
        return self._loc

    @property
    def scale(self):
        """
        Posterior std dictionary
        """
        return self._scale

    def _embed(self, x):
        """
        Embed all inputs
        """

        return {"prior": self._embed_prior, "posterior": self._embed_posterior}[
            self.mode
        ](x, self._get_epsilon(x))

    def _embed_prior(self, x, epsilon):
        """
        Map an individual input
        """

        return torch.stack([epsilon[xi] for xi in x])

    def _embed_posterior(self, x, epsilon):
        for xi in x:
            if xi not in self.loc:
                # Randomly initialize to break symmetry and prevent posteriors from 
                # starting in the same spot
                self._loc[xi] = torch.nn.Parameter({
                    "random": torch.randn,
                    "prior": torch.zeros
                }[self.unseen_policy](self.d_out))
                self._scale[xi] = Param({
                    "random": lambda s: 0.1 * torch.ones(s),
                    "prior": torch.ones
                }[self.unseen_policy](self.d_out), transform=ExpTransform())

        # NB "epsilon" takes care of whether self.random or not.
        return torch.stack([
            self.loc[xi] + self.scale[xi].transform() * epsilon[xi]
            for xi in x
        ])

    def _get_epsilon(self, x):
        """
        We need a call to embed all data consistently.
        This function gets a single epsilon for each distinct entry in x so that
        we can make sure that e.g. "foo" always maps to the same thing.
        """
        return {xi: torch.randn(self.d_out) if self.random else torch.zeros(self.d_out) 
            for xi in np.unique(x)}

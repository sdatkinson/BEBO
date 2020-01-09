# File: main.py
# File Created: Saturday, 8th June 2019 9:38:18 am
# Author: Steven Atkinson (212726320@ge.com)

"""
Main woker script for carrying out BO experiments

Available systems:
* Forrester functions
* Toy system
"""

import os
import sys
import argparse
from time import time
from functools import partial

import numpy as np
import scipy
import matplotlib.pyplot as plt
from gptorch.util import TensorType
import torch

base_path = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.append(base_path)

from src.embedders import GaussianEmbedder, DeterministicEmbedder
from src.bayesian_optimization import WithFunction, StaticDataset
from src import systems

util_path = os.path.join(os.path.dirname(__file__), "..")
if util_path not in sys.path:
    sys.path.append(util_path)

from experiment_utils import doe, get_x_bounds, get_system, get_legacy_data
from experiment_utils import initialize_model, pre_train
from experiment_utils import train_function_egp, train_function_gptorch
from experiment_utils import predict_function_begp, predict_function_egp

torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system",
        type=str,
        default="forrester",
        choices=["forrester", "synthetic"],
        help="Which problem to run.",
    )
    parser.add_argument(
        "--num-legacy",
        type=int,
        default=5,
        help="How many legacy systems are available",
    )
    parser.add_argument(
        "--data-per-legacy",
        type=int,
        default=5,
        help="How many data from each legacy system are available",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BEGP",
        choices=["BEGP", "EGP", "BGP"],
        help="Which model to run",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for experiment"
    )
    parser.add_argument("--show", action="store_true", help="show running best")
    parser.add_argument("--save", action="store_true", help="Save results")

    return parser.parse_args()


def get_bo_functions(model_name, model, system):
    """
    functions for the metamodel being used.
    
    * Train
    * predict
    * append 
    """

    if model_name == "BEGP":
        return (
            partial(train_function_egp, model),
            partial(predict_function_begp, model, system),
            append_function_egp,
        )
    elif model_name == "EGP":
        return (
            partial(train_function_egp, model),
            partial(predict_function_egp, model, system),
            append_function_egp,
        )
    elif model_name == "BGP":
        return (model.fit, model.predict_y, model.append_data)
    else:
        raise ValueError("Unhandled model_name %s" % model_name)


# Train functions used during BO


def train_function_gpr(max_iter=100):
    if model.num_data == 0:
        return
    if model.num_data == 1:
        max_iter = min(max_iter, 5)
    train_function_gptorch(model, method="L-BFGS-B", max_iter=max_iter)


# Append functions used during BO


def append_function_egp(x_new, y_new):
    x_new, y_new = np.atleast_2d(x_new), np.atleast_2d(y_new)
    n_new = x_new.shape[0]
    xg_new = np.array([["0"] * system.general_dimensions] * n_new)

    model.xr = torch.cat((model.xr, TensorType(x_new)))
    model.xg = np.concatenate((model.xg, xg_new))
    model.Y = torch.cat((model.Y, TensorType(y_new)))


def append_function_gpr(x_new, y_new):
    model.X = torch.cat((model.X, TensorType(np.atleast_2d(x_new))))
    model.Y = torch.cat((model.Y, TensorType(np.atleast_2d(y_new))))


def append_function_bgp(x_new, y_new):
    model.x = torch.cat((model.X, TensorType(np.atleast_2d(x_new))))
    model.Y = torch.cat((model.Y, TensorType(np.atleast_2d(y_new))))


def train_callback():
    if system.real_dimensions == 1 and system.has_function:
        # Plot the posterior over the whole 1D input space
        x_test = np.linspace(0, 1, 100)
        m, v = bo.predict_function(x_test[:, np.newaxis])
        m, u = m.flatten(), 2.0 * np.sqrt(v.flatten())

        plt.figure()
        plt.fill_between(x_test, m - u, m + u, color=[0.8] * 3)
        plt.plot(x_test, m, label="Prediction")
        plt.plot(x_test, eval_function(x_test), label="Ground truth")
        plt.scatter(np.array(bo.x).flatten(), np.array(bo.y).flatten())
        plt.legend()
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        plt.show()
    if not system.has_function:
        # Validation plot
        m, v = bo.predict_function(bo.x_all)
        m, u = m.flatten(), 2.0 * np.sqrt(v.flatten())

        plt.figure()
        plt.errorbar(bo.y_all.flatten(), m, u, color="C0", linestyle="none", marker="o")
        plt.plot(plt.xlim(), plt.xlim(), linestyle="--", color="C1")
        plt.xlabel("Targets")
        plt.ylabel("Predictions")
        plt.show()


def show_results(system, bo):
    plt.plot(scipy.minimum.accumulate(bo.y))
    plt.xlabel("Number of evaluations")
    plt.ylabel("Running best")
    plt.show()
    if system.real_dimensions == 1:
        plt.figure()
        plt.scatter(
            np.array(bo.x).flatten(), np.array(bo.y).flatten(), c=np.arange(len(bo.y))
        )
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.show()
    if isinstance(bo, StaticDataset):
        plt.figure()
        # Colors: red (high) to blue (low)
        color = lambda a: a * np.array([1, 0, 0]) + (1 - a) * np.array([0, 0, 1])
        alpha = (bo.y_all - min(bo.y_all)) / (max(bo.y_all) - min(bo.y_all))
        for i, p in enumerate(np.array(bo.p_best).T):
            plt.plot(p, label="Datum %i" % i, color=color(alpha[i]))
        plt.xlabel("Iteration")
        plt.ylabel("p(best)")
        # plt.legend()
        plt.show()


if __name__ == "__main__":
    t0 = time()
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    system = get_system(args.system, args.seed)
    xr, xg, y = get_legacy_data(system, args)

    model = initialize_model(xr, xg, y, args.model)
    pre_train(model, args.model)

    # A few things to get ready for the BO class:
    assert system.general_dimensions == 1

    if system.has_function:
        eval_function = lambda xr: system(xr, [0])
        xr_bounds = np.tile(np.array([get_x_bounds()]), (system.real_dimensions, 1))
        bo = WithFunction(
            xr_bounds, *get_bo_functions(args.model, model, system), eval_function
        )
        n_points = 10
    else:
        assert system.general_dimensions == 1
        x_all, y_all = system.get_data([0])
        bo = StaticDataset(x_all, y_all, *get_bo_functions(args.model, model, system))
        n_points = x_all.shape[0]

    # bo.register_pre_selection_callback(train_callback)
    bo.add_points(n_points, verbose=True)

    if args.show:
        show_results(system, bo)
    if args.save:
        path = os.path.join(
            os.path.dirname(__file__),
            "output",
            "%s_legacy_%i_data_%i"
            % (args.system, args.num_legacy, args.data_per_legacy),
            "results",
            args.model,
        )
        filename = os.path.join(path, "%i.npy" % args.seed)
        print("Saving results to %s" % filename)
        if not os.path.isdir(path):
            os.makedirs(path)
        np.save(filename, np.array(bo.y).flatten())
    print("Done.  Run time = %i secs" % int(time() - t0))

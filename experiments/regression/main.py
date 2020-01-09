# File: main.py
# File Created: Saturday, 8th June 2019 9:38:18 am
# Author: Steven Atkinson (212726320@ge.com)

"""
Main woker script for regression problems.
"""

import os
import sys
import argparse
from time import time
from functools import partial
import json

import numpy as np
import scipy
import matplotlib.pyplot as plt
from gptorch.models.gpr import GPR
from gptorch.kernels import Rbf
from gptorch.util import TensorType
import gptorch
import torch

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if base_path not in sys.path:
    sys.path.append(base_path)

from src.models import EGP, SafeGPR as GPR
from src.embedders import GaussianEmbedder, DeterministicEmbedder
from src.bayesian_optimization import WithFunction, StaticDataset
from src import systems
from src.util import train_test_split

util_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if util_path not in sys.path:
    sys.path.append(util_path)

from experiment_utils import doe, get_x_bounds, get_system, get_legacy_data
from experiment_utils import pre_train, train_function_gptorch
from experiment_utils import predict_function_begp, predict_function_egp
import experiment_utils

torch.set_num_threads(1)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--system", 
        type=str, 
        default="forrester", 
        choices=["forrester", "synthetic"],
        help="Which problem to run."
    )
    parser.add_argument(
        "--current-task",
        type=int,
        default=0,
        help="For static, specify which task is the non-legacy task"
    )
    parser.add_argument(
        "--num-legacy",
        type=int,
        default=5,
        help="How many legacy systems are available"
    )
    parser.add_argument(
        "--data-per-legacy",
        type=int,
        default=5,
        help="How many data from each legacy system are available"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="BEGP", 
        choices=["BEGP", "EGP", "BGP"], 
        help="Which model to run"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for experiment")
    parser.add_argument(
        "--train-current", 
        type=int, 
        default=1, 
        help="Number of training examples from the current task"
    )
    parser.add_argument("--show", action="store_true", help="Show results")
    parser.add_argument("--save", action="store_true", help="Save results")

    return parser.parse_args()


def get_data(system, args):
    xr_leg, xg_leg, y_leg = get_legacy_data(system, args)
    xr_current, xg_current, y_current = _get_current_task_data(system, args)

    xrc_train, xr_test, xgc_train, xg_test, yc_train, y_test = train_test_split(
        xr_current, 
        xg_current, 
        y_current, 
        train_size=args.train_current, 
        random_state=args.seed
    )

    xr_train = np.concatenate((xr_leg, xrc_train))
    xg_train = np.concatenate((xg_leg, xgc_train))
    y_train = np.concatenate((y_leg, yc_train))

    return xr_train, xr_test, xg_train, xg_test, y_train, y_test


def _get_current_task_data(system, args):
    assert system.general_dimensions == 1, "For now."

    if system.has_function:
        n = args.train_current + 1000  # 1000 for testing.
        xr = doe(n, system.real_dimensions)
        y = system(xr, [0])  # 0 for current task by convention.
    else:
        xr, y = system.get_data([0])

    xg = np.array([["0"] * system.general_dimensions] * xr.shape[0])

    return xr, xg, y


def initialize_model(xr, xg, y, model_type):
    if model_type == "GP":
        assert xg.shape[1] == 1
        i = np.where(xg.flatten() == "0")[0]
        xr, xg, y = xr[i], xg[i], y[i]
        return GPR(
            xr, 
            y, 
            Rbf(xr.shape[1], ARD=True), 
            likelihood=gptorch.likelihoods.Gaussian(variance=0.001)
        )
    else:
        return experiment_utils.initialize_model(xr, xg, y, model_type)


def train(model, model_type):
    if model_type == "BGP":
        model.fit()
    else:
        pre_train(model, model_type)


def predict(model, model_type, system, xr):
    """
    Predictions, assuming we're predicting on the current task, task "0".
    """
    return {
        "GP": model.predict_y,
        "BGP": partial(_bgp_predict_wrapper, model),
        "EGP": partial(predict_function_egp, model, system),
        "BEGP": partial(predict_function_begp, model, system)
    }[model_type](xr)


def _bgp_predict_wrapper(model, *args, **kwargs):
    """
    Just to ensure that the outgoing shapes are right (i.e. 2D).
    """

    mean, cov = model.predict_y(*args, **kwargs)
    if len(mean.shape) == 1:
        mean = mean[:, None]
    if len(cov.shape) == 1:
        cov = cov[:, None]
    return mean, cov


def get_performance(means, stds, targets):
    """
    Compute prediction metrics MNLP, MAE, and RMSE
    """

    mnlp = -np.median(scipy.stats.norm.logpdf(targets, loc=means, scale=stds))
    mae = np.abs(targets - means).mean()
    rmse = np.sqrt(((targets - means) ** 2).mean())
    return {"MNLP": mnlp, "MAE": mae, "RMSE": rmse}


def show_results(inputs, means, stds, targets):
    assert targets.shape[1] == 1
    means, stds, targets = means.flatten(), stds.flatten(), targets.flatten()
    unc = 2.0 * stds

    plt.figure()
    plt.errorbar(targets, means, unc, linestyle="none", marker="o")
    plt.plot(plt.xlim(), plt.xlim(), linestyle="--", color="C1")
    plt.xlabel("Targets")
    plt.ylabel("Predictions")
    plt.show()

    if inputs.shape[1] == 1:
        inputs = inputs.flatten()
        i = np.argsort(inputs)
        plt.fill_between(inputs[i], (means - unc)[i], (means + unc)[i], color=[0.8] * 3)
        plt.plot(inputs[i], targets[i], marker=".", color="C1", linestyle="none")
        plt.show()


if __name__ == "__main__":
    t0 = time()
    args = parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    system = get_system(args.system, args.current_task)
    xr_train, xr_test, xg_train, xg_test, y_train, y_test = get_data(system, args)

    model = initialize_model(xr_train, xg_train, y_train, args.model)
    train(model, args.model)
    
    # Assert xg_test is current task?
    pred_mean, pred_std = predict(model, args.model, system, xr_test)
    performance = get_performance(pred_mean, pred_std, y_test)

    if args.show:
        print(
            "Performace:\n MNLP : %.6e\n MAE  : %.6e\n RMSE : %.6e" %
            (performance["MNLP"], performance["MAE"], performance["RMSE"])
        )
        print("show_results()...")
        show_results(xr_test, pred_mean, pred_std, y_test)
    if args.save:
        path = os.path.join(
            os.path.dirname(__file__), 
            "output", 
            "%s_currenttask_%i_legacytasks_%i_legacydata_%i_currentdata_%i" % (
                args.system, 
                args.current_task, 
                args.num_legacy, 
                args.data_per_legacy, 
                args.train_current
            ), 
            "results", 
            args.model
        )
        filename = os.path.join(path, "%i.json" % args.seed)
        print("Saving results to %s" % filename)
        if not os.path.isdir(path):
            os.makedirs(path)
        with open(filename, "w") as f:
            json.dump(performance, f)
    print("Done.  Run time = %i secs" % int(time() - t0))
    
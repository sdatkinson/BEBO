# File: postprocess.py
# File Created: Saturday, 2nd November 2019 11:34:59 am
# Author: Steven Atkinson (212726320@ge.com)

"""
Post-processing the results of experiments for the figures in the paper.
"""

import os

import numpy as np
import scipy
import matplotlib.pyplot as plt


def _get_raw_data(problem_dir, model):
    def _path(i):
        return os.path.join(problem_dir, model, "%i.npy" % i)

    data = {}
    for i in range(0, 101):
        if os.path.isfile(_path(i)):
            data[i] = np.load(_path(i))
    print("%s: found %i runs" % (model, len(data)))

    # Pad shorter experiments with NaNs
    n = max([len(d) for d in data.values()])
    for i, d in data.items():
        d = np.concatenate((d, np.nan * np.zeros(n - len(d))))
        data[i] = d

    return data


def get_all_raw_and_best(problem_dir, models):
    """
    Pull raw data for all models, then compute teh running best for each 
    run.

    :param models: models we want to look at.
    :type models: List(str)

    :return: dict with one entry for each model with the following fields:
        * "raw": the raw data (np.ndarray, shape=n_runs x n_experiments)
        * "best": The running best for each run
    """
    results = {}
    for model in models:
        results[model] = {}
        results[model]["raw"] = _get_raw_data(problem_dir, model)
        results[model]["valid"] = bool(results[model]["raw"])
        if results[model]["valid"]:
            results[model]["best"] = {
                key: scipy.minimum.accumulate(val)
                for key, val in results[model]["raw"].items()
            }

    return results


def get_best_offset(results, models, relative_to="all"):
    """
    Get "best_offset" field for the results dict for each mdoel in models

    :param results: the ersults dict
    :param models: list of model strings
    :param relative_to: None, "all", or "seed"
    """

    all_seeds = set()
    for model in models:
        all_seeds = all_seeds.union(results[model]["best"].keys())

    if relative_to is None:
        results["offset"] = {key: 0.0 for key in all_seeds}
    elif relative_to == "all":
        best = min(
            [
                min([np.nanmin(v) for v in results[model]["best"].values()])
                for model in models
            ]
        )
        results["offset"] = {key: best for key in all_seeds}
    elif relative_to == "seed":
        results["offset"] = {}
        for seed in all_seeds:
            best = None
            for model in models:
                if seed in results[model]["best"]:
                    if best is None or np.nanmin(results[model]["best"][seed]) < best:
                        best = np.nanmin(results[model]["best"][seed])
            results["offset"][seed] = best

    # Now compute offset bests:
    for model in models:
        results[model]["best_offset"] = {
            key: val - results["offset"][key]
            for key, val in results[model]["best"].items()
        }


def get_statistics(results, models):
    """
    Get results of the "best_offset" field
    """

    for model in models:
        if not results[model]["valid"]:
            continue

        mtx = np.stack([v for v in results[model]["best_offset"].values()])
        results[model]["median"] = np.median(mtx, axis=0)
        results[model]["upper"] = np.percentile(mtx, 90.0, axis=0)
        results[model]["lower"] = np.percentile(mtx, 10.0, axis=0)


def plot_results(results, models, curve_style="fill"):
    """
    :param relative_to: Specify whether we offset the curves relative to 
        something.  Chocies are:
        * None (no offset)
        * "seed": all models are offset relative to whatever the best result for
          that seed was.  Useful for Finite dataset problems (pump, additive) 
          where different seeds switch what the current task is, and different
          tasks have different minima!
        * "all": offset everyone according to the best result that was found for
          any model, any seed.
    :type relative_to: str
    :param curve_style: "fill" or "raw"
    """
    plt.figure()

    for model in models:
        if not results[model]["valid"]:
            continue
        n = np.arange(1, results[model]["median"].size + 1)
        if curve_style == "fill":
            plt.fill_between(
                n,
                results[model]["lower"],
                results[model]["upper"],
                color=colors[model],
                alpha=0.2,
            )
        else:
            for curve in results[model]["best_offset"].values():
                plt.plot(n, curve, color=colors[model], alpha=0.3)
        plt.plot(n, results[model]["median"], color=colors[model], label=model)
    plt.xlabel("Experiments")
    plt.ylabel("Best")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    problem = "forrester_legacy_5_data_5"
    # problem = "synthetic_legacy_5_data_5"
    # problem = "pump_legacy_-1_data_-1"
    # problem = "additive_legacy_-1_data_-1"

    problem_dir = os.path.join(os.path.dirname(__file__), "output", problem, "results")
    models = ["BGP", "EGP", "BEGP"]
    relative_to = "seed"  # None, "seed", or "all"
    curve_style = "fill"  # "fill" or "raw"

    colors = {
        "BGP": "C0",  # Bayesian GP
        "EGP": "C1",  # (Deterministic) Embedding GP
        "BEGP": "C2",  # Bayesian embedding GP
        "GP": "C3"  # (Single-task) GP
        # BoTorch (multi-task) GP?
        # Multi-fidelity GP?
    }

    results = get_all_raw_and_best(problem_dir, models)
    get_best_offset(results, models, relative_to=relative_to)
    get_statistics(results, models)
    plot_results(results, models, curve_style=curve_style)

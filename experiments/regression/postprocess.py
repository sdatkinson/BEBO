# File: postprocess.py
# File Created: Tuesday, 12th November 2019 9:19:41 am
# Author: Steven Atkinson (212726320@ge.com)

"""
Post-processing for regression

Plots have # experiments on the current task as the abcissa and performance 
metric (RMSE, MAE, MNLP) as the ordinate.
We plot the mean and 2 stds confidence interval over all available splits
for each data point.
(We don't do median+quantiles because we usually only have ~10 splits per 
setting)

We put all models on the same plot.

A different plot is made for each problem/current task/legacy configuration 
(#legacy tasks, #data per legacy task).
"""

import os
import sys
from argparse import ArgumentParser
import json

import matplotlib.pyplot as plt
import numpy as np

base_path = os.path.dirname(__file__)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("problem", type=str)
    parser.add_argument("legacy_tasks", type=int)
    parser.add_argument("legacy_data", type=int, help="Data per task")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--limit-mnlp", action="store_true")
    parser.add_argument(
        "--shuffle-tasks", action="store_true", help="Assigns random labels to tasks"
    )

    return parser.parse_args()


def problem_dirname(
    problem, current_task, legacy_tasks, legacy_data, current_data=None, model=None
):

    s = [
        "%s_currenttask_%i_legacytasks_%i_legacydata_%i"
        % (problem, current_task, legacy_tasks, legacy_data)
    ]
    if current_data is not None:
        s[0] = s[0] + "_currentdata_%i" % current_data

        if model is not None:
            s += ["results", model]

    return os.path.join(base_path, "output", *s)


def get_num_tasks(args):
    """
    Find out how many tasks we have
    """

    current_task = 0
    while os.path.isdir(
        problem_dirname(
            args.problem,
            current_task,
            args.legacy_tasks,
            args.legacy_data,
            current_data=0,
        )
    ):
        current_task += 1

    return current_task


def all_current_tasks(args, models):
    """
    Start at current task = 0 and keep goign until we don't find results for
    one.
    """

    num_tasks = get_num_tasks(args)
    task_labels = np.arange(num_tasks)
    if args.shuffle_tasks:
        # Ensure repeatability
        np.random.seed(42)
        task_labels = np.random.permutation(task_labels)

    for current_task, task_label in enumerate(task_labels):
        plot_current_task_all_metrics(
            args.problem,
            current_task,
            args.legacy_tasks,
            args.legacy_data,
            models,
            args.save,
            args.show,
            limit_mnlp=args.limit_mnlp,
            task_label=task_label,
        )


def plot_current_task_all_metrics(
    problem,
    current_task,
    legacy_tasks,
    legacy_data,
    models,
    save,
    show,
    limit_mnlp=False,
    task_label=None,
):
    """
    Make plots for all metrics (RMSE, MAE, MNLP) for this problem + current task
    + legacy settings.

    :param limit_mnlp: If true, prevent us from showing outrageously-bad (high) MNLPs.
    :param task_label: If provided, re-label this task in the plots as the 
        provided label.
    """

    metrics = ["RMSE", "MAE", "MNLP"]
    task_label = current_task if task_label is None else task_label

    # First, get the performance data:
    performance = {}
    for model in models.keys():
        performance[model] = get_performance_all_n(
            problem, current_task, legacy_tasks, legacy_data, model, metrics
        )

    # Next, plots
    for metric in metrics:
        plt.figure()
        for model, params in models.items():
            n, mu, err = get_plot_arrays(performance[model], metric)
            if len(n) > 0:
                plt.plot(
                    n, mu, label=model, marker=params["marker"], color=params["color"]
                )
                plt.fill_between(
                    n, mu - err[0], mu + err[1], color=params["color"], alpha=0.3
                )
        plt.legend()
        plt.xlabel("$n_{train}$")
        plt.ylabel(metric)
        if metric == "MNLP" and limit_mnlp:
            ylim = {
                "forrester": [-10, 100],
                "synthetic": [-10, 100],
            }[problem]
            plt.ylim(*ylim)
        plt.title("%s, task %i" % (problem, task_label))
        if save:
            plotdir = os.path.join(base_path, "output", "plots", problem)
            if not os.path.isdir(plotdir):
                os.makedirs(plotdir)
            filename = "%s_currenttask_%i_legacytasks_%i_legacydata_%i_%s.pdf" % (
                problem,
                task_label,
                legacy_tasks,
                legacy_data,
                metric,
            )
            plt.savefig(os.path.join(plotdir, filename))
        if show:
            plt.show()
        else:
            plt.close()


def get_performance_all_n(
    problem, current_task, legacy_tasks, legacy_data, model, metrics
):
    """
    :return: dict:
        n: 
            seed:
                (RMSE, MAE, MNLP)
    """

    n_min, n_max = 0, 8
    performance = {}
    for n in range(n_min, n_max + 1):
        dirname = problem_dirname(
            problem,
            current_task,
            legacy_tasks,
            legacy_data,
            current_data=n,
            model=model,
        )
        if os.path.isdir(dirname):
            performance[n] = {}
            for filename in os.listdir(dirname):
                seed = int(filename.split(".")[0])
                with open(os.path.join(dirname, filename), "r") as f:
                    performance[n][seed] = json.load(f)

    return performance


def get_plot_arrays(performance, metric):
    """
    :return: (np.ndarray, np.ndarray, np.ndarray) x, y, err
    """

    x, y, err = [], [], []
    for n in performance.keys():
        x.append(n)
        vals = np.array([d[metric] for d in performance[n].values()])
        mu = np.median(vals)
        lower = np.percentile(vals, 10.0)
        upper = np.percentile(vals, 90.0)
        y.append(mu)
        err.append([mu - lower, upper - mu])

    i = np.argsort(x)

    x = np.array(x)[i]
    y = np.array(y)[i]
    err = np.array(err)[i].T  # (2, N)

    return x, y, err


if __name__ == "__main__":
    args = parse_args()
    models = {
        "BGP": {"color": "C0", "marker": "o"},
        "EGP": {"color": "C1", "marker": "s"},
        "BEGP": {"color": "C2", "marker": "^"},
    }
    all_current_tasks(args, models)

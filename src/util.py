# File: util.py
# File Created: Tuesday, 1st October 2019 12:37:46 pm
# Author: Steven Atkinson (212726320@ge.com)

import numpy as np
from sklearn.model_selection import train_test_split as _tts


def leave_one_in(*args, seed=None):
    """
    Use for one-shot learning with small datasets because train-test split may
    pick the same split frequently.

    :param args: Things to be split.
    :param seed: Which datum to leave in, actually.  None=pick randomly.
    """
    n = None
    for arg in args:
        if n is None:
            n = len(arg)
        else:
            if len(arg) != n:
                raise ValueError("Not all args have the same number of entries")
    
    if seed is None:
        seed = np.random.randint(0, high=n)
    if seed >= n:
        raise ValueError(
            "Tried to leave out entry %i, but only %i entries exist." %
            (seed, n)
        )

    i_all = set(np.arange(n).tolist())
    i_train = [seed]
    i_test = list(i_all - set(i_train))

    out = []
    for arg in args:
        out.append(arg[i_train])
        out.append(arg[i_test])

    return tuple(out)


def train_test_split(*args, **kwargs):
    """
    Train-test split that allows for train_size=0 ("zero-shot" learning)
    """
    
    # Zero-shot
    if "train_size" in kwargs and kwargs["train_size"] == 0:  
        outputs = []
        for a in args:
            outputs += [a[:0], a]
        return outputs
    # One-shot--use leave-one-in
    elif "train_size" in kwargs and kwargs["train_size"] == 1:
        seed = kwargs["random_state"] if "random_state" in kwargs else None
        return leave_one_in(*args, seed=seed)
    elif "test_size" in kwargs and kwargs["test_size"] == 0:
        raise ValueError("Test size must be positive")
    else:  # Nonzero-shot
        return _tts(*args, **kwargs)

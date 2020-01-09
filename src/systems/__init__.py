# File: __init__.py
# File Created: Sunday, 3rd November 2019 10:21:29 am
# Author: Steven Atkinson (212726320@ge.com)

"""
Systems are all given the call signature:

f(x, i)

Where x is the real-valued inputs, and i is a list of integers indexing the 
particular system to be queried.
See the documentation of each system for more information on how many dimensions
are allowed in i and what their interpretations are.

Generally, i=[0, 0, ...] is the "held-out" system of interest.
"""

from .base import System
from .forrester import Forrester
from .synthetic import Synthetic

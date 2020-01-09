# File: base.py
# File Created: Sunday, 3rd November 2019 10:37:35 am
# Author: Steven Atkinson (212726320@ge.com)

"""
Base class declaring the struture of System classes
"""

import abc

from sklearn.preprocessing import MinMaxScaler, StandardScaler


class System(abc.ABC):
    def __init__(self):
        # How many dimensions the real-valued input has
        self.real_dimensions = None
        # How many dimensions the general input has:
        self.general_dimensions = None
        # How many different elements are in each dimension
        # List of ints, len=self.general_dimensions
        self.num_types = None
        # Does this system have a function we can call at new locations?
        # (Or is it just a static dataset?)
        self.has_function = False


class WithFunction(System):
    def __init__(self):
        super().__init__()
        self.has_function = True

    def __call__(self, x, i):
        """
        :param x: Inputs (real-valued)
        :param i: The general input
        :type i: list of ints
        """

        if not len(i) == self.general_dimensions:
            raise ValueError("General dimensions")
        if not all([ii < n for ii, n in zip(i, self.num_types)]):
            raise ValueError("Too high an index in general dim")
    
        return self._call(x, i)

    @abc.abstractmethod
    def _call(self, x, i):
        raise NotImplementedError()


class StaticDataset(System):
    """
    Systems where all we have is a dataset.
    """
    def __init__(self):
        super().__init__()

        # dict where key is the general input, 
        # value is tuple with input & output data arrays.
        self._data = {}

    def get_data(self, xg):
        """
        Convert list-of-ints xg into string

        :param xg: general input
        :type xg: list[int]

        :return: (np.ndarray, np.ndarray) inputs & outputs
        """
        return self.data[self._clean_xg(xg)]

    def _clean_xg(self, xg):
        return str(xg)

    @abc.abstractmethod
    def _load_data(self):
        """
        Load data in and return a with keys that are the general dimensions,
        values are a pair of np.ndarrays (inputs, outputs)
        """
        raise NotImplementedError("Implement loading in data.")
    
    def _get_scalers(self, use_held_out=False):
        """
        Load all of the data and scale.

        :param use_held_out: if True, use all families; if False, hold out the 
            first family (family 2) since we "don't know" what those are.
        :type use_held_out: bool
        """

        x, y = self._get_scaler_data(use_held_out)

        x_scaler, y_scaler = MinMaxScaler(), StandardScaler()
        x_scaler.fit(x)
        y_scaler.fit(y)

        return x_scaler, y_scaler

    @abc.abstractmethod
    def _get_scaler_data(self, use_held_out):
        """
        :return: (np.ndarray, np.ndarray) x & y
        """
        raise NotImplementedError("Implement getting legacy data for scalers")

import logging

import numpy as np
import pandas as pd

from ppodd.utils import pd_freq, unwrap_array

logger = logging.getLogger(__name__)


class Lazy(object):
    """
    A hacky deferral wrapper for assigning dataset constants to processing
    module outputs potentially before they've actually been set on the dataset.
    This works in this context as variable metadata are allowed to be either
    literal or callable.
    """

    def __init__(self, parent):
        """
        Initialize a class instance.

        Args:
            parent: the object whos state we want to defer.
        """
        self.parent = parent

    def __getitem__(self, item):
        """
        Implement [x], potentially deferred to a callable.

        Args:
            item: the item to get from the parent.

        Returns:
            either <item> got from parent, or a callable deferring this
            operation.
        """

        def _callable():
            try:
                return self.parent[item]
            except KeyError:
                return None

        try:
            return self.parent[item]
        except KeyError:
            return _callable

    def __getattr__(self, attr):
        """
        Implement .x, potentially deferred to a callable.

        Args:
            attr: the attribute to get from the parent.

        Returns:
            either <attr> from parent, or a callable deferring this
            operation.
        """
        try:
            return getattr(self.parent, attr)
        except AttributeError:
            return lambda: getattr(self.parent, attr)


class DatasetNormalizer:

    def __init__(self, dataset, frequency: int):
        """
        Initialize a class instance.

        Args:
            dataset: the dataset to normalize.
        """
        self.dataset = dataset
        self.frequency = frequency

    def __enter__(self):
        """
        Enter the context manager.

        Returns:
            self.
        """
        for variable in self.dataset.variables:
            self.dataset[variable].frequency = self.frequency

        return self.dataset

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context manager.

        Args:
            exc_type: the exception type, if any.
            exc_value: the exception value, if any.
            traceback: the traceback, if any.

        Returns:
            None.
        """
        for variable in self.dataset.variables:
            self.dataset[variable].frequency = None

        return None


def resample_variable(
    variable,
    frequency: int,
    apply: str = "mean",
    start_time: pd.Timestamp | None = None,
    end_time: pd.Timestamp | None = None,
) -> pd.Series:
    """
    Resample a variable to a new frequency.

    Args:
        variable: the variable to resample.
        frequency: the new frequency to resample to.
        apply: the function to apply to the resampled data.

    Returns:
        the resampled variable.
    """
    data = variable.to_series()
    circular = getattr(variable, "circular", False)

    if start_time is None:
        start_time = variable.t0

    if end_time is None:
        end_time = variable.t1

    _index = pd.date_range(start=start_time, end=end_time, freq=pd_freq[frequency])

    if circular:
        data = unwrap_array(data)

    data = data.resample(pd_freq[frequency]).apply(apply).reindex(_index)

    if circular:
        data[~np.isnan(data)] %= 360

    return data  # type: ignore

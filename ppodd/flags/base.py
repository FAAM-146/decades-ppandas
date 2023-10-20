"""
Provides an abstract base class for flagging modules. Flagging modules run
after the main processing modules allowing modification to data quality flags
using data which may not be available at the time processing modules are run.
"""

import abc
import datetime
import logging

import numpy as np
import pandas as pd

from ppodd.decades import flags


logger = logging.getLogger(__name__)

def flatten(container):
    """
    Recursively flatten a (list or tuple) container.
    """

    for i in container:
        if isinstance(i, (list, tuple)):
            for i in flatten(i):
                yield i
        else:
            yield i


class FlaggingBase(abc.ABC):
    """
    An abstract base class from which flagging modules should inherit.
    """

    # Names of required input variables
    inputs = []
    prerequisites = []
    flagged = []

    # Super simple data for testing use
    test_flag = np.arange(5)
    test_index = pd.date_range(
        start=datetime.datetime.now(), periods=5, freq='S'
    )

    def __init__(self, dataset):
        """
        Initialize an instance.

        Args:
            dataset: a DecadesDataset from which the processing is being run.
        """
        self.dataset = dataset
        self.flags = {}

    def __str__(self):
        return f'[Flagging module: {self.__class__.__name__}]'

    def ready(self, run_modules):
        """
        Decide if the module is ready to run. It can be run if all of its
        inputs are available in the parent dataset.

        Returns:
            True if the module is ready to run, False otherwise.
        """
        for _input in self.inputs:
            if _input not in self.dataset.variables:
                logger.debug(f'{_input} not in {self.dataset.variables}')
                return False
        for prereq in self.prerequisites:
            if prereq not in run_modules:
                logger.debug(f'{prereq} not in {run_modules}')
                return False
        return True

    def add_mask(self, var, mask, meaning, description=None):
        """
        Add a new flag mask to a specified variable
        """
        self.dataset[var].flag.add_mask(mask, meaning, description)
        try:
            self.flags[var].append((meaning.replace(' ', '_'), description))
        except KeyError:
            self.flags[var] = [(meaning.replace(' ', '_'), description)]

    @abc.abstractmethod
    def _flag(self):
        """Add extra flag info to derived variables."""

    @abc.abstractmethod
    def _get_flag(self):
        """Get the flag information"""

    def _get_downstream(self, var):
        """
        Recursively-ish get all variables downstream of a given variable. These
        will also need to have their flag modified if the flag of the given
        variable has been modified.
        """

        downstream = [list(i.declarations) for i in
                      self.dataset.processor.completed_modules
                      if var in i.inputs
                      and var not in i.ignored_upstream_flags]

        for i in downstream:
            downstream += self._get_downstream(i)

        return list(flatten(downstream))

    def _reflag(self, var, downstream):
        """
        For each variable which has had its flag midified by this module, we
        need to add or modify a dependency_is_flagged flag to all of the
        downstream variables.
        """
        try:
            var = self.dataset[var]
        except KeyError:
            logger.debug(f'{var} not in dataset')
            return
            
        for _ds in downstream:
            ds = self.dataset[_ds]
            logger.debug(f'reflagging {_ds} from {var}')

            # If the upstream variable is lower frequency
            # than the downstream? We need to [b/f]fill the reindexed flag
            # rather than filling with 0

            try:
                _new_flag = var.flag().reindex(ds.flag().index)

                if ds.frequency > var.frequency:
                    _new_flag = _new_flag.fillna(method='ffill')
                    _new_flag = _new_flag.fillna(method='bfill')
                else:
                    _new_flag = _new_flag.fillna(0)

                _new_flag[_new_flag > 0] = 1
            except TypeError:
                logger.error(f'Failed to produce new flag from {ds.name}')
                continue

            try:
                ds.flag.add_mask(_new_flag, flags.DEPENDENCY)
            except AttributeError:
                # Classic rather than bitmask flag
                try:
                    flag_val = ds.flag.meanings[flags.DEPENDENCY]
                except KeyError:
                    try:
                        flag_val = max(list(ds.flag.meanings)) + 1
                    except ValueError:
                        flag_val = 1

                    ds.flag.add_meaning(
                        flag_val, flags.DEPENDENCY
                        ('A dependency, used in the derivation of this '
                         'variable has a non-zero flag.')
                    )

                ds.flag.add_flag(flag_val * _new_flag)


    def flag(self):
        """
        Run the flagging code, then reflag all parameters which depend on the
        newly flagged variable.
        """

        self._flag()

        for var in self.flagged:
            downstream = []

            # Recursively get all downstream variables
            _downstream = list(set(
                [i for i in self._get_downstream(var)]
            ))

            # Filter out any downstream variables which either didn't make it
            # into the dataset, or have no associated flag.
            for i in _downstream:
                try:
                    if self.dataset[i].flag is not None:
                        downstream.append(i)
                except Exception:
                    pass

            logger.debug(f'Downstream for {var}: {", ".join(downstream)}')

            self._reflag(var, downstream)


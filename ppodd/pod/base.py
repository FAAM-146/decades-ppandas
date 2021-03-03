"""
This module provides an abstract base class for processing modules. Processing
modules should subclass PPBase to be included in the processing.
"""
# pylint: disable=invalid-name, too-many-arguments

import abc
import datetime

import numpy as np
import pandas as pd

from ppodd.utils import pd_freq, unwrap_array
from ppodd.decades import DecadesDataset, DecadesVariable

class PPBase(abc.ABC):
    """
    PPBase is an abstract base class from which processing modules should
    inherit to be picked up by the processing code.
    """

    freq = pd_freq

    inputs = []
    test = {}

    def __init__(self, dataset, test_mode=False):
        """
        Initialize a class instance

        Args:
            dataset: the DecadesDataset from which the processing is running

        Kwargs:
            test_mode: if true, run the module in test mode, for unit testing,
                building documentation etc.
        """
        self.dataset = dataset
        self.outputs = {}
        self.declarations = {}
        self.test_mode = test_mode
        self.declare_outputs()
        self.d = None

    def __str__(self):
        """
        Implement str() - simply return the name of the class.
        """
        return self.__class__.__name__

    @abc.abstractmethod
    def declare_outputs(self):
        """
        Add outputs to be written. This is run during initialization of the
        module, and should contain a call to self.declare for each output
        variable.
        """

    @abc.abstractmethod
    def process(self):
        """Do the actual postprocessing - entry point for the module"""

    def declare(self, name, **kwargs):
        """
        Declare the output variables that the processing module is going to
        create.

        Args:
            name: the name of the declared variable

        Kwargs: key-value pairs used to initialize the decades variable.
        """
        if name in self.dataset.variables:
            if not self.dataset.allow_overwrite:
                raise ValueError(
                    f'Cannot declare {name}, as it already exists in Dataset'
                )

            self.dataset.remove(name)

        self.declarations[name] = kwargs

    def finalize(self):
        """
        Finalization tasks: ensure all declared outputs have been written and
        propogate the outputs to the calling DecadesDataset.
        """

        for declaration in self.declarations:
            if declaration not in self.outputs:
                raise RuntimeError(
                    'Output declared but not written: {}'.format(
                        declaration
                    )
                )

        for name, output in self.outputs.items():
            # Apply any modifications specified in self.dataset._variable_mods
            # - canonically as specified in the flight constants file
            # Note that we shouldn't really be accessing what's nominally a
            # private method of dataset.
            #
            # pylint: disable=protected-access
            # TODO: add a public interface to DecadesDataset
            if name in self.dataset._variable_mods:
                for key, value in self.dataset._variable_mods[name].items():
                    setattr(output, key, value)

            # And append the output to the dataset
            self.dataset.add_output(output)

        self.outputs = None
        self.d = None

    @staticmethod
    def onto(dataframe, index, limit=1, period=None):
        """
        Project a dataframe onto another index, interpolating across any gaps
        introduced to a given limit.

        Args:
            dataframe: the Pandas DataFrame or Series to reindex.
            index: the index to reindex dsataframe onto.

        Kwargs:
            limit [1]: maximum number of consecutive gaps to fill, passed to
                index.interpolate
            period [None]: passed to index.interpolate.

        Returns:
            the DataFrame dataframe reindexed onto index.
        """

        return dataframe.reindex(
            index.union(dataframe.index).sort_values()
        ).interpolate(
            'time', limit=limit, period=period
        ).loc[index]

    def get_dataframe(self, method='outerjoin', index=None, limit=1,
                      circular=None, exclude=None):
        """
        Create a dataframe containing all of the inputs required to run the
        processing module. The dataframe is stored in the class attribute 'd'.

        Kwargs:
            method ['outerjoin']: specify how to merge variables into a single
                dataframe. Should be one of 'outerjoin' (default), which uses a
                union of indexes, or 'onto', which puts all variables onto a
                specified index. If 'onto', index should preferably be
                specified. If index is not given, the index of the first
                required input will be used.
            index [None]: the index to reindex input variables onto. Used when
                method='onto'. If method is onto but no index is given, the
                index of the first required input will be used.
            limit [1]: the maximum number of consecutive empty datapoints to
                interpolate over when reindexing.
            circular [None]: an iterable of the names of any input variables
                which are congruent to themselves mod 360.
            exclude [None]: an iterable of the names of input variables to
                exclude from the resulting dataframe.
        """

        if circular is None:
            circular = []

        if exclude is None:
            exclude = []

        df = pd.DataFrame()

        _inputs = [
            i for i in self.inputs if i not in self.dataset.constants
            and i not in exclude
        ]

        if method == 'outerjoin':

            # Create a joined dataframe
            for _input in _inputs:
                df = df.join(self.dataset[_input]().dropna(), how='outer')

            # Ensure the dataframe has a continuous index at the frequency of
            # the highest frequency input.
            _freq = np.max([self.dataset[i].frequency for i in _inputs])
            df = df.reindex(pd.date_range(
                start=df.index[0], end=df.index[-1], freq=pd_freq[_freq]
            ))

        elif method == 'onto':
            # Interpolate onto a given index when creating the dataframe

            if index is None:
                df = self.dataset[_inputs[0]]()
                index = df.index
                _start = 1
            else:
                df = pd.DataFrame(index=index)
                _start = 0

            for _input in _inputs[_start:]:
                _input_name = _input

                if _input in circular:
                    # Deal with periodic (e.g. heading) data

                    _tmp = self.dataset[_input_name]()
                    _data = _tmp.values.copy()

                    _input = pd.DataFrame([], index=_tmp.index)

                    _input[_input_name] = _data
                    _input[_input_name] = unwrap_array(_input[_input_name])

                else:
                    _input = self.dataset[_input_name]()

                # Interpolate onto the instance dataframe
                df[_input_name] = _input.reindex(
                    index.union(
                        _input.index
                    ).sort_values()
                ).interpolate(
                    'linear', limit=limit
                ).loc[index]

                if _input_name in circular:
                    df[_input_name] %= 360

        self.d = df

    def add_output(self, variable):
        """
        Add an output variable. Initially outputs are stored in the processing
        module 'outputs' attribute (a list). These are propagated to the parent
        dataset when self.finalize is called.

        Args:
            variable: the output variable to add.
        """

        # All outputs must be declared before being added (see self.declare)
        if variable.name not in self.declarations:
            raise RuntimeError('Output {} has not been declared'.format(
                variable.name
            ))

        # Set all of the variable attribute specified in self.declare
        for item, value in self.declarations[variable.name].items():
            setattr(variable, item, value)

        try:
            # Find where the first and last non-nan data points are, and trim
            # the data to this period.
            good_start = np.min(np.where(~np.isnan(variable.data)))
            _df = variable()
            start_time = _df.index[good_start]
            end_time = _df.index[-1]
            variable.trim(start_time, end_time)

        except ValueError:
            # If a value is raised, there's no good data. In this case there's
            # no point including the data in the final product, so set its
            # write attribute to false.
            if not self.test_instance:
                variable.write = False
                print('Warning: no good data: {}'.format(variable.name))

        self.outputs[variable.name] = variable

    def ready(self):
        """
        Determine if the module can run. A module can run if all of its
        required inputs are present in its parent dataset.

        Returns:
            a 2-tuple, the first element of which is a boolean indicating if
                the module is ready. If the module able to run, the second
                element will be None, otherwise it will be a list of variables
                which are required to run the module, but which are not
                available in the parent dataset.
        """
        _missing_variables = []

        for _name in self.inputs:
            _inputs = (
                self.dataset.variables +
                list(self.dataset.constants.keys())
            )
            if _name not in _inputs:
                _missing_variables.append(_name)

        if _missing_variables:
            return False, _missing_variables

        return True, None

    @classmethod
    def test_instance(cls, dataset=None):
        """
        Return a test instance of a postprocessing module, initialized with a
        DecadesDataset containing the modules test data.
        """
        now = datetime.datetime.now().replace(microsecond=0)

        if dataset is None:
            d = DecadesDataset(now.date())
        else:
            d = dataset

        if callable(cls.test):
            # pylint: disable=not-callable
            _test = cls.test()
        else:
            _test = cls.test

        for key, val in _test.items():
            _type, _values = val

            if _type == 'const':
                d.constants[key] = _values

            elif _type == 'data':
                var = DecadesVariable(
                    pd.Series(
                        _values,
                        index=pd.date_range(
                            start=d.date, freq='S', periods=len(_values)
                        )
                    ),
                    name=key,
                    frequency=1
                )

                d.add_input(var)

        return cls(d, test_mode=True)

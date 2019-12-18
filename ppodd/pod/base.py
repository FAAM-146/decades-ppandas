import abc
import datetime

import numpy as np
import pandas as pd

from ..utils import pd_freq
from ..decades import DecadesDataset, DecadesVariable

class PPBase(abc.ABC):

    freq = {
        1: '1S',
        2: '500L',
        4: '250L',
        8: '125L',
        16: '62500U',
        32: '31250000N',
        64: '15625000N',
        128: '7812500N',
        256: '3906250N'
    }

    inputs = []
    test = {}

    def __init__(self, dataset, test_mode=False):
        self.dataset = dataset
        self.outputs = {}
        self.declarations = {}
        self.test_mode = test_mode
        self.declare_outputs()
        self.d = None

    def __str__(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def declare_outputs(self):
        """Add outputs to be written"""

    @abc.abstractmethod
    def process(self):
        """Do the actual postprocessing"""

    def declare(self, name, **kwargs):
        """
        Declare the output variables that the processing module is going to
        create.
        """
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
            if name in self.dataset._variable_mods:
                for key, value in self.dataset._variable_mods[name].items():
                    setattr(output, key, value)

            # And append the output to the dataset
            self.dataset.outputs.append(output)

    def onto(self, dataframe, index, limit=1, period=None):
        return dataframe.reindex(
            index.union(dataframe.index).sort_values()
        ).interpolate(
            'time', limit=limit, period=period
        ).loc[index]

    def get_dataframe(self, method='outerjoin', index=None, limit=1,
                      circular=None):

        if circular is None:
            circular = []

        df = pd.DataFrame()

        _inputs = [i for i in self.inputs if i not in self.dataset.constants]

        if method == 'outerjoin':
            for _input in _inputs:
                df = df.join(self.dataset[_input].data.dropna(), how='outer')

        elif method == 'onto':
            import matplotlib.pyplot as plt

            if index is None:
                df = self.dataset[_inputs[0]].data
                index = df[_inputs[0]].data.index
                _start = 1
            else:
                df = pd.DataFrame(index=index)
                _start = 0

            for _input in _inputs[_start:]:
                _input_name = _input

                if _input in circular:

                    _mask = ~np.isnan(
                        np.deg2rad(self.dataset[_input_name].data)
                    )

                    _data = self.dataset[_input_name].data.values.copy()
                    _data[_mask] = np.rad2deg(
                        np.unwrap(np.deg2rad(_data[_mask]))
                    ) % 360

                    _input = pd.DataFrame(
                        [],
                        index=self.dataset[_input_name].data.index
                    )

                    _input[_input_name] = _data

                else:
                    _input = self.dataset[_input_name].data

                df[_input_name] = _input.reindex(
                    index.union(
                        _input.index
                    ).sort_values()
                ).interpolate(
                    'linear', limit=limit
                ).loc[index]

        self.d = df

    def add_output(self, variable, flag=None):

        if variable.name not in self.declarations:
            raise RuntimeError('Output {} has not been declared'.format(
                variable.name
            ))

        for item, value in self.declarations[variable.name].items():
            setattr(variable, item, value)

 #       flag_name = variable.add_flag()
 #       if flag is not None:
 #           variable.flag = flag

 #       # Force the variable to the declared frequency
 #       variable._df = variable._df.asfreq(pd_freq[variable.frequency])

 #       # Flag any gaps caused by asfreq as 3
 #       variable._df.loc[~np.isfinite(variable._df[flag_name]), flag_name] = 3

        try:
            good_start = np.min(np.where(~np.isnan(variable.data)))
            variable._df = variable._df.iloc[good_start:]
        except ValueError:
            variable.write = False
            print('Warning: no good data: {}'.format(variable.name))

        self.outputs[variable.name] = variable

    def ready(self):
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

        mod = cls(d, test_mode=True)

        if callable(cls.test):
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

        return mod

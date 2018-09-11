import abc

import numpy as np
import pandas as pd


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

    def __init__(self, dataset):
        self.dataset = dataset
        self.outputs = {}
        self.declarations = {}
        self.declare_outputs()
        self.d = None

    def __str__(self):
        return 'PP module: {}'.format(self.__class__.__name__)

    @abc.abstractmethod
    def declare_outputs(self):
        """Add outputs to be written"""

    @abc.abstractmethod
    def process(self):
        """Do the actual postprocessing"""

    @abc.abstractmethod
    def inputs(self):
        """Define required inputs"""

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
            self.dataset.outputs.append(output)

        print('  -> {}'.format(', '.join(self.outputs.keys())))

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

        if method == 'outerjoin':
            for _input in self.inputs:
                df = df.join(self.dataset[_input].data, how='outer')

        elif method == 'onto':

            if index is None:
                df = self.dataset[self.inputs[0]].data
                index = df[self.inputs[0]].data.index
                _start = 1
            else:
                df = pd.DataFrame(index=index)
                _start = 0

            for _input in self.inputs[_start:]:
                _input_name = _input

                if _input in circular:
                    _data = np.rad2deg(
                        np.unwrap(np.deg2rad(self.dataset[_input_name].data))
                    )

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
                    'time', limit=limit
                ).loc[index]

        self.d = df

    def add_output(self, variable, flag=None):

        if variable.name not in self.declarations:
            raise RuntimeError('Output {} has not been declared'.format(
                variable.name
            ))

        variable.long_name = self.declarations[variable.name]['long_name']
        variable.units = self.declarations[variable.name]['units']
        variable.frequency = self.declarations[variable.name]['frequency']
        variable.standard_name = self.declarations[variable.name]['standard_name']

        flag_name = variable.add_flag()
        if flag is not None:
            variable.flag = flag

        variable._df.loc[~np.isfinite(variable._df[variable.name]), flag_name] = 3

        self.outputs[variable.name] = variable

    def ready(self):
        for _name in self.inputs:
            if _name not in self.dataset.variables:
                return False
        return True

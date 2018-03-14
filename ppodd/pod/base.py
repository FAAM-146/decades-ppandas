import abc

import numpy as np
import pandas as pd


class PPBase(abc.ABC):

    def __init__(self, dataset):
        self.dataset = dataset
        self.outputs = {}
        self.declarations = {}
        self.input_names = self.inputs()
        self.declare_outputs()

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
                raise RuntimeError('** Output declared but not written')

        for name, output in self.outputs.items():
            self.dataset.outputs.append(output)

    def get_dataframe(self, method='outerjoin', limit=1):
        df = pd.DataFrame()

        if method == 'outerjoin':
            for _input in self.input_names:
                df = df.join(self.dataset[_input], how='outer')

        elif method == 'onto':
            df = self.dataset[self.input_names[0]]
            for _input in self.input_names[1:]:
                df[_input] = self.dataset[_input].reindex(
                    df[self.input_names[0]].index.union(
                        self.dataset[_input].index).sort_values()
                ).interpolate(
                    'time', limit=limit
                ).loc[df[self.input_names[0]].index]

        return df

    def add_output(self, variable):

        if variable.name not in self.declarations:
            raise RuntimeError('Output {} has not been declared'.format(
                variable.name
            ))

        variable.long_name = self.declarations[variable.name]['long_name']
        variable.units = self.declarations[variable.name]['units']
        variable.frequency = self.declarations[variable.name]['frequency']
        variable.standard_name = self.declarations[variable.name]['standard_name']

        flag_name = '{}_FLAG'.format(variable.name)
        if flag_name not in variable:
            variable[flag_name] = 0

        variable[flag_name] = np.around(variable[flag_name])

        variable.columns = [variable.name, flag_name]

        self.outputs[variable.name] = variable

    def ready(self):
        for _name in self.input_names:
            if _name not in self.dataset.variables:
                return False
        return True

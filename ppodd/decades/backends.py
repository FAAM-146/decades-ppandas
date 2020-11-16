import gc
import glob
import os
import pickle
import sqlite3 as sql
import numpy as np
import pandas as pd


class DecadesBackend(object):

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def _dlu_from_variable(self, variable):
        return variable.name.split('_')[0]

    def decache(self):
        pass

    def collect_garbage(self, required_inputs):
        pass

    def cleanup(self):
        pass

    def trim(self, start, end):
        pass

    def add_input(self, variable):
        raise NotImplementedError

    def add_output(self, variable):
        self.outputs.append(variable)

    def remove(self, name):
        raise NotImplementedError

    def clear_outputs(self):
        self.outputs = []

    @property
    def variables(self):
        return [i.name for i in self.inputs + self.outputs]


class DefaultBackend(DecadesBackend):
    def __getitem__(self, item):
        for _var in self.inputs + self.outputs:
            if _var.name == item:
                return _var

        raise KeyError('No input: {}'.format(item))

    def trim(self, start, end):
        for _var in self.inputs:
            _var.trim(start, end)

    def remove(self, name):
        for var in self.inputs:
            if var.name == name:
                self.inputs.remove(var)
                return
        for var in self.outputs:
            if var.name == name:
                self.outputs.remove(var)
                return

    def add_input(self, var):
        if var.name not in [i.name for i in self.inputs]:
            self.inputs.append(var)
            return

        self[var.name].merge(var)
        return
        try:
            self[var.name]._df.loc[
                var._df.index, var.name
            ] = var._df[var.name]

        except KeyError:
            # The dataframe does not include all of the indicies present in
            # this variable, therefore we need to reindex

            # Create the new index as the unique union between the
            # dataframe and the variable. 
            _index = self[var.name].index.union(
                var.index
            ).sort_values().unique()

            # Reindex the dataframe
            _df = self[var.name]._df.reindex(_index)
            self[var.name]._df = _df

            # And merge in the variable
            self[var.name]._df.loc[
                var._df.index, var.name
            ] = var._df[var.name]


    def collect_garbage(self, required_inputs):
        _copied_inputs = self.inputs.copy()

        for var in self.inputs:
            if var.name not in required_inputs:
                self.inputs.remove(var)
                print('GC: {}'.format(var))
                del var

        gc.collect()

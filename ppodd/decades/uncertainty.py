import pandas as pd

from ppodd.utils import pd_freq

class Uncertainty:

    def __init__(self, data, var):
        self.data = data
        self.frequecy = var.frequency
        self.t0 = var.t0
        self.t1 = var.t1
        self.name = f'{var.name}_CU'
        self.units = var.units
        self.long_name = f'Combined uncertainty estimate for {var.name}'
        self.coverage_content_type = 'auxiliaryInformation'

    def __call__(self):
        return pd.Series(
            self.data, index=self.index, name=self.name  
        )

    @property
    def index(self):
        return pd.date_range(
            start=self.t0, end=self.t1, freq=pd_freq[self.frequecy]
        )

    def trim(self, start, end):
        _index = self.index
        loc = (_index >= start) & (_index <= end)
        self.data = self.data.loc[loc]
        self.t0 = start
        self.t1 = end
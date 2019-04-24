import pandas as pd

from ..decades import DecadesVariable
from .base import PPBase


class RioWeightOnWheels(PPBase):

    inputs = ['PRTAFT_wow_flag']

    def declare_outputs(self):
        self.declare(
            'WOW_IND',
            units='-',
            frequency=1,
            long_name='Weight on wheels indicator',
            standard_name=None
        )

    def process(self):

        self.get_dataframe()
        wow = pd.DataFrame()

        wow['WOW_IND'] = self.d['PRTAFT_wow_flag'].asfreq('1S')

        self.add_output(
            DecadesVariable(
                wow,
                name='WOW_IND',
            )
        )

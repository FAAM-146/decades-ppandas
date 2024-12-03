import pandas as pd

from ..decades import DecadesVariable
from .base import PPBase, register_pp
from .shortcuts import _c, _o, _z


@register_pp('core')
class WeightOnWheels(PPBase):
    """
    This module simply provides the aircraft weight-on-wheels flag, recorded on
    the rear core console. A value of 1 indicates weight on wheels (i.e. the
    aircraft is on the ground) and a value of 0 indicates no weight on wheels
    (i.e. the aircraft is airborne).
    """

    inputs = ['PRTAFT_wow_flag']

    @staticmethod
    def test():
        return {
            'PRTAFT_wow_flag': ('data', _c([_o(20), _z(60), _o(20)]), 1)
        }

    def declare_outputs(self):
        self.declare(
            'WOW_IND',
            frequency=1,
            units='1',
            long_name='Weight on wheels indicator'
        )

    def process(self):
        """
        Processing entry point.
        """

        self.get_dataframe()
        wow = pd.DataFrame()

        wow['WOW_IND'] = self.d['PRTAFT_wow_flag'].asfreq('1s')
        wow['WOW_IND'].bfill(inplace=True)
        wow['WOW_IND'].ffill(inplace=True)

        self.add_output(
            DecadesVariable(
                wow,
                name='WOW_IND',
                flag=None
            )
        )

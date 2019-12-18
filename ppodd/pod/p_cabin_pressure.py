import numpy as np

from ..decades import DecadesVariable
from .base import PPBase
from .shortcuts import _c, _l

class CabinPressure(PPBase):
    """
    Calculate cabin pressure..
    """

    inputs = [
        'CALCABP',
        'CORCON_cabin_p'
    ]

    @staticmethod
    def test():
        return {
            'CALCABP': ('const', [2.75, 3.3e-2, -2.5e-10]),
            'CORCON_cabin_p': (
                'data', _c([_l(30000, 25000, 50), _l(25000, 35000, 50)])
            )
        }

    def declare_outputs(self):
        """
        Declare the outputs that are going to be written by this module.
        """

        self.declare(
            'CAB_PRES',
            units='hPa',
            frequency=1,
            long_name='Cabin Pressure'
        )

    def bounds_flag(self):
        """
        Create a flag based on simple bounds checking. Flag as 2 wherever the
        data is outside a specified min/max.
        """

        valid_min = 650
        valid_max = 1050

        self.d['BOUNDS_FLAG'] = 0
        self.d.loc[self.d['CAB_PRES'] < valid_min, 'BOUNDS_FLAG'] = 1
        self.d.loc[self.d['CAB_PRES'] > valid_max, 'BOUNDS_FLAG'] = 1

    def process(self):
        """
        Processing entry point.
        """
        _cals = self.dataset['CALCABP'][::-1]
        self.get_dataframe()

        self.d['CAB_PRES'] = np.polyval(_cals, self.d['CORCON_cabin_p'])

        self.bounds_flag()

        cab_press = DecadesVariable(self.d['CAB_PRES'])

        cab_press.flag.add_meaning(0, 'data good')
        cab_press.flag.add_meaning(1, 'pressure out of range')

        cab_press.flag.add_flag(self.d['BOUNDS_FLAG'])

        self.add_output(cab_press)

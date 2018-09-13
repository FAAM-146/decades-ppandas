import numpy as np

from ..decades import DecadesVariable
from .base import PPBase

class CabinPressure(PPBase):
    """
    Calculate cabin pressure..
    """

    inputs = [
        'CALCABP',
        'CORCON_cabin_p'
    ]

    def declare_outputs(self):
        """
        Declare the outputs that are going to be written by this module.
        """

        self.declare(
            'CAB_PRES',
            units='hPa',
            frequency=1,
            number=579,
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
        self.d.loc[self.d['CAB_PRES'] < valid_min, 'BOUNDS_FLAG'] = 2
        self.d.loc[self.d['CAB_PRES'] > valid_max, 'BOUNDS_FLAG'] = 2

    def process(self):
        """
        Processing entry point.
        """
        _cals = self.dataset['CALCABP'][::-1]
        self.get_dataframe()

        self.d['CAB_PRES'] = np.polyval(_cals, self.d['CORCON_cabin_p'])

        self.bounds_flag()

        s9 = DecadesVariable(self.d['CAB_PRES'])
        s9.add_flag(self.d['BOUNDS_FLAG'])
        self.add_output(s9)

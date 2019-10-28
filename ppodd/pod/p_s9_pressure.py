import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from .base import PPBase

class S9Pressure(PPBase):
    """
    Calculate static pressure from the S9 fuselage ports.
    """

    inputs = [
        'CALS9SP',
        'CORCON_s9_press'
    ]

    def declare_outputs(self):
        """
        Declare the outputs that are going to be written by this module.
        """

        self.declare(
            'P9_STAT',
            units='hPa',
            frequency=32,
            number=579,
            long_name='Static pressure from S9 fuselage ports',
            standard_name='air_pressure'
        )

    def bounds_flag(self):
        """
        Create a flag based on simple bounds checking. Flag as 2 wherever the
        data is outside a specified min/max.
        """

        valid_min = 100
        valid_max = 1050

        self.d['BOUNDS_FLAG'] = 0
        self.d.loc[self.d['P9_STAT'] < valid_min, 'BOUNDS_FLAG'] = 2
        self.d.loc[self.d['P9_STAT'] > valid_max, 'BOUNDS_FLAG'] = 2

    def process(self):
        """
        Processing entry point.
        """
        _cals = self.dataset['CALS9SP'][::-1]
        self.get_dataframe()

        self.d['P9_STAT'] = np.polyval(_cals, self.d['CORCON_s9_press'])

        self.bounds_flag()

        s9 = DecadesVariable(self.d['P9_STAT'], flag=DecadesBitmaskFlag)
        s9.flag.add_mask(self.d['BOUNDS_FLAG'], flags.OUT_RANGE)
        self.add_output(s9)

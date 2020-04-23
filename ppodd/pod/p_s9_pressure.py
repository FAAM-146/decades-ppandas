import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from .base import PPBase
from ppodd.utils.calcs import sp_mach
from .shortcuts import _o

class S9Pressure(PPBase):
    """
    Calculate static pressure from the S9 fuselage ports.
    """

    inputs = [
        'CALS9SP',
        'S9_PE_C',
        'CORCON_s9_press',
        'PS_RVSM',
        'Q_RVSM',
    ]

    @staticmethod
    def test():
        return {
            'CALS9SP': ('const', [-130, .035, 1.85e-9]),
            'CORCON_s9_press': ('data', 850 * _o(100))
        }

    def declare_outputs(self):
        """
        Declare the outputs that are going to be written by this module.
        """

        self.declare(
            'P9_STAT',
            units='hPa',
            frequency=32,
            long_name='Static pressure from S9 fuselage ports',
            standard_name='air_pressure'
        )

        self.declare(
            'P9_STAT_U',
            units='hPa',
            frequency=32,
            long_name='Static pressure from S9 fuselage ports, uncorrected',
            standard_name='air_pressure',
            write=False
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
        _cors = self.dataset['S9_PE_C']

        self.get_dataframe(
            method='onto',
            index=self.dataset['CORCON_s9_press'].index
        )

        self.d['P9_STAT'] = np.polyval(_cals, self.d['CORCON_s9_press'])
        self.d['P9_STAT_U'] = np.polyval(_cals, self.d['CORCON_s9_press'])
        self.d['MACH'] = sp_mach(self.d.Q_RVSM, self.d.PS_RVSM)

        # A position error correction, based on minimising the difference
        # between s9 and RVSM pressures in mach#.
        self.d['PE_CORR'] = (
            _cors[0] * self.d.MACH ** _cors[1]
        )

        self.d.P9_STAT -= self.d.PE_CORR

        self.bounds_flag()

        for name in ('P9_STAT', 'P9_STAT_U'):
            var = DecadesVariable(self.d[name], flag=DecadesBitmaskFlag)
            var.flag.add_mask(self.d['BOUNDS_FLAG'], flags.OUT_RANGE)
            self.add_output(var)

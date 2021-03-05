"""
Provides a postprocessing module for the S10 pressure transducer. See class
docstring for more info.
"""
import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from .base import PPBase
from .shortcuts import _o

S10_VALID_MIN = 100
S10_VALID_MAX = 1050


class S10Pressure(PPBase):
    r"""
    Calculate static pressure from the S10 fuselage ports. Static pressure is
    calulated by applying a polynomial transformation to the DLU signal. The
    coefficients, specified in the flight constants parameter ``CALS10SP``,
    combine both the DLU and pressure transducer calibrations.

    Data are flagged when they are considered out of range.
    """

    inputs = [
        'CALS10SP',
        'S10SP_SN',
        'CORCON_s10_press'
    ]

    @staticmethod
    def test():
        """
        Return dummy inputs for testing.
        """
        return {
            'CALS10SP': ('const', [-130, .035, 1.85e-9]),
            'S10SP_SN': ('const', 'xxxx'),
            'CORCON_s10_press': ('data', 850 * _o(100)),
        }

    def declare_outputs(self):
        """
        Declare the outputs that are going to be written by this module.
        """

        self.declare(
            'P10_STAT',
            units='hPa',
            frequency=32,
            long_name='Static pressure from S10 fuselage ports',
            standard_name='air_pressure',
            sensor_manufacturer='Rosemount Aerospace Inc.',
            sensor_model='1201F2',
            sensor_serial_number=self.dataset['S10SP_SN']
        )

    def bounds_flag(self):
        """
        Create a flag based on simple bounds checking. Flag as 2 wherever the
        data is outside a specified min/max.
        """

        self.d['BOUNDS_FLAG'] = 0
        self.d.loc[self.d['P10_STAT'] < S10_VALID_MIN, 'BOUNDS_FLAG'] = 1
        self.d.loc[self.d['P10_STAT'] > S10_VALID_MAX, 'BOUNDS_FLAG'] = 1

    def process(self):
        """
        Processing entry point.
        """
        _cals = self.dataset['CALS10SP'][::-1]

        self.get_dataframe(
            method='onto',
            index=self.dataset['CORCON_s10_press'].index
        )

        self.d['P10_STAT'] = np.polyval(_cals, self.d['CORCON_s10_press'])

        self.bounds_flag()

        var = DecadesVariable(
            self.d['P10_STAT'], name='P10_STAT', flag=DecadesBitmaskFlag
        )

        var.flag.add_mask(
            self.d['BOUNDS_FLAG'], flags.OUT_RANGE,
            f'Pressure outside acceptable limits [{S10_VALID_MIN}, '
            f'{S10_VALID_MAX}]'
        )
        self.add_output(var)

"""
This module provides a postprocessing module which calculates static pressure
from the S9 fuselage port. See class docstring for more info.
"""

import numpy as np

from ..utils.calcs import sp_mach
from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from .base import PPBase
from .shortcuts import _o

S9_VALID_MIN = 100
S9_VALID_MAX = 1050


class S9Pressure(PPBase):
    r"""
    Calculate static pressure from the S9 fuselage ports. Static pressure is
    calulated by applying a polynomial transformation to the DLU signal. The
    coefficients, specified in the flight constants parameter ``CALS9SP``,
    combine both the DLU and pressure transducer calibrations. Additionally, a
    Mach dependent 'position error' correction term is applied, aimed at
    accounting for the unknown position error associated with the S9 port,
    derived by minimising errors between the S9 and RVSM static pressure
    measurements. This correction is of the form

    .. math::
        \Delta P = \alpha M ^ \beta,

    with parameters :math:`\alpha` and :math:`\beta` specified in the flight
    constants parameter ``S9_PE_C``.

    Data are flagged when they are considered out of range.
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
        """
        Return dummy input data for testing.
        """
        return {
            'CALS9SP': ('const', [-130, .035, 1.85e-9]),
            'S9_PE_C': ('const', [1, 1]),
            'CORCON_s9_press': ('data', 850 * _o(100)),
            'PS_RVSM': ('data', 850 * _o(100)),
            'Q_RVSM': ('data', 150 * _o(100))
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
            standard_name='air_pressure',
            sensor_manufacturer='Rosemount Aerospace Inc.',
            sensor_model='1201F2',
            sensor_serial_number=self.dataset.lazy['S9SP_SN']
        )

        self.declare(
            'P9_STAT_U',
            units='hPa',
            frequency=32,
            long_name='Static pressure from S9 fuselage ports, uncorrected',
            standard_name='air_pressure',
            sensor_manufacturer='Rosemount Aerospace Inc.',
            sensor_model='1201F2',
            sensor_serial_number=self.dataset.lazy['S9SP_SN'],
            write=False
        )

    def bounds_flag(self):
        """
        Create a flag based on simple bounds checking. Flag as 2 wherever the
        data is outside a specified min/max.
        """

        self.d['BOUNDS_FLAG'] = 0
        self.d.loc[self.d['P9_STAT'] < S9_VALID_MIN, 'BOUNDS_FLAG'] = 2
        self.d.loc[self.d['P9_STAT'] > S9_VALID_MAX, 'BOUNDS_FLAG'] = 2

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
            var.flag.add_mask(
                self.d['BOUNDS_FLAG'], flags.OUT_RANGE,
                f'Pressure outside acceptable limits [{S9_VALID_MIN}, '
                f'{S9_VALID_MAX}]'
            )
            self.add_output(var)

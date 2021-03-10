"""
Provides a postprocessing module for the Heimann radiometer. See the class
docstring for more information.
"""

import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..utils.conversions import celsius_to_kelvin
from .base import PPBase
from .shortcuts import _c, _o, _z

VALID_MIN = celsius_to_kelvin(-20)
VALID_MAX = celsius_to_kelvin(40)


class Heimann(PPBase):
    r"""
    Processing for the Heimann Radiometer. The Heimann outputs a voltage with a
    range of 0 - 10 V corresponding to an inferred brightness temperature of
    :math:`-50` - :math:`50` :math:`^\circ\text{C}`. This module simply applies
    a linear transformation to the counts recorded on the DLU to convert counts
    :math:`\rightarrow` volts :math:`\rightarrow` temperature. During a
    calibration, temperature from the black body are reported. Parameters for
    the linear transformations are taken from the flight constant parameters
    ``HEIMCAL`` for the Heimann and ``PRTCCAL`` for the PRT on the black body.
    """

    inputs = [
        'PRTCCAL',
        'HEIMCAL',
        'SREG',
        'CORCON_heim_t',
        'CORCON_heim_c',
        'WOW_IND'
    ]

    @staticmethod
    def test():
        """
        Return some dummy input data for testing.
        """
        return {
            'PRTCCAL': ('const', [-20, 2e-3, 0]),
            'HEIMCAL': ('const', [-45, 3e-3, 0]),
            'SREG': ('data', _z(100), 1),
            'CORCON_heim_t': ('data', 2e5 * _o(100), 4),
            'CORCON_heim_c': ('data', 185e2 * _o(100), 4),
            'WOW_IND': ('data', _c([_o(20), _z(80)]), 1)
        }

    def declare_outputs(self):
        """
        Declare module output variables.
        """
        self.declare(
            'BTHEIM_U',
            units='K',
            frequency=4,
            long_name=('Uncorrected brightness temperature from the Heimann '
                       'radiometer'),
            instrument_manufacturer='Heitronics',
            instrument_model='KT19.82',
            instrument_serial_number=self.dataset.lazy['HEIM_SN']
        )

    @staticmethod
    def temperature(cals, series):
        """
        Conversion from Heimann is simply a quadratic fit.

        Args:
            cals: constants for the quadratic fit, least significant first.
            series: the timeseries of Heimann data to convert to a temperature.

        Returns:
            Heimann temperature, in Kelvin.
        """
        return celsius_to_kelvin(
            cals[0] + cals[1] * series + cals[2] * series ** 2
        )

    def flag(self):
        """
        Create a flag for Heimann temperature.

        Flagging regime:
            In calibration
            Data missing
            Aircraft on ground
            Aata outside user limits
        """

        self.d['RANGE_FLAG'] = 0
        self.d['WOW_FLAG'] = 0
        self.d['CAL_FLAG'] = 0
        self.d['MISSING_FLAG'] = 0

        self.d.loc[self.d.BTHEIM_U < VALID_MIN, 'RANGE_FLAG'] = 1
        self.d.loc[self.d.BTHEIM_U > VALID_MAX, 'RANGE_FLAG'] = 1
        self.d.loc[self.d.WOW_IND == 1, 'WOW_FLAG'] = 1
        self.d.loc[self.d.INCAL == 1, 'CAL_FLAG'] = 1
        self.d.loc[~np.isfinite(self.d.BTHEIM_U), 'MISSING FLAG'] = 1

    def process(self):
        """
        Processing entry point.
        """
        vector_binrep = np.vectorize(np.binary_repr)

        self.get_dataframe()

        # Back / forward fill nans in the signal register. (The signal register
        # is at 2 Hz, while the Heimann data is at 4 Hz)
        self.d.SREG.fillna(method='bfill', inplace=True)
        self.d.SREG.fillna(method='ffill', inplace=True)

        # Back / forward fill nans in WOW flag.
        self.d.WOW_IND.fillna(method='bfill', inplace=True)
        self.d.WOW_IND.fillna(method='ffill', inplace=True)

        # The Heiman calibration is signified by the least significant bit in
        # the signal register. This is somewhat legacy, but...
        self.d['INCAL'] = [
            int(i[-1]) for i in vector_binrep(self.d.SREG.astype(int))
        ]

        # Temperature from the Heimann when measuring
        measuring = self.temperature(
            self.dataset['HEIMCAL'], self.d.CORCON_heim_t
        )

        # Temperature from the BB when in calibration
        caling = self.temperature(
            self.dataset['PRTCCAL'], self.d.CORCON_heim_c
        )

        # Combined measurement / calibration timeseries
        combined = measuring
        combined.loc[self.d.INCAL == 1] = caling
        combined.name = 'BTHEIM_U'
        self.d['BTHEIM_U'] = combined

        # Create data flags
        self.flag()

        heimann = DecadesVariable(combined, flag=DecadesBitmaskFlag)

        heimann.flag.add_mask(
            self.d.WOW_FLAG, flags.WOW, 'The aircraft is on the ground'
        )
        heimann.flag.add_mask(
            self.d.RANGE_FLAG, flags.OUT_RANGE,
            (f'Brightness temperature is outside the range {VALID_MIN:0.2f} - '
             f'{VALID_MAX:0.2f} K')
        )
        heimann.flag.add_mask(
            self.d.CAL_FLAG, flags.CALIBRATION,
            ('The Heimann is in a calibration cycle. Black body temperature '
             'is being reported')
        )
        heimann.flag.add_mask(
            self.d.MISSING_FLAG, flags.DATA_MISSING,
            'Data are expected but not present'
        )

        self.add_output(heimann)

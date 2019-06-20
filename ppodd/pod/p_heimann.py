import numpy as np

from ..decades import DecadesVariable
from ..utils.conversions import celsius_to_kelvin
from .base import PPBase


class Heimann(PPBase):

    VALID_MIN = celsius_to_kelvin(-20)
    VALID_MAX = celsius_to_kelvin(40)

    inputs = [
        'PRTCCAL',
        'HEIMCAL',
        'SREG',
        'CORCON_heim_t',
        'CORCON_heim_c',
        'WOW_IND'
    ]

    def declare_outputs(self):
        self.declare(
            'BTHEIM_U',
            units='K',
            frequency=4,
            number=537,
            long_name=('Uncorrected brightness temperature from the Heimann '
                       'radiometer')
        )

    def temperature(self, cals, series):
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
            In calibration              -> 3
            Data missing                -> 3
            Aircraft on ground          -> 2
            Aata outside user limits    -> 1
        """

        self.d['FLAG'] = 0

        self.d.loc[self.d.BTHEIM_U < self.VALID_MIN, 'FLAG'] = 1
        self.d.loc[self.d.BTHEIM_U > self.VALID_MAX, 'FLAG'] = 1
        self.d.loc[self.d.WOW_IND == 1, 'FLAG'] = 2
        self.d.loc[self.d.INCAL == 1, 'FLAG'] = 3
        self.d.loc[~np.isfinite(self.d.BTHEIM_U), 'FLAG'] = 3

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

        heimann = DecadesVariable(combined)
        heimann.add_flag(self.d.FLAG)

        self.add_output(heimann)

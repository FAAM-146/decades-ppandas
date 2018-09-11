import numpy as np
import pandas as pd

from ..decades import DecadesVariable
from .base import PPBase


class RioFastThermistor(PPBase):

    inputs = ['CORCON_fast_temp', 'CORCON_fasttemp_hi_lo', 'PS_RVSM',
                'Q_RVSM']  # , 'TRFCTR']

    def declare_outputs(self):
        self.declare(
            'R_FT',
            units='ohm',
            frequency=32,
            long_name='Resistance of the indicated air temperature sensor',
            standard_name=None
        )

        self.declare(
            'IAT_FT',
            units='degK',
            frequency=32,
            long_name='Indicated air temperature from the fast temperature sensor',
            standard_name='indicated_air_temperature'
        )

        self.declare(
            'TAT_FT',
            units='degK',
            frequency=32,
            long_name='True air temperature from the fast temperature sensor',
            standard_name='true_air_temperature'
        )

    def process(self):
        self.get_dataframe()

        # calibration constants
        A, B, C, D, E, F, G, H = (1.1692, -29.609, 361.04, -0.030927, 0.23438,
                                  -2.4769E-7, 0.59181, -9.7277E-7)

        TRFCTR = 9.9280E-1

        df = self.d

        df['R'] = (1.0 / (D + E * np.exp(df['CORCON_fast_temp'] * F)
                          + G * np.exp(H * df['CORCON_fast_temp'])))

        df['IAT'] = A * np.log(df['R'])**2 + B * np.log(df['R']) + C

        df['MACHNO'] = np.sqrt(
            5 * ((1 + df['Q_RVSM'] / df['PS_RVSM'])**(2 / 7) - 1)
        )

        df['TAT'] = df['IAT'] / (1 + (0.2 * df['MACHNO']**2 * TRFCTR))

        df['FLAG'] = 3

        r_ft = pd.DataFrame({
            'R_FT': df['R'],
            'R_FT_FLAG': df['FLAG']
        })

        iat_ft = pd.DataFrame({
            'IAT_FT': df['IAT'],
            'IAT_FT_FLAG': df['FLAG']
        })

        tat_ft = pd.DataFrame({
            'TAT_FT': df['TAT'],
            'TAT_FT_FLAG': df['FLAG']
        })

        self.add_output(DecadesVariable(r_ft, name='R_FT'))
        self.add_output(DecadesVariable(tat_ft, name='TAT_FT'))
        self.add_output(DecadesVariable(iat_ft, name='IAT_FT'))


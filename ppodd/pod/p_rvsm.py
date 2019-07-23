import pandas as pd
import numpy as np

from ..decades import DecadesVariable
from .base import PPBase


class RioRvsm(PPBase):

    inputs = ['PRTAFT_pressure_alt', 'PRTAFT_ind_air_speed']

    def declare_outputs(self):
        self.declare(
            'PS_RVSM',
            units='hPa',
            frequency=32,
            number=576,
            standard_name='air_pressure',
            long_name=('Static pressure from the aircraft RVSM (air data) '
                       'system')
        )

        self.declare(
            'Q_RVSM',
            units='hPa',
            frequency=32,
            number=577,
            standard_name=None,
            long_name=('Pitot static pressure inverted from RVSM (air data) '
                       'system indicated airspeed')
        )

        self.declare(
            'PALT_RVS',
            units='m',
            frequency=32,
            number=578,
            standard_name='barometric_altitude',
            long_name=('Pressure altitude from the aircraft RVSM (air data) '
                       'system')
        )

    def calc_altitude(self):
        d = self.d

        d['PALT_FEET'] = d['PRTAFT_pressure_alt'] * 4
        d['FLAG_ALT'] = 0

        d.loc[d['PALT_FEET'] < -2000, 'FLAG_ALT'] = 1
        d.loc[d['PALT_FEET'] > 50000, 'FLAG_ALT'] = 1

        d['PALT_METRES'] = d['PALT_FEET'] / 3.28084

    def calc_pressure(self):
        d = self.d

        high = self.d['PALT_METRES'] > 11000

        T0 = 288.15
        L0 = -0.0065
        h0 = 0.0
        go = 9.80665
        M = 0.0289644
        R = 8.31432
        P0 = 1013.2500

        # Calulate pressure from standard atmosphere
        d['P'] = P0 * (
            T0 / (T0 + L0 * (d['PALT_METRES'] - h0))
        )**(go * M / (R * L0))

        T1 = 216.65
        P1 = 226.3210
        h1 = 11000.0

        d.loc[high, 'P'] = P1 * np.exp(
            (-go * M * (d['PALT_METRES'].loc[high] - h1)) / (R * T1)
        )

    def calc_ias(self):
        d = self.d
        d['IAS'] = d['PRTAFT_ind_air_speed'] * 0.514444 / 32.0
        d['IAS_FLAG'] = 0
        d.loc[d['IAS'] < -50, 'IAS_FLAG'] = 1
        d.loc[d['IAS'] > 500, 'IAS_FLAG'] = 1
        d.loc[d['FLAG_ALT'] == 1, 'IAS_FLAG'] = 1

    def calc_mach(self):
        d = self.d
        d['MACH'] = d['IAS'] / (340.294 * np.sqrt(d['P'] / 1013.25))

    def calc_pitot(self):
        d = self.d
        d['PITOT'] = d['P'] * ((((d['MACH']**2) / 5 + 1)**3.5) - 1)

    def calc_ps_rvsm(self):
        d = self.d

        ps_rvsm = pd.DataFrame([], index=d.index)
        ps_rvsm['PS_RVSM'] = d['P']
        ps_rvsm['PS_RVSM_FLAG'] = d['FLAG_ALT'].astype(np.int8)
        ps_rvsm = ps_rvsm.asfreq(self.freq[32]).interpolate()

        return ps_rvsm

    def calc_palt_rvs(self):
        d = self.d

        palt_rvs = pd.DataFrame([], index=d.index)
        palt_rvs['PALT_RVS'] = d['PALT_METRES']
        palt_rvs['PALT_RVS_FLAG'] = d['FLAG_ALT'].astype(np.int8)
        palt_rvs = palt_rvs.asfreq(self.freq[32]).interpolate()

        return palt_rvs

    def calc_q_rvsm(self):
        d = self.d
        q_rvsm = pd.DataFrame([], index=d.index)
        q_rvsm['Q_RVSM'] = d['PITOT']
        q_rvsm['Q_RVSM_FLAG'] = d['IAS_FLAG']
        q_rvsm = q_rvsm.asfreq(self.freq[32]).interpolate()

        return q_rvsm

    def process(self):

        self.get_dataframe()

        self.calc_altitude()
        self.calc_pressure()
        self.calc_ias()
        self.calc_mach()
        self.calc_pitot()

        self.add_output(
            DecadesVariable(self.calc_ps_rvsm(), name='PS_RVSM')
        )

        self.add_output(
            DecadesVariable(self.calc_palt_rvs(), name='PALT_RVS')
        )

        self.add_output(
            DecadesVariable(self.calc_q_rvsm(), name='Q_RVSM')
        )
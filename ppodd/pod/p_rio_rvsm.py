import pandas as pd
import numpy as np

from ..decades import DecadesVariable
from .base import PPBase


class RioRvsm(PPBase):

    def inputs(self):
        return ['PRTAFT_PRESSURE_ALT', 'PRTAFT_IND_AIR_SPEED']

    def declare_outputs(self):
        self.declare(
            'PS_RVSM',
            units='hPa',
            frequency=32,
            number=576,
            long_name='Static pressure from the aircraft RVSM (air data) system',
            standard_name='air_pressure'
        )

        self.declare(
            'Q_RVSM',
            units='hPa',
            frequency=32,
            number=577,
            long_name='Pitot static pressure inverted from RVSM (air data) system indicated airspeed',
            standard_name=None
        )

        self.declare(
            'PALT_RVS',
            units='m',
            frequency=32,
            number=578,
            long_name='Pressure altitude from the aircraft RVSM (air data) system',
            standard_name='barometric_altitude'
        )

    def process(self):

        wf = self.get_dataframe()

        wf['PALT_FEET'] = wf['PRTAFT_PRESSURE_ALT'] * 4
        wf['FLAG_ALT'] = 0
        wf.loc[wf['PALT_FEET'] < -2000, 'FLAG_ALT'] = 1
        wf.loc[wf['PALT_FEET'] > 50000, 'FLAG_ALT'] = 1
        wf['PALT_METERS'] = wf['PALT_FEET'] / 3.28084

        high = wf['PALT_METERS'] > 11000

        T0 = 288.15
        L0 = -0.0065
        h0 = 0.0
        go = 9.80665
        M = 0.0289644
        R = 8.31432
        P0 = 1013.2500

        wf['P'] = P0 * (T0 / (T0 + L0 * (wf['PALT_METERS'] - h0)))**(go * M / (R * L0))

        T1 = 216.65
        P1 = 226.3210
        h1 = 11000.0

        wf.loc[high, 'P'] = P1 * np.exp((-go * M * (wf['PALT_METERS'].loc[high] - h1)) / (R * T1))

        ps_rvsm = pd.DataFrame([], index=wf.index)
        ps_rvsm['PS_RVSM'] = wf['P']
        ps_rvsm['PS_RVSM_FLAG'] = wf['FLAG_ALT'].astype(np.int8)
        ps_rvsm = ps_rvsm.asfreq('31250000N').interpolate()
        self.add_output(DecadesVariable(ps_rvsm, name='PS_RVSM'))

        palt_rvs = pd.DataFrame([], index=wf.index)
        palt_rvs['PALT_RVS'] = wf['PALT_METERS']
        palt_rvs['PALT_RVS_FLAG'] = wf['FLAG_ALT'].astype(np.int8)
        palt_rvs = palt_rvs.asfreq('31250000N').interpolate()
        self.add_output(DecadesVariable(palt_rvs, name='PALT_RVS'))

        wf['IAS'] = wf['PRTAFT_IND_AIR_SPEED'] * 0.514444 / 32.0
        wf['IAS_FLAG'] = 0
        wf.loc[wf['IAS'] < -50, 'IAS_FLAG'] = 1
        wf.loc[wf['IAS'] > 500, 'IAS_FLAG'] = 1
        wf.loc[wf['FLAG_ALT'] == 1, 'IAS_FLAG'] = 1
        wf['MACH'] = wf['IAS'] / (340.294 * np.sqrt(wf['P'] / 1013.25))
        wf['PITOT'] = wf['P'] * ((((wf['MACH']**2) / 5 + 1)**3.5) - 1)

        q_rvsm = pd.DataFrame([], index=wf.index)
        q_rvsm['Q_RVSM'] = wf['PITOT']
        q_rvsm['Q_RVSM_FLAG'] = wf['IAS_FLAG']
        q_rvsm = q_rvsm.asfreq('31250000N').interpolate()
        self.add_output(DecadesVariable(q_rvsm, name='Q_RVSM'))

        self.finalize()

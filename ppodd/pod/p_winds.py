import numpy as np
import pandas as pd

from ..decades import DecadesVariable
from .base import PPBase
from .shortcuts import _o, _z

from .c_winds import c_winds

class TurbWinds(PPBase):

    inputs = [
        'INSPOSN',
        'TAS',
        'VELN_GIN',
        'VELE_GIN',
        'VELD_GIN',
        'ROLL_GIN',
        'PTCH_GIN',
        'HDG_GIN',
        'ROLR_GIN',
        'PITR_GIN',
        'HDGR_GIN',
        'AOA',
        'AOSS'
    ]

    @staticmethod
    def test():
        return {
            'INSPOSN': ('const', [16, -.8, -.4]),
            'TAS': ('data', 120 * _o(100)),
            'VELN_GIN': ('data', 100 * _o(100)),
            'VELE_GIN': ('data', 100 * _o(100)),
            'VELD_GIN': ('data', _z(100)),
            'ROLL_GIN': ('data', _z(100)),
            'PTCH_GIN': ('data', 6 * _o(100)),
            'HDG_GIN': ('data', _z(100)),
            'ROLR_GIN': ('data', _z(100)),
            'PITR_GIN': ('data', _z(100)),
            'HDGR_GIN': ('data', _z(100)),
            'AOA': ('data', 6 * _o(100)),
            'AOSS': ('data', _z(100))
        }

    def declare_outputs(self):
        self.declare(
            'V_C',
            units='m s-1',
            frequency=32,
            long_name='Northward wind component from turbulence probe and GIN',
            standard_name='northward_wind'
        )

        self.declare(
             'U_C',
             units='m s-1',
             frequency=32,
             long_name='Eastward wind component from turbulence probe and GIN',
             standard_name='eastward_wind'
         )

        self.declare(
            'W_C',
            units='m s-1',
            frequency=32,
            long_name='Vertical wind component from turbulence probe and GIN',
            standard_name='upward_air_velocity'
        )

    def process(self):
        self.get_dataframe()
        d = self.d
        ginpos = self.dataset['INSPOSN']

        u, v, w = c_winds(
            d.TAS.values.astype('float64'),
            d.AOA.values.astype('float64'),
            d.AOSS.values.astype('float64'),
            d.VELN_GIN.values.astype('float64'),
            d.VELE_GIN.values.astype('float64'),
            d.VELD_GIN.values.astype('float64'),
            d.HDG_GIN.values.astype('float64'),
            d.PTCH_GIN.values.astype('float64'),
            d.ROLL_GIN.values.astype('float64'),
            ginpos[0],
            ginpos[1],
            ginpos[2],
            d.HDGR_GIN.values.astype('float64'),
            d.PITR_GIN.values.astype('float64'),
            d.ROLR_GIN.values.astype('float64')
        )

        d['U_C'] = u
        d['V_C'] = v
        d['W_C'] = w

        u = DecadesVariable(d['U_C'])
        v = DecadesVariable(d['V_C'])
        w = DecadesVariable(d['W_C'])

        for var in (u, v, w):
            self.add_output(var)

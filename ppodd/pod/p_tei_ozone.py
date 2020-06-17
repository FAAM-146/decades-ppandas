import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from .base import PPBase
from .shortcuts import _c, _o, _z

FLOW_THRESHOLD = 0.5
CONC_THRESHOLD = -10
FLAG_AFTER_TO = 20


class TeiOzone(PPBase):
    r"""
    Provides ozone concentration from the TE49C O$_3$ analyser. Ozone data are
    taken as-is from the instrument; this module provides flagging information
    based on threshold values and instrument status flags.
    """

    inputs = [
        'TEIOZO_conc',
        'TEIOZO_flag',
        'TEIOZO_FlowA',
        'TEIOZO_FlowB',
        'WOW_IND'
    ]

    @staticmethod
    def test():
        return {
            'TEIOZO_conc': ('data', 50 * _o(100)),
            'TEIOZO_flag': ('data', [b'0C100000'] * 100),
            'TEIOZO_FlowA': ('data', .7 * _o(100)),
            'TEIOZO_FlowB': ('data', .7 * _o(100)),
            'WOW_IND': ('data', _c([_o(20), _z(60), _o(20)]))
        }

    def declare_outputs(self):

        self.declare(
            'O3_TECO',
            units='ppb',
            frequency=1,
            long_name=('Mole fraction of Ozone in air from the TECO 49 '
                       'instrument'),
            standard_name='mole_fraction_of_ozone_in_air'
        )

    def flag(self):
        d = self.d

        d['STATUS_FLAG'] = 0
        d['STATUS_FLAG1'] = 0
        d['STATUS_FLAG2'] = 0

        d['CONC_FLAG'] = 0
        d['FLOW_FLAG'] = 0
        d['WOW_FLAG'] = 0

        d['TEIOZO_flag'].fillna(value='', inplace=True)
        flag = np.array([i.lower() for i in d['TEIOZO_flag']])
        flag[flag == ''] = 3

        if '1c100000' in flag:
            d.STATUS_FLAG1 = flag != '1c100000'

        if '0c100000' in flag:
            d.STATUS_FLAG2 = flag != '0c100000'

        d.STATUS_FLAG = d.STATUS_FLAG1 | d.STATUS_FLAG2

        d.loc[d['TEIOZO_conc'] < CONC_THRESHOLD, 'CONC_FLAG'] = 1
        d.loc[d['TEIOZO_FlowA'] < FLOW_THRESHOLD, 'FLOW_FLAG'] = 1
        d.loc[d['TEIOZO_FlowB'] < FLOW_THRESHOLD, 'FLOW_FLAG'] = 1
        d.loc[d['WOW_IND'] != 0, 'WOW_FLAG'] = 1

    def process(self):
        self.get_dataframe()

        self.flag()

        dv = DecadesVariable(self.d['TEIOZO_conc'], name='O3_TECO',
                             flag=DecadesBitmaskFlag)

        dv.flag.add_mask(
            self.d['STATUS_FLAG'], 'instrument_alarm',
            'The status flag provided by the instrument is indicating an '
            'alarm state'
        )

        dv.flag.add_mask(
            self.d['CONC_FLAG'], 'conc_out_of_range',
            'Reported ozone concentration is below the valid minimum of '
            f'{CONC_THRESHOLD}'
        )

        dv.flag.add_mask(
            self.d['FLOW_FLAG'], 'flow_out_of_range',
            'At least one of the recorded flow rates is below the valid '
            f'minimum of {FLOW_THRESHOLD}'
        )

        dv.flag.add_mask(
            self.d['WOW_FLAG'], flags.WOW,
            'The aircraft is on the ground'
        )

        self.add_output(dv)

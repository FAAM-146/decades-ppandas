import numpy as np

from ..decades import DecadesVariable
from .base import PPBase

class TeiOzone(PPBase):

    inputs = [
        'TEIOZO_conc',
        'TEIOZO_flag',
        'TEIOZO_FlowA',
        'TEIOZO_FlowB',
        'WOW_IND'
    ]

    def declare_outputs(self):

        self.declare(
            'O3_TECO',
            units='ppb',
            frequency=1,
            number=574,
            long_name=('Mole fraction of Ozone in air from the TECO 49 '
                       'instrument'),
            standard_name='mole_fraction_of_ozone_in_air'
        )

    def flag(self):
        FLOW_THRESHOLD = 0.5
        CONC_THRESHOLD = -10
        FLAG_AFTER_TO = 20

        d = self.d

        d['TEIOZO_flag'].fillna(value='', inplace=True)
        flag =  np.array([i.lower() for i in d['TEIOZO_flag']])
        flag[flag == ''] = 3

        if '1c100000' in flag:
            flag = flag != '1c100000'

        if '0c100000' in flag:
            flag = flag != '0c100000'


        flag[d['TEIOZO_conc'] < CONC_THRESHOLD] = 2
        flag[d['TEIOZO_FlowA'] < FLOW_THRESHOLD] = 3
        flag[d['TEIOZO_FlowB'] < FLOW_THRESHOLD] = 3
        flag[d['WOW_IND'] != 0] = 3

        to_ix = int(np.where(d['WOW_IND'].diff() == -1)[0][0])
        flag[to_ix:to_ix + FLAG_AFTER_TO + 1] = 3

        flag = flag.astype('int8')

        d['FLAG'] = flag

    def process(self):
        self.get_dataframe()

        self.flag()

        dv = DecadesVariable(self.d['TEIOZO_conc'], name='O3_TECO')
        dv.add_flag(self.d['FLAG'])

        self.add_output(dv)

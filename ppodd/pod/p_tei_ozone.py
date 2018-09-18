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

        d = self.d

        flag =  np.array([i.lower() for i in d['TEIOZO_flag']])

        if '1c100000' in flag:
            flag = flag != '1c100000'

        if '0c100000' in flag:
            flag = flag!='0c100000'

        flag = flag.astype('int8')

        d['FLAG'] = flag

        d.loc[d['TEIOZO_conc'] < -10, 'FLAG'] = 2
        d.loc[d['TEIOZO_FlowA'] < FLOW_THRESHOLD, 'FLAG'] = 3
        d.loc[d['TEIOZO_FlowB'] < FLOW_THRESHOLD, 'FLAG'] = 3
        d.loc[d['WOW_IND'] != 0, 'FLAG'] = 3

        to_ix = int(np.where(d['WOW_IND'][:-1] - d['WOW_IND'][1:] == 1)[0])
        d.iloc[to_ix:to_ix + 20, 'FLAG'] = 3


    def process(self):
        self.get_dataframe()

        self.flag()

        dv = DecadesVariable(self.d['TEIOZO_conc'], name='O3_TECO')
        dv.add_flag(self.d['FLAG'])

        self.add_output(dv)

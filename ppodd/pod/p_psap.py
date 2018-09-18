import numpy as np

from ..decades import DecadesVariable
from .base import PPBase

class PSAP(PPBase):

    inputs = [
        'AERACK_psap_flow',         #  PSAP flow (dlu)
        'AERACK_psap_lin',          #  PSAP lin (dlu)
        'AERACK_psap_log',          #  PSAP log (dlu)
        'AERACK_psap_transmission'  #  PSAP transmission (dlu)
    ]

    def declare_outputs(self):
        self.declare(
            'PSAP_LIN',
            units='m-1',
            frequency=1,
            number=648,
            long_name=('Uncorrected absorbtion coefficient at 565nm, linear, '
                       'from PSAP')
        )

        self.declare(
            'PSAP_LOG',
            units='1',
            frequency=1,
            number=649,
            long_name=('Uncorrected absorption coefficient at 565nm, log, '
                       'from PSAP')
        )

        self.declare(
            'PSAP_FLO',
            units='standard l min-1',
            frequency=1,
            long_name='PSAP Flow'
        )

        self.declare(
            'PSAP_TRA',
            units='percent',
            frequency=1,
            long_name='PSAP Transmittance'
        )

    def flag(self):
        d = self.d
        d['TRA_FLAG'] = 0
        d['FLO_FLAG'] = 0
        d['COM_FLAG'] = 0
        d.loc[d['PSAP_TRA'] < 0.5, 'TRA_FLAG'] = 1
        d.loc[d['PSAP_TRA'] > 1.05, 'TRA_FLAG'] = 1
        d.loc[d['PSAP_FLO'] < 1, 'FLO_FLAG'] = 2
        d['COM_FLAG'] = d['FLO_FLAG'] + d['TRA_FLAG']

        # TODO: Original flagging adds a 2 second buffer when the PSAP pump is
        # toggled.
        pump_on = np.where(np.diff(d['FLO_FLAG']) == 1)
        pump_off = np.where(np.diff(d['FLO_FLAG']) == -1)

    def process(self):
        self.get_dataframe()
        d = self.d

        d['PSAP_FLO'] = d['AERACK_psap_flow'] * 0.5
        d['PSAP_LIN'] = d['AERACK_psap_lin'] * 0.5E-5
        d['PSAP_LOG'] = 10**((d['AERACK_psap_log'] /2) - 7)
        d['PSAP_TRA'] = d['AERACK_psap_transmission'] * 0.125

        self.flag()

        psap_flo = DecadesVariable(d['PSAP_FLO'])
        psap_lin = DecadesVariable(d['PSAP_LIN'])
        psap_log = DecadesVariable(d['PSAP_LOG'])
        psap_tra = DecadesVariable(d['PSAP_TRA'])

        for dv in (psap_flo, psap_lin, psap_log, psap_tra):
            dv.add_flag(d['TRA_FLAG'])
            dv.add_flag(d['FLO_FLAG'])
            dv.add_flag(d['COM_FLAG'])
            self.add_output(dv)


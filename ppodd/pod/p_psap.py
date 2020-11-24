import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from .base import PPBase
from .shortcuts import _o

TRANS_VALID_MIN = 0.5
TRANS_VALID_MAX = 1.05
FLOW_VALID_MIN = 1

class PSAP(PPBase):
    r"""
    Reports data from the Radiance Research Particle Soot Absorbtion
    Photometer.  The following transformations are applied to the data
    from the aerosol rack DLU:

    .. math::
        F = \frac{F_\text{DLU}}{2},

    .. math::
        P_\text{lin} = \frac{P_{\text{lin}_\text{DLU}}}{2\times10^{5}},

    .. math::
        P_\text{log} = 10^{(P_{\text{log}_\text{DLU}} / 2) - 7},

    .. math::
        T = \frac{T_\text{DLU}}{8},

    where :math:`P_\text{lin}`, :math:`P_\text{log}`, :math:`F`, and :math:`T`
    correspond to the outputs ``PSAP_LIN``, ``PSAP_LOG``, ``PSAP_FLO``, and
    ``PSAP_TRA`` respectively.

    Flagging is based on the flow rate and transmittance ratio limits.
    """

    inputs = [
        'AERACK_psap_flow',         #  PSAP flow (dlu)
        'AERACK_psap_lin',          #  PSAP lin (dlu)
        'AERACK_psap_log',          #  PSAP log (dlu)
        'AERACK_psap_transmission'  #  PSAP transmission (dlu)
    ]

    @staticmethod
    def test():
        return {
            'AERACK_psap_flow': ('data', 2 * _o(100)),
            'AERACK_psap_lin': ('data', 1e5 * _o(100)),
            'AERACK_psap_log': ('data', _o(100)),
            'AERACK_psap_transmission': ('data', _o(100))
        }

    def declare_outputs(self):
        self.declare(
            'PSAP_LIN',
            units='m-1',
            frequency=1,
            long_name=('Uncorrected absorbtion coefficient at 565nm, linear, '
                       'from PSAP')
        )

        self.declare(
            'PSAP_LOG',
            units='1',
            frequency=1,
            long_name=('Uncorrected absorption coefficient at 565nm, log, '
                       'from PSAP')
        )

        self.declare(
            'PSAP_FLO',
            units='l min-1',
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

        psap_flo = DecadesVariable(d['PSAP_FLO'], flag=DecadesBitmaskFlag)
        psap_lin = DecadesVariable(d['PSAP_LIN'], flag=DecadesBitmaskFlag)
        psap_log = DecadesVariable(d['PSAP_LOG'], flag=DecadesBitmaskFlag)
        psap_tra = DecadesVariable(d['PSAP_TRA'], flag=DecadesBitmaskFlag)

        for dv in (psap_flo, psap_lin, psap_log, psap_tra):
            dv.flag.add_mask(
                d['TRA_FLAG'], 'Transmittance out of range',
                ('Transmittance ratio is outside the valid range '
                 f'[{TRANS_VALID_MIN}, {TRANS_VALID_MAX}')
            )
            dv.flag.add_mask(
                d['FLO_FLAG'], 'Flow out of range',
                ('PSAP flow is out of range. This most likely indicates that '
                 'the PSAP interrupt is active, to prevent water ingress')
            )
            self.add_output(dv)


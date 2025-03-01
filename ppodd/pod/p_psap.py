"""
This module provides a postprocessing module for the PSAP instrument on the
aerosol rack. See class docstring for more information.
"""
# pylint: disable=invalid-name
import numpy as np
import pandas as pd

from ..decades import DecadesVariable, DecadesBitmaskFlag
from .base import PPBase, register_pp
from .shortcuts import _o

TRANS_VALID_MIN = 0.5
TRANS_VALID_MAX = 1.05
FLOW_VALID_MIN = 1


@register_pp('core')
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
        """
        Return dummy input data for testing.
        """
        return {
            'AERACK_psap_flow': ('data', 2 * _o(100), 1),
            'AERACK_psap_lin': ('data', 1e5 * _o(100), 1),
            'AERACK_psap_log': ('data', _o(100), 1),
            'AERACK_psap_transmission': ('data', _o(100), 1)
        }

    def declare_outputs(self):
        """
        Declare module outputs.
        """
        self.declare(
            'PSAP_LIN',
            units='m-1',
            frequency=1,
            long_name=('Uncorrected absorbtion coefficient at 565nm, linear, '
                       'from PSAP'),
            instrument_manufacturer='Radiance Research'
        )

        self.declare(
            'PSAP_LOG',
            units='1',
            frequency=1,
            long_name=('Uncorrected absorption coefficient at 565nm, log, '
                       'from PSAP'),
            instrument_manufacturer='Radiance Research'
        )

        self.declare(
            'PSAP_FLO',
            units='l min-1',
            frequency=1,
            long_name='PSAP Flow',
            instrument_manufacturer='Radiance Research'
        )

        self.declare(
            'PSAP_TRA',
            units='percent',
            frequency=1,
            long_name='PSAP Transmittance',
            instrument_manufacturer='Radiance Research'
        )

    def flag(self):
        """
        Provide flagging info.
        """
        d = self.d
        d['TRA_FLAG'] = 0
        d['FLO_FLAG'] = 0
        d['COM_FLAG'] = 0
        d.loc[d['PSAP_TRA'] < 0.5, 'TRA_FLAG'] = 1
        d.loc[d['PSAP_TRA'] > 1.05, 'TRA_FLAG'] = 1
        d.loc[d['PSAP_FLO'] < 1, 'FLO_FLAG'] = 2
        

        # Adds a 2 second buffer to the flag when the PSAP pump is
        # toggled.
        pump_off = d['FLO_FLAG'].loc[d['FLO_FLAG'].diff() == 2]
        pump_on = d['FLO_FLAG'].loc[d['FLO_FLAG'].diff() == -2]

        sec = pd.Timedelta('1s')
        for index in pump_on.index:
            d.loc[[index, index + sec, index + (2 * sec)], 'FLO_FLAG'] = 2
        for index in pump_off.index:
            d.loc[[index-(2 * sec), index - sec, index], 'FLO_FLAG'] = 2

        d['COM_FLAG'] = d['FLO_FLAG'] + d['TRA_FLAG']

    def process(self):
        """
        Processing entry hook.
        """
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

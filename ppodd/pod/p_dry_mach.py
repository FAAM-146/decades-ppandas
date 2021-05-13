"""
Provides a processing module which calculates a mach number for dry
air.
"""
# pylint: disable=invalid-name
import numpy as np
import pandas as pd

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..utils.constants import MOL_MASS_H20, MOL_MASS_DRY_AIR, c_pd, c_vd
from ..utils.calcs import sp_mach
from .base import PPBase, register_pp
from .shortcuts import _o


@register_pp('core')
class DryMach(PPBase):
    r"""
    This module calculate a drt-air Mach number using RVSM pressure
    measurements.

    Dry-air Mach number, :math:`M`, is given by

    .. math::

        M =
        \sqrt{\left(\frac{2c_v}{R_a}\right)\left(\left(\frac{p+q}{q}\right)^\frac{R_a}{c_p}
        - 1\right)},


    The module also provides the parameter ``SH_GAMMA``, which is the ratio of
    specific heats at constant pressure and constant volume, along with
    uncertainty estimates for ``MACH`` and ``SH_GAMMA``. The former is an
    uncertainty reported in BAe Reoprt 126, while the latter is a constant and
    assumed zero.
    """

    inputs = [
        'PS_RVSM',
        'Q_RVSM',
        'WOW_IND',
        'MACH_UNC_BAE'
    ]

    @staticmethod
    def test():
        """
        Provide dummy input data for testing.
        """
        n = 100
        return {
            'PS_RVSM': ('data', 850 * _o(n), 32),
            'Q_RVSM': ('data', 70 * _o(n), 32),
            'WOW_IND': ('data', 0 * _o(n), 1),
            'MACH_UNC_BAE': ('const', 0.005),
        }

    def declare_outputs(self):
        """
        Declare outputs created by this module.
        """
        self.declare(
            'MACH',
            units=1,
            frequency=32,
            long_name='Dry air Mach derived from and RVSM',
            write=False
        )

        self.declare(
            'MACH_CU',
            units=1,
            frequency=32,
            long_name='Uncertainty estimate for MACH',
            write=False
        )

        self.declare(
            'SH_GAMMA',
            units=1,
            frequency=32,
            long_name=('Ratio of specific heats at constant pressure and '
                       'constant pressure'),
            write=False
        )

        self.declare(
            'SH_GAMMA_CU',
            units=1,
            frequency=32,
            long_name='Uncertainty estimate for SH_GAMMA',
            write=False
        )

    def process(self):
        """
        Processing entry hook.
        """
        self.get_dataframe()
        d = self.d

        q = d.Q_RVSM
        p = d.PS_RVSM
        wow = d.WOW_IND.fillna(method='bfill').fillna(method='ffill')

        mach = sp_mach(q, p)
        gamma = pd.Series(
            (c_pd / c_vd) * np.ones_like(mach),
            index=mach.index
        )

        mach_var = DecadesVariable(
            mach, name='MACH', flag=DecadesBitmaskFlag
        )

        # Simple flag for aircraft on the ground
        mach_var.flag.add_mask(
            wow==1, 'aircraft on ground',
            ('The aircraft is on the ground, as indicated by the '
             'weight-on-wheels indicator')
        )
        self.add_output(mach_var)

        # Create gamma output
        gamma_var = DecadesVariable(
            gamma, name='SH_GAMMA', flag=DecadesBitmaskFlag
        )
        self.add_output(gamma_var)


        # Uncertainties
        u_gamma = np.zeros_like(gamma)
        u_mach = self.dataset['MACH_UNC_BAE']

        # Create gamma uncertainty output
        u_gamma_out = DecadesVariable(
            pd.Series(u_gamma, index=gamma.index),
            name='SH_GAMMA_CU', flag=DecadesBitmaskFlag
        )
        self.add_output(u_gamma_out)

        # Create mach uncertainty output
        u_mach_out = DecadesVariable(
            pd.Series(u_mach, index=mach.index),
            name='MACH_CU', flag=DecadesBitmaskFlag
        )
        self.add_output(u_mach_out)
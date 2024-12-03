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
    This module calculate a dry-air Mach number using RVSM pressure
    measurements.

    Dry-air Mach number, :math:`M`, is given by

    .. math::

        M =
        \sqrt{\left(\frac{2c_v}{R_a}\right)\left(\left(\frac{p+q}{q}\right)^\frac{R_a}{c_p}
        - 1\right)},


    The module also provides the parameter ``SH_GAMMA``, which is the ratio of
    specific heats at constant pressure and constant volume, along with
    uncertainty estimates for ``MACH`` and ``SH_GAMMA``. The former is the
    combined uncertainty of the uncertainy in BAe report 127 and the
    uncertainty in gamma, while the latter is a constant and assumed to be
    0.003, which is derived from the impact of neglecting humidity at the
    highest expected VMR of 35000 at 1100 hPa.
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
            units='1',
            frequency=32,
            long_name='Dry air Mach derived from and RVSM',
            write=False
        )

        self.declare(
            'MACH_CU',
            units='1',
            frequency=32,
            long_name='Uncertainty estimate for MACH',
            write=False
        )

        self.declare(
            'SH_GAMMA',
            units='1',
            frequency=32,
            long_name=('Ratio of specific heats at constant pressure and '
                       'constant pressure'),
            write=False
        )

        self.declare(
            'SH_GAMMA_CU',
            units='1',
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
        wow = d.WOW_IND.bfill().ffill()

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

        # The calculated gamma for vmr=35000 at 1100 mb is 1.397. This is the
        # maximum realistic change to gamma that we'd miss out on by not doing
        # the humidity correction. Not ideal to do this because it's a
        # systematic error not a randomly distributed unceratinty
        u_gamma = 0.003 * np.ones_like(gamma)

        # This is the uncertainty in Mach when no humidity correction is
        # applied. It's a combination of the BAE uncertainty in Mach given in
        # report 126 and the uncertainty from not taking humidity into account. 
        # RSS of u_mach_bae + uncertainty from not doing a humidity correction
        # so 0.005 and 0.0003 (that's the differece the humidity correction
        # makes to the Mach number for VMR=35000 at 1100mb (also tested other
        # conditions seen in a flight over India but this was biggest difference
        # seen) Not ideal to do this because it's a systematic error not a
        # randomly distributed unceratinty)
        u_mach = np.sqrt(
            (self.dataset['MACH_UNC_BAE'] * np.ones_like(mach))**2
            + u_gamma**2
        )

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

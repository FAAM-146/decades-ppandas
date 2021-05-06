"""
Provides a processing module which calculates a mach number for moist
air.
"""
# pylint: disable=invalid-name
import numpy as np
import pandas as pd

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..utils.constants import MOL_MASS_H20, MOL_MASS_DRY_AIR, c_pd, c_vd
from .base import PPBase, register_pp
from .shortcuts import _o


@register_pp('core')
class MoistMach(PPBase):
    r"""
    This module calculate a moist-air Mach number using RVSM pressure
    measurements and the volume mixing ratio from the flush mounted WVSSII
    (assumed to identify itself as ``WVSS2A``).

    Moist-air Mach number, :math:`M`, is given by

    .. math::

        M =
        \sqrt{\left(\frac{2c_v}{R_a}\right)\left(\left(\frac{p+q}{q}\right)^\frac{R_a}{c_p}
        - 1\right)},

    where

    .. math::

        R_a = c_p - c_v

    and

    .. math::

        c_p &= c_{pd} \left(1 + q_h\left(\frac{8}{7\epsilon} - 1\right)\right)\\
        c_v &= c_{vd} \left(1 + q_h\left(\frac{6}{5\epsilon} - 1\right)\right)\\

    Here :math:`c_{pd}` and :math:`c_{vd}` are specific heats for dry air at
    constant pressure and volume, respectively, :math:`\epsilon` is the ratio
    of the molecular mass of water to that of dry air, and the specific
    humidity, :math:`q_h`, is given by

    .. math::

        q_h = \frac{\epsilon V}{\epsilon V + 1},

    where :math:`V = 1000000\times\text{vmr}_\text{WVSS2F}`.

    The module also provides the parameter ``SH_GAMMA``, which is the ratio of
    specific heats at constant pressure and constant volume, along with
    uncertainty estimates for ``MOIST_MACH`` and ``SH_GAMMA``. The former is a
    combination from uncertainties in the corrected WVSS2 VMR and the
    uncertainty reported in BAe Reoprt 126, while the latter arises from the
    uncertainty in the corrected WVSS2 VMR.
    """

    inputs = [
        'WVSS2F_VMR_C',
        'PS_RVSM',
        'Q_RVSM',
        'WOW_IND',
        'SH_UNC_GAMMA',
        'MACH_UNC_BAE',
        'MACH_UNC_HUMIDITY'
    ]

    @staticmethod
    def test():
        """
        Provide dummy input data for testing.
        """
        n = 100
        return {
            'WVSS2F_VMR_C': ('data', 100 * _o(n), 1),
            'PS_RVSM': ('data', 850 * _o(n), 32),
            'Q_RVSM': ('data', 70 * _o(n), 32),
            'WOW_IND': ('data', 0 * _o(n), 1),
            'SH_UNC_GAMMA': ('const', [1E-5, 0, 0]),
            'MACH_UNC_BAE': ('const', 0.005),
            'MACH_UNC_HUMIDITY': ('const', [1E-5, 0, 0])
        }

    def declare_outputs(self):
        """
        Declare outputs created by this module.
        """
        self.declare(
            'MOIST_MACH',
            units=1,
            frequency=32,
            long_name='Moist air Mach derived from WVSS-II(F) and RVSM',
            write=False
        )

        self.declare(
            'MOIST_MACH_CU',
            units=1,
            frequency=32,
            long_name='Uncertainty estimate for moist-air Mach',
            write=True
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
        wvss2_vmr = d.WVSS2F_VMR_C.interpolate(limit=32)
        wow = d.WOW_IND.fillna(method='bfill')

        # epsilon is the mass ratio of water and dry air
        eps = MOL_MASS_H20 / MOL_MASS_DRY_AIR

        # Convert wvss-ii from ppmv to a ratio
        vmr_ratio = wvss2_vmr * 1e-6

        # Calculate specific humidity from the vmr ratio
        qh = eps * vmr_ratio / (eps * vmr_ratio + 1)

        # Specific heats at constant pressure and volume
        c_p = c_pd * (1 + qh * ((8 / (7 * eps)) - 1))
        c_v = c_vd * (1 + qh * ((6 / (5 * eps)) - 1))

        R_a = c_p - c_v
        gamma = c_p / c_v

        # Moist mach number
        mach = np.sqrt(
            (2 * c_v / R_a) * (((p + q) / p) ** (R_a / c_p) - 1)
        )

        mach_var = DecadesVariable(
            mach, name='MOIST_MACH', flag=DecadesBitmaskFlag
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
        u_gamma = np.polyval(self.dataset['SH_UNC_GAMMA'][::-1], wvss2_vmr)
        u_mach_bae = self.dataset['MACH_UNC_BAE']
        u_mach_humidity = mach * np.polyval(
            self.dataset['MACH_UNC_HUMIDITY'], wvss2_vmr
        )
        u_mach = np.sqrt(u_mach_bae**2. + u_mach_humidity**2.)

        # Create gamma uncertainty output
        u_gamma_out = DecadesVariable(
            pd.Series(u_gamma, index=wvss2_vmr.index),
            name='SH_GAMMA_CU', flag=DecadesBitmaskFlag
        )
        self.add_output(u_gamma_out)

        # Create mach uncertainty output
        u_mach_out = DecadesVariable(
            pd.Series(u_mach, index=wvss2_vmr.index),
            name='MOIST_MACH_CU', flag=DecadesBitmaskFlag
        )
        self.add_output(u_mach_out)

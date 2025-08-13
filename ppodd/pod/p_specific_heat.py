import numpy as np
import pandas as pd

from ..decades import DecadesVariable, DecadesBitmaskFlag
from .base import PPBase, register_pp
from .shortcuts import _l
from ..utils.constants import MOL_MASS_H20, MOL_MASS_DRY_AIR, c_pd, c_vd


@register_pp("core")
class SpecificHeat(PPBase):
    r"""
    Calculate the ratio of specific heats at constant pressure and constant
    volume, :math:`\gamma`, using the moist-air Mach number, :math:`M`, and
    the volume mixing ratio from the WVSSII.

    The ratio of specific heats is given by

    .. math::
        \gamma = \frac{c_p}{c_v} = 1 + \frac{q_h}{\epsilon} \left(\frac{8}{7} - 1\right),

    where :math:`q_h` is the specific humidity and :math:`\epsilon` is the
    ratio of the molecular mass of water to that of dry air.
    """

    inputs = [
        "WVSS2F_VMR_C",  # Volume mixing ratio from WVSSII (corrected)
        "SH_UNC_GAMMA",  # Uncertainty in the ratio of specific heats at constant pressure and constant volume, corrected for humidity
    ]

    @staticmethod
    def test():
        """
        Return some dummy input data for testing usage.
        """
        return {
            "WVSS2F_VMR_C": ("data", _l(0.001, 0.005, 100), 32),
            "SH_UNC_GAMMA": ("data", _l(0.01, 0.02, 100), 32),
            "MACH_UNC_BAE": ("data", _l(0.001, 0.002, 100), 32),
            "MACH_UNC_HUMIDITY": ("data", _l(0.001, 0.002, 100), 32),
        }

    def declare_outputs(self) -> None:
        """
        Declare all of the output variables produced by this module, through
        calls to self.declare.
        """
        self.declare(
            "SH_GAMMA",
            units="1",
            frequency=1,
            long_name="Ratio of specific heats at constant pressure and constant volume",
            write=False,
        )

        self.declare(
            "SH_GAMMA_CU",
            units="1",
            frequency=1,
            long_name="Uncertainty in the ratio of specific heats at constant pressure and constant volume, corrected for humidity",
            write=False,
        )

        self.declare(
            "C_P",
            units="J kg-1 K-1",
            frequency=1,
            long_name="Specific heat at constant pressure, corrected for humidity",
            write=False,
        )

        self.declare(
            "C_V",
            units="J kg-1 K-1",
            frequency=1,
            long_name="Specific heat at constant volume, corrected for humidity",
            write=False,
        )

    def process(self):
        wvss2_vmr = self.d.WVSS2F_VMR_C

        # epsilon is the mass ratio of water and dry air
        eps = MOL_MASS_H20 / MOL_MASS_DRY_AIR

        # Convert wvss-ii from ppmv to a ratio
        vmr_ratio = wvss2_vmr * 1e-6

        # Calculate specific humidity from the vmr ratio
        qh = eps * vmr_ratio / (eps * vmr_ratio + 1)

        # Specific heats at constant pressure and volume
        c_p = c_pd * (1 + qh * ((8 / (7 * eps)) - 1))
        c_v = c_vd * (1 + qh * ((6 / (5 * eps)) - 1))

        gamma = c_p / c_v

        # Create gamma output
        gamma_var = DecadesVariable(gamma, name="SH_GAMMA", flag=DecadesBitmaskFlag)
        self.add_output(gamma_var)

        u_gamma = np.polyval(self.dataset["SH_UNC_GAMMA"][::-1], wvss2_vmr)

        # Create gamma uncertainty output
        u_gamma_out = DecadesVariable(
            pd.Series(u_gamma, index=wvss2_vmr.index),
            name="SH_GAMMA_CU",
            flag=DecadesBitmaskFlag,
        )
        self.add_output(u_gamma_out)

        # Add specific heats outputs
        c_p_var = DecadesVariable(c_p, name="C_P", flag=DecadesBitmaskFlag)
        self.add_output(c_p_var)

        c_v_var = DecadesVariable(c_v, name="C_V", flag=DecadesBitmaskFlag)
        self.add_output(c_v_var)

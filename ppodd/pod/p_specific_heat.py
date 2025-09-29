import numpy as np
import pandas as pd

from ..decades import DecadesVariable, DecadesBitmaskFlag
from .base import PPBase, TestData, register_pp
from .shortcuts import _l
from ..utils.constants import MOL_MASS_H20, MOL_MASS_DRY_AIR, c_pd, c_vd


@register_pp("core")
class WetSpecificHeat(PPBase):
    r"""
    Calculate the specific heats at constant pressure, :math:`c_p` and constant
    volume, :math:`c_v`, and the ratio of specific heats, :math:`\gamma`, using the moist-air Mach number, :math:`M`, and
    the volume mixing ratio from the WVSSII.

    The specific humidity, :math:`q_h`, is given by

    .. math::
        q_h = \frac{\epsilon \cdot \text{VMR}}{1 + \epsilon \cdot \text{VMR}},

    where :math:`\epsilon` is the ratio of the molecular mass of water to that of dry air,
    and :math:`\text{VMR}` is the volume mixing ratio.

    Specific heat at constant pressure, :math:`c_p`, is given by

    .. math::
        c_p = c_{pd} \cdot \left(1 + q_h \cdot \left(\frac{8}{7 \cdot \epsilon} - 1\right)\right),

    where :math:`c_{pd}` is the specific heat of dry air at constant pressure.

    Specific heat at constant volume, :math:`c_v`, is given by

    .. math::
        c_v = c_{vd} \cdot \left(1 + q_h \cdot \left(\frac{6}{5 \cdot \epsilon} - 1\right)\right),

    where :math:`c_{vd}` is the specific heat of dry air at constant volume.

    The ratio of specific heats is given by

    .. math::
        \gamma = \frac{c_p}{c_v}.

    The uncertainty in the ratio of specific heats, :math:`\sigma_\gamma`, is given
    as polynomial coefficients for VMR in the flight constants file, and is evaluated
    here to produce an uncertainty estimate.

    See also:
        * :ref:`WVSS2Calibrated`
    """

    inputs = [
        "WVSS2F_VMR_C",  # Volume mixing ratio from WVSSII (corrected)
        "SH_UNC_GAMMA",  # Uncertainty in the ratio of specific heats at constant pressure and constant volume, corrected for humidity
    ]

    @staticmethod
    def test() -> TestData:
        """
        Return some dummy input data for testing usage.
        """
        return {
            "WVSS2F_VMR_C": ("data", _l(0.001, 0.005, 100), 32),
            "SH_UNC_GAMMA": ("const", [6.39e-05, 5.68e-08, -3.71e-12, 9.42e-17]),
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

    def process(self) -> None:
        """
        Process the input data and compute the outputs.
        """
        wvss2_vmr = self.dataset["WVSS2F_VMR_C"]()

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

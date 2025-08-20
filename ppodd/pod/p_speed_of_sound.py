import numpy as np

from ppodd.decades.flags import DecadesBitmaskFlag
from ppodd.decades.variable import DecadesVariable

from .base import PPBase, TestData, register_pp
from .shortcuts import _l
from ..utils.constants import ICAO_STD_DENSITY, ICAO_STD_PRESS, ICAO_STD_TEMP


@register_pp("core")
class SpeedOfSound(PPBase):
    r"""
    Calculates the speed of sound local to the aircraft.

    Speed of sound, :math:`a`, is given as

    .. math::
        a = \sqrt{\gamma \cdot \frac{p_0}{\rho_0} \cdot \frac{T}{T_0}},

    where :math:`\gamma` is the specific heat ratio, :math:`p_0` is the reference
    pressure, :math:`\rho_0` is the reference density, :math:`T` is the local temperature,
    from the deiced temperature probe, and :math:`T_0` is the reference temperature.

    See also:
        * :ref:`PRTTemperatures`
        * :ref:`WetSpecificHeat`
    """

    inputs = ["SH_GAMMA", "TAT_DI_R"]

    @staticmethod
    def test() -> TestData:
        return {
            "SH_GAMMA": ("data", _l(1.4, 1.4, 100), 32),
            "TAT_DI_R": ("data", _l(250, 300, 100), 32),
        }

    def declare_outputs(self) -> None:
        self.declare(
            "SPEED_OF_SOUND",
            units="m s-1",
            frequency=32,
            long_name="Local speed of sound",
            write=False,
        )

    def process(self) -> None:
        self.get_dataframe()
        assert self.d is not None, "Instance dataframe is None"

        self.d["SPEED_OF_SOUND"] = np.sqrt(
            self.d["SH_GAMMA"].interpolate(limit=32)
            * (ICAO_STD_PRESS * 100)
            / ICAO_STD_DENSITY
            * self.d["TAT_DI_R"]
            / ICAO_STD_TEMP
        )

        self.add_output(
            DecadesVariable(self.d["SPEED_OF_SOUND"], flag=DecadesBitmaskFlag)
        )

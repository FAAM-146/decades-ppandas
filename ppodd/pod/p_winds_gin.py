import pandas as pd
import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from .base import PPBase, register_pp
from .shortcuts import _o, _z

ROLL_THRESH = 2


@register_pp("core")
class GINWinds(PPBase):
    r"""
    Calculates  a horizontal wind vector from the aircraft true air speed,
    derived from the air data computer, and the speed and heading from the
    GPS-aided inertial navigation unit.

    The resulting corrected TAS may be modified with a scaling correction,
    specified in the constants as ``GINWIND_TASCOR``.

    The eastward and northward components of TAS, :math:`\text{TAS}_u` and
    :math:`\text{TAS}_v` are given by

    .. math::
        \text{TAS}_u &= \text{TAS}\cos(\theta - 90),\\
        \text{TAS}_v &= \text{TAS}\sin(\theta - 90).

    Where :math:`\theta` is the aircraft heading, which may be corrected with
    an offset given in the constant ``GIN_HDG_OFFSET`` to account for any
    misalignment of the GIN to the aircraft axis. The horizontal winds
    :math:`u` and :math:`v` are then given by

    .. math::
        u &= u_G - \text{TAS}_u,\\
        v &= v_G + \text{TAS}_v,

    where :math:`u_G` and :math:`v_G` are the eastward and northward components
    of the aircraft speed, reported by the GIN.
    """

    inputs = [
        "GIN_HDG_OFFSET",
        "VELE_GIN",
        "VELN_GIN",
        "HDG_GIN",
        "TAS_RVSM",
        "ROLL_GIN",
    ]

    @staticmethod
    def test() -> dict[str, tuple]:
        return {
            "GIN_HDG_OFFSET": ("const", 0),
            "VELE_GIN": ("data", 100 * _o(100), 32),
            "VELN_GIN": ("data", 100 * _o(100), 32),
            "HDG_GIN": ("data", _z(100), 32),
            "TAS_RVSM": ("data", 130 * _o(100), 32),
            "ROLL_GIN": ("data", _z(100), 32),
        }

    def declare_outputs(self) -> None:
        """
        Declare the output variables.
        """
        self.declare(
            "U_NOTURB",
            units="m s-1",
            frequency=1,
            long_name=(
                "Eastward wind component derived from aircraft " "instruments and GIN"
            ),
            standard_name="eastward_wind",
        )

        self.declare(
            "V_NOTURB",
            units="m s-1",
            frequency=1,
            long_name=(
                "Northward wind component derived from aircraft " "instruments and GIN"
            ),
            standard_name="northward_wind",
        )

    def calc_noturb_wspd(self) -> None:
        """
        Calculate the noturb u and v wind components,  as the difference
        between the aircraft ground and air vectors.
        """
        d = self.d
        assert d is not None

        try:
            tas_scale_factor = self.dataset["GINWIND_TASCOR"]
        except KeyError:
            # No GINWIND_TASCOR
            tas_scale_factor = 1

        d.TAS *= tas_scale_factor

        d.HDG_GIN += self.dataset["GIN_HDG_OFFSET"]
        d.HDG_GIN %= 360
        air_spd_east = np.cos(np.deg2rad(d.HDG_GIN - 90.0)) * d.TAS
        air_spd_north = np.sin(np.deg2rad(d.HDG_GIN - 90.0)) * d.TAS

        d["U_NOTURB"] = d.VELE_GIN - air_spd_east
        d["V_NOTURB"] = d.VELN_GIN + air_spd_north

    def process(self) -> None:
        """
        Processing entry point.
        """
        start_time = self.dataset["TAS_RVSM"].index[0].round("1s")
        end_time = self.dataset["TAS_RVSM"].index[-1].round("1s")

        self.get_dataframe(
            method="onto",
            index=pd.date_range(start=start_time, end=end_time, freq="1s"),
            circular=["HDG_GIN"],
            limit=50,
        )
        assert self.d is not None

        self.calc_noturb_wspd()

        u = DecadesVariable(self.d.U_NOTURB, flag=DecadesBitmaskFlag)
        v = DecadesVariable(self.d.V_NOTURB, flag=DecadesBitmaskFlag)

        for var in (u, v):
            assert isinstance(var.flag, DecadesBitmaskFlag)
            var.flag.add_mask(
                self.d.ROLL_GIN.abs() > ROLL_THRESH, "roll exceeds threshold"
            )

            self.add_output(var)

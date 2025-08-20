"""
This module provides the postprocessing module AirSpeed, which calculates the
aircraft airspeeds from the RVSM system. See the class docstring for further
details.
"""

# pylint: disable=invalid-name

from ppodd.decades import DecadesVariable, DecadesBitmaskFlag
from ppodd.pod.base import PPBase, register_pp, TestData
from ppodd.pod.shortcuts import _l
from ppodd.utils.conversions import knots_to_ms


@register_pp("core")
class AirSpeed(PPBase):
    r"""
    Calculates aircraft indicated and true air speeds.

    Indicated airspeed, IAS (note that we refer to this as indicated airspeed, but
    in aviation terms it is more accurately described as calibrated or computed airspeed),
    is read from the air data computer over the ARINC-429 bus. These data are stored as
    knots * 32, so the indicated airspeed is given as

    .. math::
        \text{IAS} = \text{IAS}_{\text{ARINC}} \times \frac{0.514444}{32},

    where :math:`\text{IAS}_{\text{ARINC}}` is the value recorded from the ARINC bus. The
    conversion from knots to meters per second is done using the function `knots_to_ms` from
    `ppodd.utils.conversions`.

    True airspeed is given as

    .. math::
        \text{TAS} = T_c M a

    where :math:`T_c` is a TAS correction term, defined in the flight constants,
    :math:`M` is the Mach number, and :math:`a` is the local speed of sound.

    See also:
        * :ref:`WetMach`
        * :ref:`DryMach`
        * :ref:`SpeedOfSound`

    """

    inputs = [
        "TASCORR",  #  Airspeed correction factor (const)
        "PRTAFT_ind_air_speed",  # indicated airspeed from the RVSM system (DLU)
        "MACH",  #  Mach number (derived)
        "SPEED_OF_SOUND",  #  Speed of sound (derived)
    ]

    @staticmethod
    def test() -> TestData:
        """
        Return some dummy input data for testing usage.
        """
        return {
            "TASCORR": ("const", 1.0),
            "MACH": ("data", _l(0.3, 0.8, 100), 32),
            "SPEED_OF_SOUND": ("data", _l(340, 350, 100), 32),
            "PRTAFT_ind_air_speed": ("data", _l(250 * 32, 300 * 32, 100), 32),
        }

    def declare_outputs(self) -> None:
        """
        Declare all of the output variables produced by this module, through
        calls to self.declare.
        """

        self.declare(
            "IAS_RVSM",
            units="m s-1",
            frequency=32,
            long_name=("Computed air speed from the aircraft RVSM (air data) system"),
        )

        self.declare(
            "TAS_RVSM",
            units="m s-1",
            frequency=32,
            long_name=(
                "True air speed from the aircraft RVSM (air data) "
                "system and deiced temperature"
            ),
            standard_name="platform_speed_wrt_air",
        )

    def calc_ias(self) -> None:
        """
        Calculate indicated airspeed from the RVSM system. Store this in the
        instance dataframe.
        """
        d = self.d

        if d is None:
            raise ValueError("Instance dataframe is None")

        ias = knots_to_ms(d["PRTAFT_ind_air_speed"] / 32)

        d["IAS_RVSM"] = ias

    def calc_tas(self) -> None:
        """
        Calculate true airspeed from the RVSM system and the deiced
        temperature. Store this in the instance dataframe.
        """
        d = self.d

        if d is None:
            raise ValueError("Instance dataframe is None")

        tas = self.dataset["TASCORR"] * d["SPEED_OF_SOUND"] * d["MACH"]

        d["TAS_RVSM"] = tas

    def process(self) -> None:
        """
        Module entry hook.
        """

        # Get all of the required inputs.
        self.get_dataframe()

        if self.d is None:
            raise ValueError("Instance dataframe is None")

        # Run required calculations in turn. These are stored in instance
        # state.
        self.calc_ias()
        self.calc_tas()

        # Create output variables for the indicated and true airspeeds.
        ias = DecadesVariable(self.d["IAS_RVSM"], flag=DecadesBitmaskFlag)
        tas = DecadesVariable(self.d["TAS_RVSM"], flag=DecadesBitmaskFlag)

        # Flag the data wherever the mach number is out of range.
        for _var in (ias, tas):
            assert isinstance(
                _var.flag, DecadesBitmaskFlag
            )  # TODO: shouldn't need this with generics

            # _var.flag.add_mask(
            #     self.d["MACHNO_FLAG"],
            #     "mach out of range",
            #     (
            #         "Either static or dynamic pressure out of acceptable limits "
            #         "during calculation of mach number."
            #     ),
            # )
            self.add_output(_var)

"""
This module provides a processing module which calculates cabin pressure, see
class docstring for more info.
"""

import numpy as np

from vocal.types import DerivedString

from ppodd.decades.flags import DecadesClassicFlag

from ..decades import DecadesVariable
from ..decades.attributes import DocAttribute
from .base import PPBase, register_pp, TestData
from .shortcuts import _c, _l

VALID_MIN = 650
VALID_MAX = 1050


@register_pp("core")
class CabinPressure(PPBase):
    r"""
    Derives cabin pressure from a pressure transducer located in the core
    console. A polynomial fit, with coefficients provided in the constants
    variable ``CALCABP``, converts DLU counts :math:`\rightarrow` transducer
    voltage :math:`\rightarrow` pressure.
    """

    inputs = ["CALCABP", "CORCON_cabin_p"]

    @staticmethod
    def test() -> TestData:
        """
        Return some dummy input data for testing.
        """
        return {
            "CALCABP": ("const", [2.75, 3.3e-2, -2.5e-10]),
            "CABP_SN": ("const", DocAttribute(value="1234", doc_value=DerivedString)),
            "CORCON_cabin_p": (
                "data",
                _c([_l(30000, 25000, 50), _l(25000, 35000, 50)]),
                1,
            ),
        }

    def declare_outputs(self) -> None:
        """
        Declare the outputs that are going to be written by this module.
        """

        self.declare(
            "CAB_PRES",
            units="hPa",
            frequency=1,
            long_name="Cabin Pressure",
            sensor_manufacturer="Rosemount Aerospace Inc.",
            sensor_model="1201F2",
            sensor_serial_number=self.dataset.lazy["CABP_SN"],
        )

    def bounds_flag(self) -> None:
        """
        Create a flag based on simple bounds checking. Flag as 2 wherever the
        data is outside a specified min/max.
        """

        if self.d is None:
            raise ValueError("No data to process")

        self.d["BOUNDS_FLAG"] = 0
        self.d.loc[self.d["CAB_PRES"] > VALID_MAX, "BOUNDS_FLAG"] = 1
        self.d.loc[self.d["CAB_PRES"] < VALID_MIN, "BOUNDS_FLAG"] = 1

    def process(self) -> None:
        """
        Processing entry point.
        """
        _cals = self.dataset["CALCABP"][::-1]
        self.get_dataframe()

        if self.d is None:
            raise ValueError("No data to process")

        # Polynomial conversion of dlu counts to pressure
        self.d["CAB_PRES"] = np.polyval(_cals, self.d["CORCON_cabin_p"])

        # Create a simple flag based on a defined valid limit
        self.bounds_flag()

        # Create the output variable
        cab_press = DecadesVariable(self.d["CAB_PRES"])

        assert isinstance(cab_press.flag, DecadesClassicFlag)  # TODO: use generics

        # Create the data quality flag meanings.
        cab_press.flag.add_meaning(0, "data good", "Data are considered valid")
        cab_press.flag.add_meaning(
            1,
            "pressure out of range",
            ("Data are outside the valid range " f"[{VALID_MIN}, {VALID_MAX}] hPa"),
        )

        # Add the flag to the output
        cab_press.flag.add_flag(self.d["BOUNDS_FLAG"])

        self.add_output(cab_press)

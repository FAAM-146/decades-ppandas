import numpy as np
import pandas as pd

from dataclasses import dataclass

import scipy
from vocal.types import OptionalDerivedString

from ppodd.utils.utils import get_constant_groups

from ..decades import DecadesVariable, DecadesBitmaskFlag, DecadesDataset
from ..utils import flagged_avg
from ..decades.attributes import DocAttribute
from ..exceptions.messages import EM_CANNOT_RUN_MODULE_CONSTANTS, EM_CANNOT_INIT_MODULE
from .base import PPBase, register_pp
from .shortcuts import _o


@dataclass
class Calibration:
    sensitivity: float
    zero: float
    sensitivity_unc: float
    zero_unc: float
    time: pd.Timestamp | None = None


@dataclass
class TargetComparison:
    bias: float
    uncertainty: float
    mask: pd.Series


def two_point_calibration(
    x1: float,
    y1: float,
    sx1: float,
    sy1: float,
    x2: float,
    y2: float,
    sx2: float,
    sy2: float,
) -> Calibration:
    """
    Two-point calibration with analytic uncertainty propagation.

    Args:
        x1: first x point
        y1: first y point
        sx1: uncertainty in first x point
        sy1: uncertainty in first y point
        x2: second x point
        y2: second y point
        sx2: uncertainty in second x point
        sy2: uncertainty in second y point

    Returns:
        Calibration: the calibration constants with 1-sigma uncertainties.
    """
    dx = x2 - x1
    dy = y2 - y1

    # Sensitivity
    m = dy / dx

    # Sensitivity uncertainty
    sm2 = (sy1**2 + sy2**2) / dx**2 + (dy**2 / dx**4) * (sx1**2 + sx2**2)
    sm = np.sqrt(sm2)

    # Zero
    c = y1 - m * x1

    # Zero uncertainty
    sc2 = sy1**2 + (x1**2) * sm2 + (m**2) * sx1**2
    sc = np.sqrt(sc2)

    return Calibration(sensitivity=m, zero=c, sensitivity_unc=sm, zero_unc=sc)


@dataclass
class AL55COCalData:
    """
    A container for the AL55CO calibration constants.
    """

    # High concentration mixing ratio
    cyl_hi_mr: float

    # Low concentration mixing ratio
    cyl_lo_mr: float

    # High concentration mixing ratio uncertainty
    cyl_hi_unc: float

    # Low concentration mixing ratio uncertainty
    cyl_lo_unc: float

    # High concentration cylinder serial number
    cyl_hi_sn: str

    # Low concentration cylinder serial number
    cyl_lo_sn: str

    # WMO scale in use
    wmo_scale: str

    # Calibration information URL
    calinfo_url: str

    @classmethod
    def from_dataset(cls, dataset: DecadesDataset) -> "AL55COCalData":
        """
        Extract the calibration constants from the dataset.

        Args:
            dataset: The dataset to extract the constants from.

        Returns:
            An AL55COCalData instance populated with the constants.

        Raises:
            RuntimeError: If any of the required constants are not found in the
                dataset.
        """

        try:
            return cls(
                cyl_hi_mr=dataset["AL55CO_HI_MR"],
                cyl_lo_mr=dataset["AL55CO_LO_MR"],
                cyl_hi_unc=dataset["AL55CO_HI_CU"] / 2,
                cyl_lo_unc=dataset["AL55CO_LO_CU"] / 2,
                cyl_hi_sn=dataset["AL55CO_HI_SN"],
                cyl_lo_sn=dataset["AL55CO_LO_SN"],
                wmo_scale=dataset["AL55CO_WMO_SCALE"],
                calinfo_url=dataset["AL55CO_CALINFO_URL"],
            )
        except KeyError as err:
            message = EM_CANNOT_RUN_MODULE_CONSTANTS.format(
                module_name="AL55CO",
                constants=",".join(
                    [
                        "AL55CO_HI_MR",
                        "AL55CO_LO_MR",
                        "AL55CO_HI_CU",
                        "AL55CO_LO_CU",
                        "AL55CO_HI_SN",
                        "AL55CO_LO_SN",
                        "AL55CO_WMO_SCALE",
                        "AL55CO_CALINFO_URL",
                    ]
                ),
            )
            raise RuntimeError(message) from err


@register_pp("core")
class AL55CO(PPBase):
    r"""
    Provides Carbon Monoxide concentration from the AL5005 instrument.

    The module can be run in two modes, depending on the value of the
    `AL55CO_USE_CALS` constant. If this is `True`, the module will use the
    in-flight target, zero and span calibrations to calculate the CO
    concentration, and will provide an uncertainty estimate. If `False`, the
    module will simply pass through the concentration provided by the instrument
    without any uncertainty estimate. In this case, an uncertainty estimate
    may be provided in the document referenced in the calibration URL.
    """

    inputs = [
        "AL55CO_conc",  # CO concentration
        "AL55CO_counts",
        "AL55CO_V1",
        "AL55CO_V2",
        "AL55CO_V3",
        "AL55CO_V4",
        "WOW_IND",
        "AL55CO_SN",
        "AL55CO_USE_CALS",
    ]

    @staticmethod
    def test():
        """
        Provide dummy input data for testing.
        """
        return {
            "AL55CO_conc": ("data", _o(100), 1),
            "AL55CO_counts": ("data", _o(100), 1),
            "AL55CO_USE_CALS": ("const", True),
            "AL55CO_CALINFO_URL": (
                "const",
                DocAttribute(value="https://some.url", doc_value=OptionalDerivedString),
            ),
        }

    def declare_outputs(self):
        try:
            _ = self.dataset["AL55CO_USE_CALS"]
        except KeyError as err:
            message = EM_CANNOT_INIT_MODULE.format(
                module_name=self.__class__.__name__, constants="AL55CO_USE_CALS"
            )
            raise RuntimeError(message) from err

        self.declare(
            "CO_AERO",
            units="ppb",
            frequency=1,
            long_name=(
                "Carbon monoxide concentration measured by the AL5005 instrument"
            ),
            standard_name="mole_fraction_of_carbon_monoxide_in_air",
            instrument_manufacturer="Aero-Laser GmbH",
            instrument_model="AL5005",
            instrument_serial_number=self.dataset.lazy["AL55CO_SN"],
            calibration_information=None,
            calibration_url=self.dataset.lazy["AL55CO_CALINFO_URL"],
        )

        if self.dataset["AL55CO_USE_CALS"]:
            self.declare(
                "AL55CO_interpolated_sens", units="Hz ppb-1", frequency=1, write=False
            )

            self.declare(
                "AL55CO_interpolated_zero", units="Hz", frequency=1, write=False
            )
            self.declare(
                "CO_AERO_CU",
                units="ppb",
                frequency=1,
                long_name=("Combined uncertainty estimate for CO_AERO"),
                coverage_content_type="auxiliaryInformation",
            )

    def process_without_cals(self) -> None:
        co_out = DecadesVariable(
            self.dataset["AL55CO_conc"].data, name="CO_AERO", flag=DecadesBitmaskFlag
        )

        self.add_output(co_out)

    def get_bias_and_uncertainty(self) -> TargetComparison:
        """
        Get the bias and uncertainty for the AL55CO instrument.

        Returns:
            A tuple of (bias, uncertainty)
        """
        df = self.d
        assert df is not None

        *_, target_mask = self.get_masks()

        target_groups = get_constant_groups(target_mask)

        for group in target_groups:
            target_mask[group[1].index[:5]] = False

        target_conc = self.dataset["AL55CO_TAR_MR"]
        target_data = df[target_mask == 1]["conc_final"]
        mu, std = scipy.stats.norm.fit(target_data - target_conc)

        return TargetComparison(
            bias=float(mu), uncertainty=float(std), mask=target_mask
        )

    def get_masks(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Get the calibration and target masks from the AL55CO status bits.

        Returns:
            A tuple of three pandas Series, containing the hi_cal_mask,
            lo_cal_mask and target_mask.
        """
        v1 = self.dataset["AL55CO_V1"]()
        v2 = self.dataset["AL55CO_V2"]()
        v3 = self.dataset["AL55CO_V3"]()
        v4 = self.dataset["AL55CO_V4"]()

        hi_cal_mask = v1 & v2 & v3 & v4
        lo_cal_mask = v1 & v2 & (~v3) & v4
        target_mask = v1 & (~v2) & (~v3) & v4

        return hi_cal_mask, lo_cal_mask, target_mask

    def get_calibration_flag(self) -> pd.Series:
        """
        Get a calibration flag indicating when the instrument is in any
        calibration mode.

        Returns:
            A pandas Series containing the calibration flag.
        """
        assert self.d is not None
        cal_flag = pd.Series(0, index=self.d.index)
        v1_flag = self.d["AL55CO_V1"]
        cal_groups = get_constant_groups(cal_flag)

        # Flag when V1 is set, and for 5 seconds after leaving calibration
        cal_flag.loc[v1_flag == 1] = 1
        for group in cal_groups:
            last_index = group[1].index[-1]
            cal_flag.loc[last_index : last_index + pd.Timedelta(seconds=5)] = 1
        return cal_flag

    def get_wow_flag(self) -> pd.Series:
        """
        Get the WOW flag for the dataset.

        Returns:
            A pandas Series containing the WOW flag.
        """
        assert self.d is not None
        wow_flag = self.d["WOW_IND"].ffill().bfill()
        takeoff_time = self.dataset.takeoff_time

        # Maintain the WOW flag for 5 seconds after takeoff
        wow_flag.loc[
            takeoff_time - pd.Timedelta(seconds=1) : takeoff_time
            + pd.Timedelta(seconds=5)
        ] = 1
        return wow_flag
    
    def get_low_flow_flag(self) -> pd.Series:

    def get_flags(self) -> dict[str, tuple[str, pd.Series]]:
        return {}

    def process_with_cals(self) -> None:
        self.get_dataframe()
        assert self.d is not None
        constants = AL55COCalData.from_dataset(self.dataset)

        hi_cal_mask, lo_cal_mask, target_mask = self.get_masks()

        self.d = self.d.assign(
            hi_cal_flag=hi_cal_mask, lo_cal_flag=lo_cal_mask, target_flag=target_mask
        )

        df = self.d

        flagged_avg(
            df,
            flag_col="hi_cal_flag",
            data_col="AL55CO_counts",
            out_name="hi_cal_avg",
            with_std=True,
            std_name="hi_cal_std",
            skip_start=5,
        )
        flagged_avg(
            df,
            flag_col="lo_cal_flag",
            data_col="AL55CO_counts",
            out_name="lo_cal_avg",
            with_std=True,
            std_name="lo_cal_std",
            skip_start=5,
        )
        flagged_avg(
            df,
            flag_col="target_flag",
            data_col="AL55CO_counts",
            out_name="target_avg",
            with_std=True,
            std_name="target_std",
            skip_start=5,
        )

        hi_points = df["hi_cal_avg"].dropna().values
        hi_times = df["hi_cal_avg"].dropna().index
        lo_points = df["lo_cal_avg"].dropna().values
        lo_times = df["lo_cal_avg"].dropna().index
        hi_std = df["hi_cal_std"].dropna().values
        lo_std = df["lo_cal_std"].dropna().values

        df["interpolated_sens"] = np.nan
        df["interpolated_zero"] = np.nan
        calibrations = []

        for hi, lo, hi_std, lo_std, hi_time, lo_time in zip(
            hi_points, lo_points, hi_std, lo_std, hi_times, lo_times
        ):
            cal = two_point_calibration(
                constants.cyl_lo_mr,
                lo,
                constants.cyl_lo_unc,
                lo_std,
                constants.cyl_hi_mr,
                hi,
                constants.cyl_hi_unc,
                hi_std,
            )

            mean_time = (lo_time + (hi_time - lo_time) / 2).floor("s")

            cal.time = mean_time
            calibrations.append(cal)

            df.at[mean_time, "interpolated_sens"] = cal.sensitivity
            df.at[mean_time, "interpolated_zero"] = cal.zero

        df["interpolated_sens"] = (
            df["interpolated_sens"].interpolate(method="time").ffill().bfill()
        )
        df["interpolated_zero"] = (
            df["interpolated_zero"].interpolate(method="time").ffill().bfill()
        )

        df["conc_final"] = (df["AL55CO_counts"] - df["interpolated_zero"]) / df[
            "interpolated_sens"
        ]

        target_comparison = self.get_bias_and_uncertainty()

        self.add_output(
            DecadesVariable(df["conc_final"], name="CO_AERO", flag=DecadesBitmaskFlag)
        )

        self.add_output(
            DecadesVariable(
                pd.Series(
                    np.ones(df.shape[0]) * target_comparison.uncertainty, index=df.index
                ),
                name="CO_AERO_CU",
                flag=None,
            )
        )

        self.add_output(
            DecadesVariable(
                df["interpolated_sens"],
                name="AL55CO_interpolated_sens",
                flag=None,
            )
        )

        self.add_output(
            DecadesVariable(
                df["interpolated_zero"],
                name="AL55CO_interpolated_zero",
                flag=None,
            )
        )

        self.dataset.add_constant(
            "AL55CO_CALCULATED_CALS",
            calibrations,
        )

    def process(self):
        """
        Processing entry hook.
        """
        print("************** AL55CO PROCESSING ****************")
        if self.dataset["AL55CO_USE_CALS"]:
            self.process_with_cals()
        else:
            self.process_without_cals()

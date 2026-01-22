import logging

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy

from vocal.types import OptionalDerivedString, OptionalDerivedFloat32

from .base import PPBase, register_pp
from .shortcuts import _o, _c, _z, _rn
from ..utils import get_constant_groups
from ..decades import DecadesVariable, DecadesBitmaskFlag, DecadesDataset, flags
from ..utils import flagged_avg
from ..decades.attributes import DocAttribute
from ..exceptions.messages import EM_CANNOT_RUN_MODULE_CONSTANTS, EM_CANNOT_INIT_MODULE

logger = logging.getLogger(__name__)


@dataclass
class CalibrationPoint:
    """
    Represents a single calibration point with its uncertainty.
    """

    x: float
    y: float
    x_unc: float
    y_unc: float


@dataclass
class Calibration:
    """
    Represents a two-point calibration with sensitivity and zero,
    along with their uncertainties.
    """

    sensitivity: float
    zero: float
    sensitivity_unc: float
    zero_unc: float
    time: pd.Timestamp | None = None
    points: list[CalibrationPoint] | None = None


@dataclass
class TargetComparison:
    """
    Represents the comparison of measured values against target values.
    """

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

    return Calibration(
        sensitivity=m,
        zero=c,
        sensitivity_unc=sm,
        zero_unc=sc,
        points=[
            CalibrationPoint(x=x1, y=y1, x_unc=sx1, y_unc=sy1),
            CalibrationPoint(x=x2, y=y2, x_unc=sx2, y_unc=sy2),
        ],
    )


@dataclass
class AL55COCalData:
    """
    A container for the AL55CO calibration constants.
    """

    # High concentration mixing ratio
    cyl_hi_mr: float

    # Low concentration mixing ratio
    cyl_lo_mr: float

    # Target cylinder concentration mixing ratio
    cyl_tar_mr: float

    # High concentration mixing ratio uncertainty
    cyl_hi_unc: float

    # Low concentration mixing ratio uncertainty
    cyl_lo_unc: float

    # Target cylinder concentration mixing ratio uncertainty
    cyl_tar_unc: float

    # High concentration cylinder serial number
    cyl_hi_sn: str

    # Low concentration cylinder serial number
    cyl_lo_sn: str

    # Target cylinder serial number
    cyl_tar_sn: str

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
                cyl_tar_mr=dataset["AL55CO_TAR_MR"],
                cyl_hi_unc=dataset["AL55CO_HI_EU"] / 2,
                cyl_lo_unc=dataset["AL55CO_LO_EU"] / 2,
                cyl_tar_unc=dataset["AL55CO_TAR_EU"] / 2,
                cyl_hi_sn=dataset["AL55CO_HI_SN"],
                cyl_lo_sn=dataset["AL55CO_LO_SN"],
                cyl_tar_sn=dataset["AL55CO_TAR_SN"],
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
                        "AL55CO_TAR_MR",
                        "AL55CO_HI_EU",
                        "AL55CO_LO_EU",
                        "AL55CO_TAR_EU",
                        "AL55CO_HI_SN",
                        "AL55CO_LO_SN",
                        "AL55CO_TAR_SN",
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
    ``AL55CO_USE_CALS`` constant.

    If ``AL55CO_USE_CALS`` is `False`, the module simply
    passes through the ``AL55CO_conc`` variable as the output ``CO_AERO``,
    with appropriate flagging and metadata. If an uncertainty is provided
    in the flight constants as ``AL55CO_CU``, this is also passed through
    as the ``CO_AERO_CU`` variable.

    If ``AL55CO_USE_CALS`` is `True`, the module applies in-flight calibrations
    to derive the CO concentration. Two-point calibrations are performed
    using the high and low concentration cylinder measurements, and the
    sensitivity and zero are interpolated to provide a final concentration, given by

    .. math::
        \left[\text{CO}\right] = \frac{\text{CO}_{|\text{counts}} - Z}{S},

    where :math:`\text{CO}_{|\text{counts}}` are the counts reported
    by the instrument, :math:`Z` is the interpolated zero, and :math:`S` is the
    interpolated sensitivity.

    An uncertainty estimate, ``CO_AERO_CU``, is calculated as the standard deviation
    of the instrument during target measurements, summed in quadrature with the
    uncertainties of the target cylinder concentration.

    Any bias against the target cylinder concentration is calculated and corrected
    for.

    The following constants may be given:
        * ``AL55CO_USE_CALS`` (required) = true/false. Whether to use in-flight calibrations (true) or rely on the laboratory calibration (false)
        * ``AL55CO_CU`` The (combined [k=1]) uncertainty (ppb) to use when ``AL55CO_USE_CALS`` is False
        * ``AL55CO_MFM_MIN`` (optional, default=1.2) The minimum flow rate (V), below which data are flagged
        * ``AL55CO_HI_MR`` (required if ``AL55CO_USE_CALS`` is True) The CO mixing ratio (ppb), high cylinder
        * ``AL55CO_LO_MR`` (required if ``AL55CO_USE_CALS`` is True) The CO mixing ratio (ppb), low cylinder
        * ``AL55CO_TAR_MR`` (required if ``AL55CO_USE_CALS`` is True) The CO mixing ratio (ppb), target cylinder
        * ``AL55CO_HI_EU`` (required if ``AL55CO_USE_CALS`` is True) The expanded [k=2] uncertainty (ppb), high cylinder
        * ``AL55CO_LO_EU`` (required if ``AL55CO_USE_CALS`` is True) The expanded [k=2] uncertainty (ppb), low cylinder
        * ``AL55CO_TAR_EU`` (required if ``AL55CO_USE_CALS`` is True) The expanded [k=2] uncertainty (ppb), target cylinder
        * ``AL55CO_HI_SN`` (required if ``AL55CO_USE_CALS`` is True) The serial number of the high cylinder
        * ``AL55CO_LO_SN`` (required if ``AL55CO_USE_CALS`` is True) The serial number of the low cylinder
        * ``AL55CO_TAR_SN``(required if ``AL55CO_USE_CALS`` is True)  The serial number of the target cylinder
        * ``AL55CO_WMO_SCALE`` (required) The WMO scale in use
        * ``AL55CO_CALINFO_URL`` (optional) The DOI for calibration information
        * ``AL55CO_INSTRINFO_URL`` (optional) The URL for instrument information
    """

    inputs = [
        "AL55CO_conc",  # CO concentration
        "AL55CO_counts",
        "AL55CO_V1",
        "AL55CO_V2",
        "AL55CO_V3",
        "AL55CO_V4",
        "AL55CO_MFM",
        "WOW_IND",
        "AL55CO_SN",
        "AL55CO_USE_CALS",
    ]

    @staticmethod
    def test():
        """
        Provide dummy input data for testing.
        """

        conc_data = _c(
            [
                _rn(150, 2, 100),
                _rn(600, 3, 50),
                _rn(150, 2, 25),
                _rn(100, 2, 50),
                _rn(150, 2, 275),
                _rn(300, 2, 100),
                _rn(150, 2, 400),
            ]
        )

        return {
            "AL55CO_conc": (
                "data",
                conc_data,
                1,
            ),
            "AL55CO_counts": ("data", (conc_data * 10000).astype(int), 1),
            "AL55CO_V1": (
                "data",
                _c([_z(100), _o(50), _z(25), _o(50), _z(275), _o(100), _z(400)]),
                1,
            ),
            "AL55CO_V2": (
                "data",
                _c([_z(100), _o(50), _z(25), _o(50), _z(275), _z(100), _z(400)]),
                1,
            ),
            "AL55CO_V3": (
                "data",
                _c([_z(100), _o(50), _z(25), _z(50), _z(275), _z(100), _z(400)]),
                1,
            ),
            "AL55CO_V4": (
                "data",
                _c([_z(100), _o(50), _z(25), _o(50), _z(275), _o(100), _z(400)]),
                1,
            ),
            "AL55CO_MFM": ("data", _o(1000) * 1.5, 1),
            "WOW_IND": ("data", _z(1000), 1),
            "AL55CO_SN": (
                "const",
                DocAttribute(value="A1234", doc_value=OptionalDerivedString),
            ),
            "AL55CO_USE_CALS": ("const", True),
            "AL55CO_MFM_MIN": ("const", 1.2),
            "AL55CO_HI_MR": (
                "const",
                DocAttribute(value=600.0, doc_value=OptionalDerivedFloat32),
            ),
            "AL55CO_LO_MR": (
                "const",
                DocAttribute(value=100.0, doc_value=OptionalDerivedFloat32),
            ),
            "AL55CO_TAR_MR": (
                "const",
                DocAttribute(value=300.0, doc_value=OptionalDerivedFloat32),
            ),
            "AL55CO_HI_EU": (
                "const",
                DocAttribute(value=2.0, doc_value=OptionalDerivedFloat32),
            ),
            "AL55CO_LO_EU": (
                "const",
                DocAttribute(value=2.0, doc_value=OptionalDerivedFloat32),
            ),
            "AL55CO_TAR_EU": (
                "const",
                DocAttribute(value=2.0, doc_value=OptionalDerivedFloat32),
            ),
            "AL55CO_HI_SN": (
                "const",
                DocAttribute(value="12345", doc_value=OptionalDerivedString),
            ),
            "AL55CO_LO_SN": (
                "const",
                DocAttribute(value="23456", doc_value=OptionalDerivedString),
            ),
            "AL55CO_TAR_SN": (
                "const",
                DocAttribute(value="34567", doc_value=OptionalDerivedString),
            ),
            "AL55CO_WMO_SCALE": (
                "const",
                DocAttribute(value="WMO-X2004A", doc_value=OptionalDerivedString),
            ),
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
            instrument_description=self.dataset.lazy["AL55CO_DESCRIPTION"],
            calibration_information=self.dataset.lazy["AL55CO_CALIBRATION_INFO"],
            calibration_url=self.dataset.lazy["AL55CO_CALINFO_URL"],
        )

        self.declare(
            "CO_AERO_CU",
            units="ppb",
            frequency=1,
            long_name=("Combined uncertainty estimate for CO_AERO"),
            coverage_content_type="auxiliaryInformation",
        )

        if self.dataset["AL55CO_USE_CALS"]:
            self.declare(
                "AL55CO_interpolated_sens",
                long_name="Interpolated sensitivity from AL55CO in-flight calibrations",
                units="Hz ppb-1",
                frequency=1,
                write=False,
            )

            self.declare(
                "AL55CO_interpolated_zero",
                long_name="Interpolated zero from AL55CO in-flight calibrations",
                units="Hz",
                frequency=1,
                write=False,
            )

    def process_without_cals(self) -> None:
        """
        Processing without in-flight calibrations. This is a simple pass-through
        of the AL55CO_conc variable.
        """
        logger.info("Processing AL55CO without in-flight calibrations.")

        self.dataset.add_constant(
            "AL55CO_CALIBRATION_INFO",
            DocAttribute(
                value=self.get_calibration_info_without_cals(),
                doc_value=OptionalDerivedString,
            ),
        )

        conc_output = DecadesVariable(
            self.dataset["AL55CO_conc"].data, name="CO_AERO", flag=DecadesBitmaskFlag
        )

        assert isinstance(conc_output.flag, DecadesBitmaskFlag)
        for flag_name, (flag_desc, flag_series) in self.get_flags().items():
            conc_output.flag.add_mask(flag_series, flag_name, flag_desc)

        self.add_output(conc_output)

        # We can't calculate uncertainty without calibrations, but it may be
        # given in the flight constants. If so, pass it through, otherwise
        # create a dummy variable filled with NaNs, marked as not to be written.
        try:
            uncertainty = self.dataset["AL55CO_CU"]
            unc_output = DecadesVariable(
                pd.Series(uncertainty, index=conc_output.data.index), name="CO_AERO_CU"
            )
        except KeyError:
            unc_output = DecadesVariable(
                pd.Series(np.nan, index=conc_output.data.index), name="CO_AERO_CU"
            )
            unc_output.write = False

        self.add_output(unc_output)

    def get_bias_and_uncertainty(self) -> TargetComparison:
        """
        Get the bias and uncertainty for the AL55CO instrument.

        Returns:
            A TargetComparison instance containing the bias, uncertainty and
            target mask.
        """
        df = self.d
        assert df is not None

        *_, target_mask = self.get_masks()

        target_groups = get_constant_groups(target_mask)

        logger.info(f"Found {len(target_groups)} target groups for bias calculation.")

        for group in target_groups:
            target_mask[group[1].index[:5]] = 0

        target_conc = self.dataset["AL55CO_TAR_MR"]
        target_data = df[target_mask == 1]["conc_final"]
        mu, std = scipy.stats.norm.fit(target_data - target_conc)

        logger.info(
            f"Calculated target bias: {mu:.2f} ppb and uncertainty: {std:.2f} ppb."
        )

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
        cal_flag.loc[v1_flag == 1] = 1
        cal_groups = get_constant_groups(cal_flag)

        # Flag when V1 is set, and for 5 seconds after leaving calibration
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
        """
        Get the low flow flag for the dataset. This flags when the MFM flow is below
        the valid minimum, as defined by the AL55CO_MFM_MIN constant.

        Returns:
            A pandas Series containing the low flow flag.
        """
        assert self.d is not None
        mfm_flow = self.d["AL55CO_MFM"]
        try:
            mfm_valid_min = self.dataset["AL55CO_MFM_MIN"]
        except KeyError:
            mfm_valid_min = 1.2
        low_flow_flag = pd.Series(0, index=self.d.index)
        low_flow_flag.loc[mfm_flow < mfm_valid_min] = 1
        return low_flow_flag

    def get_flags(self) -> dict[str, tuple[str, pd.Series]]:
        """
        Get the flags for the CO_AERO output variable.

        Returns:
            A dictionary mapping flag names to tuples of (description, flag Series).
        """
        return {
            flags.WOW: (
                "The aircraft is, or very recently was, on the ground.",
                self.get_wow_flag(),
            ),
            flags.CALIBRATION: (
                "The instrument is in, or has recently been in, a calibration mode.",
                self.get_calibration_flag(),
            ),
            "flow_below_min": (
                "The instrument flow is below the valid minimum.",
                self.get_low_flow_flag(),
            ),
        }

    def get_calibration_info_without_cals(self) -> str:
        try:
            wmo_scale = self.dataset["AL55CO_WMO_SCALE"]
        except KeyError:
            wmo_scale = "unknown"

        return (
            "Instrument not calibrated in flight - "
            "laboratory derived calibration coefficients are used instead. "
            "Instrument inlet is shared with ozone instruments. "
            f"WMO Scale: {wmo_scale}. "
        )

    def get_calibration_info_with_cals(self, bias: float) -> str:
        """
        Get a string describing the calibration information.

        Returns:
            A string containing the calibration information.
        """

        constants = AL55COCalData.from_dataset(self.dataset)

        cal_info = (
            "Instrument calibrated in flight using shared greenhouse gasses inlet. "
            f"High Cylinder: SN {constants.cyl_hi_sn}, "
            f"Mixing Ratio {constants.cyl_hi_mr} ppb ± {constants.cyl_hi_unc} ppb. "
            f"Low Cylinder: SN {constants.cyl_lo_sn}, "
            f"Mixing Ratio {constants.cyl_lo_mr} ppb ± {constants.cyl_lo_unc} ppb. "
            f"Target Cylinder: SN {constants.cyl_tar_sn}, "
            f"Mixing Ratio {constants.cyl_tar_mr} ppb ± {constants.cyl_tar_unc} ppb. "
            "Uncertainties are given as k=1 standard uncertainties (±1σ). "
            f"WMO Scale: {constants.wmo_scale}. "
            f"Measured bias against target cylinder: {bias:.2f} ppb, which has "
            f"been corrected in the output concentration. "
            f"Further information can be found at: {constants.calinfo_url}"
        )

        return cal_info

    def process_with_cals(self) -> None:
        """
        Processing with in-flight calibrations.
        """
        logger.info("Processing AL55CO with in-flight calibrations.")

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
        self.dataset.add_constant("AL55CO_TARGET_COMPARISON", target_comparison)

        conc_output = DecadesVariable(
            df["conc_final"], name="CO_AERO", flag=DecadesBitmaskFlag
        )

        assert isinstance(conc_output.flag, DecadesBitmaskFlag)
        for flag_name, (flag_desc, flag_series) in self.get_flags().items():
            conc_output.flag.add_mask(flag_series, flag_name, flag_desc)

        self.dataset.add_constant(
            "AL55CO_CALIBRATION_INFO",
            DocAttribute(
                value=self.get_calibration_info_with_cals(target_comparison.bias),
                doc_value=OptionalDerivedString,
            ),
        )
        self.add_output(conc_output)

        # Add the uncertainty output. This is the sun in quadrature of the
        # target comparison uncertainty and the uncertainty of the target
        # cylinder.
        self.add_output(
            DecadesVariable(
                pd.Series(
                    np.sqrt(
                        (np.ones(df.shape[0]) * target_comparison.uncertainty) ** 2
                        + (np.ones(df.shape[0]) * constants.cyl_hi_unc) ** 2
                    ),
                    index=df.index,
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
        if self.dataset["AL55CO_USE_CALS"]:
            self.process_with_cals()
        else:
            self.process_without_cals()

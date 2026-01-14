import numpy as np
import pandas as pd

from scipy import odr

from dataclasses import dataclass

from vocal.types import OptionalDerivedString

from ..decades import DecadesVariable, DecadesBitmaskFlag, DecadesDataset
from ..utils import flagged_avg
from ..decades.attributes import DocAttribute
from ..exceptions.messages import EM_CANNOT_RUN_MODULE_CONSTANTS, EM_CANNOT_INIT_MODULE
from .base import PPBase, register_pp
from .shortcuts import _o


def linear_model(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    A simple linear model function.

    Args:
        coeffs: The coefficients of the linear model, in the form
            [slope, intercept].
        x: The input x values.

    Returns:
        The output y values.
    """
    return coeffs[0] * x + coeffs[1]


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
                cyl_hi_unc=dataset["AL55CO_HI_CU"],
                cyl_lo_unc=dataset["AL55CO_LO_CU"],
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
        "AL55CO_USE_CALS",
        "AL55CO_SN",
        "AL55CO_V1",
        "AL55CO_V2",
        "AL55CO_V3",
        "AL55CO_V4",
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
            pass
            # self.declare(
            #     "CO_AERO_CU",
            #     units="ppb",
            #     frequency=1,
            #     long_name=("Combined uncertainty estimate for CO_AERO"),
            #     coverage_content_type="auxiliaryInformation",
            # )

    def process_without_cals(self):
        co_out = DecadesVariable(
            self.dataset["AL55CO_conc"].data, name="CO_AERO", flag=DecadesBitmaskFlag
        )

        self.add_output(co_out)

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

    def process_with_cals(self) -> None:
        self.get_dataframe()
        df = self.d
        assert df is not None
        constants = AL55COCalData.from_dataset(self.dataset)

        hi_cal_mask, lo_cal_mask, target_mask = self.get_masks()

        df = df.assign(
            hi_cal_flag=hi_cal_mask, lo_cal_flag=lo_cal_mask, target_flag=target_mask
        )

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

        color_cycles = iter(["tab:blue", "tab:orange", "tab:green", "tab:purple"])

        for i, (hi, lo, hi_std, lo_std, hi_time, lo_time) in enumerate(
            zip(hi_points, lo_points, hi_std, lo_std, hi_times, lo_times)
        ):
            # fit = np.polyfit(
            #     [constants.cyl_lo_mr, constants.cyl_hi_mr],
            #     [lo, hi],
            #     deg=1,
            # )
            data = odr.RealData(
                [constants.cyl_lo_mr, constants.cyl_hi_mr],
                [lo, hi],
                sx=[constants.cyl_lo_unc, constants.cyl_hi_unc],
                sy=[lo_std, hi_std],
            )

            model = odr.ODR(data, odr.Model(linear_model), beta0=[1.0, 0.0])
            output = model.run()
            fit = output.beta

            mean_time = (lo_time + (hi_time - lo_time) / 2).floor("s")

            df.at[mean_time, "interpolated_sens"] = fit[0]
            df.at[mean_time, "interpolated_zero"] = fit[1]

            import matplotlib.pyplot as plt

            # plt.figure()
            col = next(color_cycles)
            plt.errorbar(
                [constants.cyl_lo_mr, constants.cyl_hi_mr],
                [lo, hi],
                xerr=[constants.cyl_lo_unc, constants.cyl_hi_unc],
                yerr=[lo_std, hi_std],
                ls=" ",
                ecolor=col,
                elinewidth=0.5,
                label=f"Calibration {i + 1}",
            )
            x_fit = np.array(
                [100, max(constants.cyl_hi_mr * 1.1, constants.cyl_lo_mr * 1.1)]
            )
            y_fit = linear_model(fit, x_fit)
            plt.plot(x_fit, y_fit, "-", label="ODR Fit", linewidth=0.5, color=col)
        plt.xlabel("Mixing Ratio (ppb)")
        plt.ylabel("AL55CO Counts")
        plt.title("AL55CO Calibrations")
        plt.legend(fontsize="small")
        plt.show()

        df["interpolated_sens"] = (
            df["interpolated_sens"].interpolate(method="time").ffill().bfill()
        )
        df["interpolated_zero"] = (
            df["interpolated_zero"].interpolate(method="time").ffill().bfill()
        )

        df["conc_final"] = (df["AL55CO_counts"] - df["interpolated_zero"]) / df[
            "interpolated_sens"
        ]

        self.add_output(
            DecadesVariable(df["conc_final"], name="CO_AERO", flag=DecadesBitmaskFlag)
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

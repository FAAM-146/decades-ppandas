"""
This module provides flagging modules for the Rosemount temperature probes,
providing data quality information which can not be inferred during processing
of the temperatures.
"""

import numpy as np
import pandas as pd

from ppodd.flags.base import FlaggingBase

TEMPERATURE_THRESHOLD = 1.0
TEMPERATURE_VARIABLES = ("TAT_DI_R", "TAT_ND_R")
MASKED = 1
UNMASKED = 0


class RosemountTempDeltaFlag(FlaggingBase):
    """
    This class adds a flag to the rosemount temperatures if the
    two temperatures disagree by more than a given absolute value. This is
    given by the module constant TEMPERATURE_THRESHOLD.
    """

    inputs = list(TEMPERATURE_VARIABLES)
    flagged = list(TEMPERATURE_VARIABLES)

    def _get_flag(self) -> pd.Series:
        """
        Get the flag value for the new flag
        """

        # Get the absolute difference between the deiced and non-deiced
        # temperatures
        _diff = np.abs(self.dataset["TAT_DI_R"].array - self.dataset["TAT_ND_R"].array)

        # Create a mask for the flag
        mask = pd.Series(
            np.zeros_like(self.dataset["TAT_DI_R"].array),
            index=self.dataset["TAT_DI_R"].index,
        )
        mask.loc[_diff > TEMPERATURE_THRESHOLD] = MASKED

        return mask

    def _flag(self, test: bool = False) -> None:
        """
        Entry point for the flagging module.
        """

        if test:
            flag = self.test_flag
        else:
            flag = self._get_flag()

        # For each of the temperature add the threshold mask
        for var in TEMPERATURE_VARIABLES:
            self.add_mask(
                var,
                flag,
                "discrepancy threshold exceeded",
                (
                    "The discrepancy between the deiced and non-deiced temperature "
                    f"sensors is greater than {TEMPERATURE_THRESHOLD} K."
                ),
            )


class RosemountTempCloudFlag(FlaggingBase):
    """
    This class adds a flag to all rosemount temperature variables if the aircraft
    is in cloud, as indicated by NV_CLEAR_AIR_MASK, derived from the Nevzorov
    probe.
    """

    inputs = ["NV_CLEAR_AIR_MASK"]
    flagged = list(TEMPERATURE_VARIABLES)

    def _get_flag(self, var: str) -> pd.Series:
        """
        Entry point for the flagging module.
        """

        try:
            index = self.dataset[var].index
        except (KeyError, AttributeError):
            raise ValueError("Unable to get variable index")

        clear_air_series = self.dataset["NV_CLEAR_AIR_MASK"]()
        mask = clear_air_series == UNMASKED
        nv_start = clear_air_series.index[0]
        nv_end = clear_air_series.index[-1]
        mask = mask.reindex(index)
        mask.loc[(mask.index <= nv_start) | (mask.index >= nv_end)] = 0
        mask.fillna(method="ffill", inplace=True)
        mask.fillna(method="bfill", inplace=True)

        return mask

    def _flag(self, test: bool = False) -> None:
        for var in TEMPERATURE_VARIABLES:
            if test:
                flag = self.test_flag
            else:
                try:
                    flag = self._get_flag(var)
                except ValueError:
                    continue

            self.add_mask(
                var,
                flag,
                "in cloud",
                (
                    "The aircraft is indicated as being in cloud, according to "
                    "the clear air mask derived from the Nevzorov power variance."
                ),
            )

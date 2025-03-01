"""
This module provides flagging modules for the nephelometer, providing data quality
information which can not be inferred during processing of the nephelometer data.
"""

import numpy as np
import pandas as pd

from ppodd.flags.base import FlaggingBase

MASKED = 1
UNMASKED = 0
NEPH_VARIABLES = (
    "TSC_BLUU",
    "TSC_GRNU",
    "TSC_REDU",
    "BSC_BLUU",
    "BSC_GRNU",
    "BSC_REDU",
)


class NephelometerCloudFlag(FlaggingBase):
    """
    This class adds a flag to nephelometer scattering data if the
    aircraft is in cloud, as indicated by NV_CLEAR_AIR_MASK, derived from the
    Nevzorov probe.
    """

    inputs = ["NV_CLEAR_AIR_MASK"]
    flagged = list(NEPH_VARIABLES)

    def _get_flag(self, var: str) -> pd.Series:

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

        for var in NEPH_VARIABLES:
            if test:
                flag: pd.Series | np.ndarray = self.test_flag
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

"""
This module provides flagging modules for the Rosemount temperature probes,
providing data quality information which can not be inferred during processing
of the temperatures.
"""

import numpy as np
import pandas as pd

from ppodd.flags.base import FlaggingBase

TEMPERATURE_THRESHOLD = 1.
TEMPERATURE_VARIABLES = ('TAT_DI_R', 'TAT_ND_R')
MASKED = 1
UNMASKED = 0

class RosemountTempDeltaFlag(FlaggingBase):
    """
    The temperature delta flag adds a flag to the rosemount temperatures if the
    two temperatures disagree by more than a given absolute value. This is
    given by the module constant TEMPERATURE_THRESHOLD.
    """

    inputs = list(TEMPERATURE_VARIABLES)

    def flag(self):
        """
        Entry point for the flagging module.
        """

        # Get the absolute difference between the deiced and non-deiced
        # temperatures
        _diff = np.abs(
            self.dataset['TAT_DI_R'].array - self.dataset['TAT_ND_R'].array
        )

        # Create a mask for the flag
        mask = pd.Series(
            np.zeros_like(self.dataset['TAT_DI_R'].array),
            index=self.dataset['TAT_DI_R'].index
        )
        mask.loc[_diff > TEMPERATURE_THRESHOLD] = MASKED

        # For each of the temperature add the threshold mask
        for var in TEMPERATURE_VARIABLES:
            self.dataset[var].flag.add_mask(
                mask, 'discrepancy threshold exceeded'
            )


class RosemountTempCloudFlag(FlaggingBase):
    """
    The cloud flags adds a flag to all temperature variables if the aircraft is
    in cloud, as indicated by NV_CLEAR_AIR_MASK, derived from the Nevzerov
    probe.
    """

    inputs = ['NV_CLEAR_AIR_MASK']

    def flag(self):
        """
        Entry point for the flagging module.
        """

        for var in TEMPERATURE_VARIABLES:
            try:
                index = self.dataset[var].index
            except (KeyError, AttributeError):
                continue

            clear_air_series = self.dataset['NV_CLEAR_AIR_MASK']()
            mask = clear_air_series == UNMASKED
            nv_start = clear_air_series.index[0]
            nv_end = clear_air_series.index[-1]
            mask = mask.reindex(index)
            mask.loc[(mask.index <= nv_start) | (mask.index >= nv_end)] = 0
            mask.fillna(method='ffill', inplace=True)
            mask.fillna(method='bfill', inplace=True)

            self.dataset[var].flag.add_mask(
                mask, 'in cloud'
            )

"""
This module provides flagging modules for the Rosemount temperature probes,
providing data quality information which can not be inferred during processing
of the temperatures.
"""

import numpy as np
import pandas as pd

from ppodd.flags.base import FlaggingBase

MASKED = 1
UNMASKED = 0
WVSS2_VARIABLES = ('WVSS2F_VMR_U', 'WVSS2R_VMR_U')

class WVSS2CloudFlag(FlaggingBase):
    """
    The cloud flags adds a flag to all temperature variables if the aircraft is
    in cloud, as indicated by NV_CLEAR_AIR_MASK, derived from the Nevzerov
    probe.
    """

    inputs = ['NV_CLEAR_AIR_MASK']

    def _flag(self):
        """
        Entry point for the flagging module.
        """

        for var in WVSS2_VARIABLES:
            if var not in self.dataset.variables:
                continue

            self.flagged.append(var)

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

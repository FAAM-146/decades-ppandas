"""
This module provides flagging modules for the Rosemount temperature probes,
providing data quality information which can not be inferred during processing
of the temperatures.
"""

import datetime

import numpy as np
import pandas as pd

from ppodd.flags.base import FlaggingBase

CPC_VARIABLES = ('CPC_CNTS',)
MASKED = 1
UNMASKED = 0


class CPCCloudFlag(FlaggingBase):
    """
    This class adds a flag to all rosemount temperature variables if the aircraft
    is in cloud, as indicated by NV_CLEAR_AIR_MASK, derived from the Nevzorov
    probe. Before application, the NV_CLEAR_AIR_MASK is shifted by CPC_LAG_SECS
    seconds to account for the time lag between the Nevzorov (considered the
    reference instrument) and the CPC.
    """

    inputs = ['NV_CLEAR_AIR_MASK']
    flagged = list(CPC_VARIABLES)

    def _get_flag(self, var):
        """
        Entry point for the flagging module.
        """

        try:
            index = self.dataset[var].index
        except (KeyError, AttributeError):
            raise ValueError('Unable to get variable index')

        clear_air_series = self.dataset['NV_CLEAR_AIR_MASK']()
        clear_air_series.index += datetime.timedelta(seconds=self.dataset['CPC_LAG_SECS'])
        clear_air_series = clear_air_series.reindex(index).ffill().bfill()
        mask = clear_air_series == UNMASKED
        nv_start = clear_air_series.index[0]
        nv_end = clear_air_series.index[-1]
        mask = mask.reindex(index)
        mask.loc[(mask.index <= nv_start) | (mask.index >= nv_end)] = 0
        mask.fillna(method='ffill', inplace=True)
        mask.fillna(method='bfill', inplace=True)

        return mask

    def _flag(self, test=False):
        
        for var in CPC_VARIABLES:
            if test:
                flag = self.test_flag
            else:
                try:
                    flag = self._get_flag(var)
                except ValueError:
                    continue

            self.add_mask(
                var, flag, 'in cloud',
                ('The aircraft is indicated as being in cloud, according to '
                 'the clear air mask derived from the Nevzorov power variance.')
            )

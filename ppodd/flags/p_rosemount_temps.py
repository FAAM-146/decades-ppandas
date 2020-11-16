import numpy as np
import pandas as pd

from .base import FlaggingBase


class RosemountTempDeltaFlag(FlaggingBase):
    inputs = ['TAT_DI_R', 'TAT_ND_R']

    def flag(self):
        _diff = np.abs(
            self.dataset['TAT_DI_R'].array - self.dataset['TAT_ND_R'].array
        )

        mask = pd.Series(
            np.zeros_like(self.dataset['TAT_DI_R'].array),
            index=self.dataset['TAT_DI_R'].index
        )
        mask.loc[_diff > 1.] = 1

        for var in ('TAT_DI_R', 'TAT_ND_R'):
            self.dataset[var].flag.add_mask(
                mask, 'discrepancy threshold exceeded'
            )


class RosemountTempCloudFlag(FlaggingBase):
    inputs = ['NV_CLEAR_AIR_MASK']

    def flag(self):
        for var in ('TAT_DI_R', 'TAT_ND_R'):
            try:
                index = self.dataset[var].index
            except (KeyError, AttributeError):
                continue

            clear_air_series = self.dataset['NV_CLEAR_AIR_MASK']()
            mask = clear_air_series == 0
            nv_start = clear_air_series.index[0]
            nv_end = clear_air_series.index[-1]
            mask = mask.reindex(index)
            mask.loc[(mask.index <= nv_start) | (mask.index >= nv_end)] = 0
            mask.fillna(method='ffill', inplace=True)
            mask.fillna(method='bfill', inplace=True)

            self.dataset[var].flag.add_mask(
                mask, 'in cloud'
            )

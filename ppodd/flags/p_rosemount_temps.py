import numpy as np
import pandas as pd

from .base import FlaggingBase

class RosemountTempFlag(FlaggingBase):
    inputs = ['TAT_DI_R', 'TAT_ND_R']

    def flag(self):
        _diff = np.abs(
            self.dataset['TAT_DI_R'].data - self.dataset['TAT_ND_R'].data
        )

        mask = pd.Series(
            np.zeros_like(self.dataset['TAT_DI_R'].data),
            index=self.dataset['TAT_DI_R'].index
        )
        mask.loc[_diff > 1.] = 1

        for var in ('TAT_DI_R', 'TAT_ND_R'):
            self.dataset[var].flag.add_mask(
                mask, 'discrepancy threshold exceeded'
            )

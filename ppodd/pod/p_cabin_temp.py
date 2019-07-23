import numpy as np

from .base import PPBase
from ..decades import DecadesVariable


class CabinTemp(PPBase):

    inputs = [
        'CALCABT',
        'CORCON_cabin_t',
    ]

    def declare_outputs(self):
        self.declare(
            'CAB_TEMP',
            units='degC',
            frequency=1,
            number=660,
            long_name='Cabin temperature at the core consoles'
        )

    def process(self):
        """
        Processing entry point
        """
        self.get_dataframe()
        d = self.d

        _cals = self.dataset['CALCABT'][::-1]

        # Temperature is just a polynomial cal from CORCON_cabin_t
        d['CAB_TEMP'] = np.polyval(_cals, d.CORCON_cabin_t)
        d['CAB_TEMP_FLAG'] = 2

        self.add_output(DecadesVariable(d[['CAB_TEMP', 'CAB_TEMP_FLAG']]))

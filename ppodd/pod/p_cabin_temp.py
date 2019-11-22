import numpy as np

from .base import PPBase
from ..decades import DecadesVariable
from ..decades import flags


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
        d['CAB_TEMP_FLAG'] = 1

        temp = DecadesVariable(d['CAB_TEMP'], name='CAB_TEMP')

        temp.flag.add_meaning(0, flags.DATA_GOOD)
        temp.flag.add_meaning(1, 'sensor uncalibrated')
        temp.flag.add_meaning(2, flags.DATA_MISSING)

        temp.flag.add_flag(d['CAB_TEMP_FLAG'])

        self.add_output(temp)

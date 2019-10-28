import numpy as np

from ..decades import DecadesVariable
from .base import PPBase


class ElectricFieldJci140(PPBase):

    inputs = ['PRTAFT_jci140_signal']

    def declare_outputs(self):
        self.declare(
            'EXX_JCI',
            units='adc_counts',
            frequency=1,
            standard_name=None,
            long_name=('Raw data from the Fwd Core Console JCI static '
                       'monitor, static signal')
        )

    def process(self):
        self.get_dataframe()
        df = self.d.asfreq('1S')

        output = DecadesVariable(df['PRTAFT_jci140_signal'], name='EXX_JCI')

        output.flag.add_meaning(0, 'not flagged')
        output.flag.add_meaning(1, 'uncalibrated counts')
        output.flag.add_flag(np.ones((len(output),)))

        self.add_output(output)

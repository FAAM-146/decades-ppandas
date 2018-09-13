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
        df['EXX_JCI'] = df['PRTAFT_jci140_signal']
        df['EXX_JCI_FLAG'] = 3

        self.add_output(DecadesVariable(df[['EXX_JCI', 'EXX_JCI_FLAG']]))

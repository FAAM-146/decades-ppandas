from ..decades import DecadesVariable
from .base import PPBase


class ElectricFieldJci140(PPBase):

    def inputs(self):
        return ['PRTAFT_JCI140_SIGNAL']

    def declare_outputs(self):
        self.declare(
            'EXX_JCI',
            units='adc_counts',
            frequency=1,
            long_name='Raw data from the Fwd Core Console JCI static monitor, static signal',
            standard_name=None
        )

    def process(self):

        df = self.get_dataframe().asfreq('1S')
        df['FLAG_EXX_JCI'] = 3

        self.add_output(
            DecadesVariable(
                df['PRTAFT_JCI140_SIGNAL'],
                name='EXX_JCI',
            ),
            flag=df['FLAG_EXX_JCI']
        )

        self.finalize()

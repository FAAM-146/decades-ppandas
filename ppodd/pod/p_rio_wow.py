from ..decades import DecadesVariable
from .base import PPBase


class RioWeightOnWheels(PPBase):

    def inputs(self):
        return ['PRTAFT_WOW_FLAG']

    def declare_outputs(self):
        self.declare(
            'WOW_IND',
            units='-',
            frequency=1,
            long_name='Weight on wheels indicator',
            standard_name=None
        )

    def process(self):

        wow_data = self.dataset['PRTAFT_WOW_FLAG'].asfreq('1S')

        self.add_output(
            DecadesVariable(
                wow_data,
                name='WOW_IND',
            )
        )

        self.finalize()

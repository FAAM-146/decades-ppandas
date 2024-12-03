"""
Provides a processing module for the static mill on the core console.
"""
# pylint: disable=invalid-name
import numpy as np

from ..decades import DecadesVariable
from .base import PPBase, register_pp
from .shortcuts import _o


@register_pp('core')
class ElectricFieldJci140(PPBase):
    r"""
    This module reports the raw counts from the JCI static monitor on the
    core console.
    """

    inputs = ['PRTAFT_jci140_signal']

    @staticmethod
    def test():
        """
        Return dummy input data for test usage.
        """
        return {
            'PRTAFT_jci140_signal': ('data', 50 * _o(100), 1)
        }

    def declare_outputs(self):
        """
        Declare the outputs produced by this module.
        """
        self.declare(
            'EXX_JCI',
            units='1',
            frequency=1,
            long_name=('Raw data from the Fwd Core Console JCI static '
                       'monitor, static signal')
        )

    def process(self):
        """
        Processing entry hook.
        """
        self.get_dataframe()
        df = self.d.asfreq('1s')

        output = DecadesVariable(df['PRTAFT_jci140_signal'], name='EXX_JCI')

        output.flag.add_meaning(
            1, 'uncalibrated counts',
            ('Indicates this data is raw, uncalibrated counts from the DLU. '
             'Use with caution.')
        )

        output.flag.add_flag(np.ones((len(output),)))

        self.add_output(output)

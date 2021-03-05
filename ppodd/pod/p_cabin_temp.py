"""
This module provides a postprocessing module which provides an in-cabin
temperature measurement at the core console. See class docstring for more info.
"""
# pylint: disable=invalid-name
import numpy as np

from .base import PPBase
from .shortcuts import _c, _l
from ..decades import DecadesVariable
from ..decades import flags


class CabinTemp(PPBase):
    r"""
    Derives cabin temperature from a sensor located on the right of the core
    console. A polynomial fit, with coefficients provided in the constants
    variable ``CALCABT``, converts DLU counts :math:`\rightarrow` raw
    :math:`\rightarrow` temperature.
    """


    inputs = [
        'CALCABT',
        'CORCON_cabin_t',
    ]

    @staticmethod
    def test():
        """
        Return dummy input data for testing.
        """
        return {
            'CALCABT': ('const', [-263, 1.5e-4]),
            'CORCON_cabin_t': (
                'data', _c([_l(180e4, 185e4, 50), _l(185e4, 180e4, 50)])
            )
        }

    def declare_outputs(self):
        """
        Declare the outputs produced by this module.
        """
        self.declare(
            'CAB_TEMP',
            units='degC',
            frequency=1,
            long_name='Cabin temperature at the core consoles',
            comment=('Should be considered a qualitative measure only, due '
                     'to lack of calibration and proximity to the core '
                     'console')
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

        temp.flag.add_meaning(0, flags.DATA_GOOD, 'Data are considered valid')
        temp.flag.add_meaning(
            1, 'sensor uncalibrated', ('Indicates that the sensor is considered '
                                      'to be poorly calibrated. Temperatures are '
                                      'to be considered qualitative.')
        )
        temp.flag.add_meaning(
            2, flags.DATA_MISSING, 'Data are expected, but are missing'
        )

        temp.flag.add_flag(d['CAB_TEMP_FLAG'])

        self.add_output(temp)

import numpy as np
import pandas as pd

from ..decades import DecadesVariable
from .base import PPBase, register_pp
from .shortcuts import _c, _o, _z


@register_pp('core')
class SignalRegister(PPBase):
    """
    The DECADES signal register is a packed representation of some boolean
    state variables, which is used internally in DECADES. Here we reproduce
    part of the register from flag variables reported in the TCP data, for use
    in other processing modules. No outputs defined here are expected to ever
    be written to file.
    """

    inputs = [
        'PRTAFT_heimann_calib_flag',
        'PRTAFT_deiced_temp_flag'
    ]

    @staticmethod
    def test():
        return {
            'PRTAFT_heimann_calib_flag': ('data', _c([_o(10), _z(90)]), 1),
            'PRTAFT_deiced_temp_flag': ('data', _c([_z(30), _o(30), _z(40)]), 1)
        }

    def declare_outputs(self):

        self.declare(
            'SREG',
            frequency=2,
            long_name='Signal Register',
            units=None,
            write=False
        )

    def process(self):
        """
        Processing entry point.
        """

        # Both of the flag Series used here are recorded at 1Hz, however the
        # legacy code produced the signal register at 2Hz, so we'll do the same
        # here.
        start = self.dataset[self.inputs[0]].index[0].round('1s')
        end = self.dataset[self.inputs[0]].index[-1].round('1s')

        index = pd.date_range(start, end, freq='500ms')
        self.get_dataframe(method='onto', index=index)

        heimann = self.d.PRTAFT_heimann_calib_flag.fillna(0).astype(int)
        deiced = self.d.PRTAFT_deiced_temp_flag.fillna(0).astype(int)

        # Left shift the DI heater flag by 5, zero pad to the right.
        # e.g. 101 -> 10100000
        deiced = np.left_shift(deiced, 5)

        # The output register is bitwise-or between the heimann flag and
        # shifted DI heater flag.
        sreg = DecadesVariable(heimann | deiced, name='SREG')
        self.add_output(sreg)

import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..utils.conversions import feet_to_metres
from .base import PPBase
from .shortcuts import _l

RADALT_MIN = 10
RADALT_MAX = 8100


class RadAlt(PPBase):
    r"""
    Calculate the radar altitude, in metres. The radar altitude, from radalt2,
    is read from the aircraft ARINC-429 data bus at the rear core console, at
    a frequency of 2 Hz. The signal is received as a 16 bit signed int, with a
    least significant bit resolution of 0.25 ft, giving a max valid value of
    :math:`(((2^{16} / 2) - 1) / 4)`.

    Data are flagged when outside a slightly narrower range than the range of
    the 16 bit signal.
    """

    inputs = [
        'PRTAFT_rad_alt'    # Radar altitude (dlu)
    ]

    @staticmethod
    def test():
        return {
            'PRTAFT_rad_alt': ('data', _l(0, 32000, 100))
        }

    def declare_outputs(self):
        self.declare(
            'HGT_RADR',
            units='m',
            frequency=2,
            long_name='Radar height from the aircraft radar altimeter',
            standard_name='height'
        )

    def flag(self):
        d = self.d
        d['RANGE_FLAG'] = 0

        d.loc[d['HGT_RADR'] >= RADALT_MAX, 'RANGE_FLAG'] = 3
        d.loc[d['HGT_RADR'] <= RADALT_MIN, 'RANGE_FLAG'] = 3

    def process(self):
        self.get_dataframe()
        d = self.d

        d['HGT_RADR'] = feet_to_metres(d['PRTAFT_rad_alt'] * 0.25)

        self.flag()

        dv = DecadesVariable(d['HGT_RADR'], flag=DecadesBitmaskFlag)
        dv.flag.add_mask(
            d['RANGE_FLAG'], flags.OUT_RANGE,
            f'RadAlt reading outside valid range ({RADALT_MIN}, {RADALT_MAX})'
        )
        self.add_output(dv)

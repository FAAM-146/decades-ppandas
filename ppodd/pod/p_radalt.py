"""
Provides a processing module for the radar altimeter (nominally radalt2)
receivec on the PRTAFR dlu, from the arinc 429 bus. See class docstring for
more info.
"""
# pylint: disable=invalid-name

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..utils.conversions import feet_to_metres
from .base import PPBase, register_pp
from .shortcuts import _l, _z

RADALT_MIN = 10
RADALT_MAX = feet_to_metres(5000)


@register_pp('core')
class RadAlt(PPBase):
    r"""
    Calculate the radar altitude, in metres. The radar altitude, from radalt2,
    is read from the aircraft ARINC-429 data bus at the rear core console, and
    recodred at a frequency of 2 Hz. The signal is received as a 16 bit signed
    integer, with a least significant bit resolution of 0.25 ft, giving a max
    valid value of :math:`(((2^{16} / 2) - 1) / 4)`.

    Data are flagged when below 10 feet or above 5000 feet, which is the
    nominal maximum range of the instrument, though altitude is reported up to
    8000 ft.
    """

    inputs = [
        'PRTAFT_rad_alt',    # Radar altitude (dlu)
        'WOW_IND'            # Weight on wheels indicator (derived)
    ]

    @staticmethod
    def test():
        """
        Provide dummy input data for testing.
        """
        return {
            'PRTAFT_rad_alt': ('data', _l(0, 32000, 100), 2),
            'WOW_IND': ('data', _z(100), 1)
        }

    def declare_outputs(self):
        """
        Declare outputs created by this module.
        """
        self.declare(
            'HGT_RADR',
            units='m',
            frequency=2,
            long_name='Radar height from the aircraft radar altimeter',
            standard_name='height',
            instrument_manufacturer='Thales',
            instrument_model='AVH16 Radar Altimeter'
        )

    def flag(self):
        """
        Create flagging info.
        """
        d = self.d
        d['RANGE_FLAG'] = 0
        d['WOW_FLAG'] = 0

        d.loc[d['HGT_RADR'] >= RADALT_MAX, 'RANGE_FLAG'] = 3
        d.loc[d['HGT_RADR'] <= RADALT_MIN, 'RANGE_FLAG'] = 3

        wow = d['WOW_IND'].fillna(method='bfill').fillna(method='ffill')
        d.loc[wow == 0, 'WOW_FLAG'] = 1

    def process(self):
        """
        Processing entry hook.
        """
        self.get_dataframe()
        d = self.d

        d['HGT_RADR'] = feet_to_metres(d['PRTAFT_rad_alt'] * 0.25)

        self.flag()

        dv = DecadesVariable(d['HGT_RADR'], flag=DecadesBitmaskFlag)
        dv.flag.add_mask(
            d['RANGE_FLAG'], flags.OUT_RANGE,
            (f'RadAlt reading outside valid range '
             f'({RADALT_MIN:0.2f}, {RADALT_MAX:0.2f})')
        )

        dv.flag.add_mask(
            d['WOW_IND'], flags.WOW,
            ('The aircraft is on the ground, as indicated by the weight '
             'on wheels flag.')
        )

        self.add_output(dv)

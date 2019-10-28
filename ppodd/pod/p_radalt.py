from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..utils.conversions import feet_to_metres
from ..utils.constants import RADALT_MIN, RADALT_MAX
from .base import PPBase

class RadAlt(PPBase):

    inputs = [
        'PRTAFT_rad_alt'    # Radar altitude (dlu)
    ]

    def declare_outputs(self):
        self.declare(
            'HGT_RADR',
            units='m',
            frequency=2,
            number=575,
            long_name='Radar height from the aircraft radar altimeter',
            standard_name='height'
        )

    def flag(self):
        d = self.d
        d['RANGE_FLAG'] = 0

        d.loc[d['HGT_RADR'] > RADALT_MAX, 'RANGE_FLAG'] = 3
        d.loc[d['HGT_RADR'] < RADALT_MIN, 'RANGE_FLAG'] = 3

    def process(self):
        self.get_dataframe()
        d = self.d

        d['HGT_RADR'] = feet_to_metres(d['PRTAFT_rad_alt'] * 0.25)

        self.flag()

        dv = DecadesVariable(d['HGT_RADR'], flag=DecadesBitmaskFlag)
        dv.flag.add_mask(d['RANGE_FLAG'], flags.OUT_RANGE)
        self.add_output(dv)

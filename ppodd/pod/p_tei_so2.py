from datetime import date, timedelta

import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..utils import flagged_avg
from .base import PPBase, register_pp
from .shortcuts import _c, _o, _z

CAL_FLUSH_START = 3
CAL_FLUSH_END = 5


@register_pp('core')
class TecoSO2(PPBase):
    r"""
    Calculate SO\ :math:`_2` concentration from the TECO 43 instrument. The
    instrument reports a nominal concentration and sensitivity, valve states V6
    (indicating cylinder air) and V8 (indicating cabin air), and a status flag.

    Zeros are taken whenever the instrument in sampling cylinder or cabin air,
    and interploated back to 1 Hz, assuming a linear drift of the offsets
    between zeros. SO\ :math:`_2` concentration is then given by

    .. math::
        \left[\text{SO}_2\right] = \frac{\left[\text{SO}_{2|\text{INS}}\right] - Z}{S},

    where :math:`\left[\text{SO}_{2|\text{INS}}\right]` is the concentration reported
    by the instrument, :math:`Z` is the zero obtained from sampling cylinder or
    cabin air, and :math:`S` is the sensitivity reported by the instrument.

    Flagging is based on valve states and the instrument status flag.
    """

    DEPRECIATED_AFTER = date(2021, 9, 1)

    inputs = [
        'CHTSOO_conc',
        'CHTSOO_flags',
        'CHTSOO_V6',
        'CHTSOO_V8',
        'CHTSOO_sensitivity',
        'WOW_IND'
    ]

    @staticmethod
    def test():
        #TODO: These values require more thought
        return {
            'CHTSOO_conc': ('data', _o(100), 1),
            'CHTSOO_flags': ('data', [b'cc0000'] * 100, 1),
            'CHTSOO_V6': ('data', _c([_z(40), _o(20), _z(40)]), 1),
            'CHTSOO_V8': ('data', _z(100), 1),
            'CHTSOO_sensitivity': ('data', _o(100), 1),
            'WOW_IND': ('data', _c([_o(10), _z(80), _o(10)]), 1)
        }

    def declare_outputs(self):
        self.declare(
            'SO2_TECO',
            units='ppb',
            frequency=1,
            long_name=('Mole fraction of Sulphur Dioxide in air from TECO 43 '
                       'instrument'),
            standard_name='mole_fraction_of_sulfur_dioxide_in_air',
            instrument_manufacturer='Thermo Fisher Scientific, Inc.',
            instrument_model='43i TLE pulsed fluorescence SO2 spectrometer'
        )

    def flag(self):
        """
        Create a flag for the SO2 output.

        Flag info:
            data good --> 0
            aircraft on ground --> 1
            before first zero --> 2
            instrument in calibration --> 3
            instrument in alarm --> 3
        """
        d = self.d

        d['WOW_FLAG'] = (d.WOW_IND == 1)
        d['ZERO_FLAG'] = 0
        d['CALIB_FLAG'] = 0
        d['ALARM_FLAG'] = 0

        # Before first zero is flagged
        first_zero = d.loc[d.zero_flag == 1].index[0]
        d.loc[d.index < first_zero, 'ZERO_FLAG'] = 1

        # Instrument in calibration is flagged
        _groups = (d.zero_flag != d.zero_flag.shift()).cumsum()
        _groups[d.zero_flag == 0] = np.nan
        _groups.dropna(inplace=True)
        groups = d.groupby(_groups)
        for group in groups:
            start = group[1].index[0]
            end = group[1].index[-1] + timedelta(seconds=CAL_FLUSH_END)
            d.loc[start:end, 'CALIB_FLAG'] = 1

        # Instrument in alarm is flagged as 3
        alarm = self.d.CHTSOO_flags.apply(lambda x: str(x)[-4:-1] != '000')
        d.loc[alarm, 'ALARM_FLAG'] = 3

    def process(self):
        """
        Processing entry point.
        """

        # The SO2 data sometimes produces duplicates in the TCP data stream,
        # ensure that the inputs are unique.
        # This is now done in reading...
        # for var in self.inputs:
        #     if 'WOW' in var:
        #         continue
        #     _df = self.dataset[var]._df
        #     if _df.index.size != _df.index.unique().size:
        #         self.dataset[var]._df = _df.groupby(_df.index).agg(
        #             {var: 'first'}
        #         )

        self.get_dataframe()

        # The instrument is in calibration when one of the valves V6 (cylinder
        # air) or V8 (cabin air) is open.
        self.d['zero_flag'] = (self.d.CHTSOO_V6 == 1) | self.d.CHTSOO_V8

        # Average across the zeros, linearly interpolate back to 1 Hz...
        flagged_avg(self.d, 'zero_flag', 'CHTSOO_conc', out_name='zero',
                    interp=True, skip_start=CAL_FLUSH_START,
                    skip_end=CAL_FLUSH_END)

        # ...and backfill times before the first zero
        self.d['zero'].bfill(inplace=True)

        # Calculate scaled SO2
        sensitivity = self.d.CHTSOO_sensitivity
        self.d['SO2_TECO'] = (self.d.CHTSOO_conc - self.d.zero) / sensitivity

        # Build QA Flag for the SO2
        self.flag()

        SO2 = DecadesVariable(self.d['SO2_TECO'], flag=DecadesBitmaskFlag)

        SO2.flag.add_mask(
            self.d.WOW_FLAG, flags.WOW, 'Aircraft is on the ground'
        )

        SO2.flag.add_mask(
            self.d.ZERO_FLAG, 'before first zero',
            'The instrument has not yet sampled cylinder or cabin air, '
            'the zero is invalid'
        )

        SO2.flag.add_mask(
            self.d.CALIB_FLAG, 'in zero',
            'The instrument is currently sampling cylinder or cabin air for '
            'a zero reading'
        )

        SO2.flag.add_mask(
            self.d.ALARM_FLAG, 'in alarm',
            'The instrument status flag is currently indicating an alarm '
            'state'
        )

        self.add_output(SO2)

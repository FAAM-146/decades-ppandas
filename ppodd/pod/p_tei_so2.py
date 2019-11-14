from datetime import timedelta

import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..utils import flagged_avg
from .base import PPBase

CAL_FLUSH_START = 3
CAL_FLUSH_END = 5


class TecoSO2(PPBase):

    inputs = [
        'CHTSOO_conc',
        'CHTSOO_flags',
        'CHTSOO_V6',
        'CHTSOO_V8',
        'CHTSOO_sensitivity',
        'WOW_IND'
    ]

    def declare_outputs(self):
        self.declare(
            'SO2_TECO',
            units='ppb',
            frequency=1,
            number=740,
            long_name=('Mole fraction of Sulphur Dioxide in air from TECO 43 '
                       'instrument'),
            standard_name='mole_fraction_of_sulfur_dioxide_in_air'
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

        d['WOW_FLAG'] = (d.WOW == 1)
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
        for var in self.inputs:
            if 'WOW' in var:
                continue
            _df = self.dataset[var]._df
            if _df.index.size != _df.index.unique().size:
                self.dataset[var]._df = _df.groupby(_df.index).agg(
                    {var: 'first'}
                )

        self.get_dataframe()

        # The instrument is in calibration when one of the valves V6 (cylinder
        # air) or V8 (cabin air) is open.
        self.d['zero_flag'] = (self.d.CHTSOO_V6 == 1) | self.d.CHTSOO_V8

        # Average across the zeros, linearly interpolate back to 1 Hz...
        flagged_avg(self.d, 'zero_flag', 'CHTSOO_conc', out_name='zero',
                    interp=True, skip_start=CAL_FLUSH_START,
                    skip_end=CAL_FLUSH_END)

        # ...and backfill times before the first zero
        self.d['zero'].fillna(method='bfill', inplace=True)

        # Calculate scaled SO2
        sensitivity = self.d.CHTSOO_sensitivity
        self.d['SO2_TECO'] = (self.d.CHTSOO_conc - self.d.zero) / sensitivity

        # Build QA Flag for the SO2
        self.flag()

        SO2 = DecadesVariable(self.d['SO2_TECO'], flag=DecadesBitmaskFlag)
        SO2.flag.add_mask(d.WOW_FLAG, flags.WOW)
        SO2.flag.add_mask(d.ZERO_FLAG, 'before first zero')
        SO2.flag.add_mask(d.CALIB_FLAG, flags.CALIBRATION)
        SO2.flag.add_mask(d.ALARM_FLAG, 'in alarm')

        self.add_output(SO2)

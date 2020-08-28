import datetime
import numpy as np
import pandas as pd

from ..decades import DecadesVariable, DecadesBitmaskFlag
from .base import PPBase
from .shortcuts import _c, _o, _z
from ..utils import flagged_avg

INIT_SKIP = 100         # Number of datapoints to skip at the start
SENS_CUTOFF = 0         # Sensitivity vals at or below considered bad
CAL_FLUSH_TIME = 3      # Time for system to flush after a cal
CO_VALID_MIN = -10      # Flag if CO below this value


class AL52CO(PPBase):
    r"""
    Process CO concentration from the AL5002 instrument. The instrument provides
    counts, concentration, sensitivity, and zero. However, the sensitivity and
    zero step change after calibrations, which propogate through to the CO
    concentration provided by the instrument. To avoid post-calibration step
    changes, we assume that the sensitivity and zero drift linearly between
    calibrations, and interpolate across the step changes to produce smoother
    sensitivity and zero-offset curves. The CO concentration is then given by
    $$
    \text{CO} = \frac{c - z_i}{S_i},
    $$
    where $c$ is the count reported by ther instrument, $z_i$ is the linearly
    interpolated zero and $S_i$ is the linearly interpolated sensitivity.
    """

    inputs = [
        'AL52CO_sens',
        'AL52CO_zero',
        'AL52CO_calpress',
        'AL52CO_cal_status',
        'AL52CO_counts',
        'WOW_IND'
    ]

    @staticmethod
    def test():
        return {
            'AL52CO_sens': (
                'data',  _c([45 * _o(25), 46 * _o(25), 47 * _o(25), 48 *
                             _o(25)] *2)
            ),
            'AL52CO_zero': (
                'data', _c(
                    [45 * _o(25), 46 * _o(25), 47 * _o(25), 48 * _o(25)] * 2
                ) * 1000
            ),
            'AL52CO_counts': ('data', 38000 * _o(200)),
            'AL52CO_calpress': ('data', _c([1.5 * _o(20), 3*_o(5)] * 8)),
            'AL52CO_cal_status': ('data', _c([_z(20), _o(5)] * 7 + [_z(25)])),
            'WOW_IND': ('data', _c([_o(110), _z(80), _o(10)]))
        }

    def declare_outputs(self):
        self.declare(
            'CO_AERO',
            units='ppb',
            frequency=1,
            long_name=('Mole fraction of Carbon Monoxide in air from the AERO '
                       'AL5002 instrument'),
            standard_name='mole_fraction_of_carbon_monoxide_in_air'
        )

    def flag(self):
        """
        Create a flag for the CO output.

        Returns:
            fdf: a pd.DataFrame with 6 boolean columns, each giving the state
                 of a mask flag.
        """

        WOW_FLAG = (
            'aircraft on ground',
            'The aircraft is on the ground, as indicated by WOW\_IND.'
        )

        CO_RANGE_FLAG = (
            'co out of range',
            'The derived CO concentration is considered out of valid range.'
        )

        IN_CAL_FLAG = (
            'in calibration',
            ('The instrument is currently, or has recently been, in '
             'calibration. Data should be disregarded.')
        )

        NO_CAL_FLAG = (
            'no calibration',
            ('No calibration has yet been performed. Data should be '
             'disregarded.')
        )

        ZERO_COUNTS_FLAG = (
            'counts zero',
            ('The instrument is reporting zero counts. This is most likely '
             'erroneous.')
        )

        d = self.d
        fdf = pd.DataFrame(index=self.d.index)

        fdf[WOW_FLAG] = 0
        fdf[CO_RANGE_FLAG] = 0
        fdf[IN_CAL_FLAG] = 0
        fdf[NO_CAL_FLAG] = 0
        fdf[ZERO_COUNTS_FLAG] = 0

        # In the processing, we nan out the start of the data, we need to
        # replace this so that the .shift()).cumsum() method works.
        d['AL52CO_cal_status'].fillna(method='bfill', inplace=True)
        d['AL52CO_cal_status'].fillna(method='ffill', inplace=True)

        # Flag when the aircraft is on the ground
        fdf.loc[d['WOW_IND'] != 0, WOW_FLAG] = 1

        # Out of range flagging
        fdf.loc[d['CO_AERO'] < CO_VALID_MIN, CO_RANGE_FLAG] = 1

        # We want to flag not only the times when the instrument is in
        # calibration, but also a few seconds afterwards, while the calibration
        # gas is flushed.
        _groups = (
            d['AL52CO_cal_status'] != d['AL52CO_cal_status'].shift()
        ).cumsum()

        _groups[d['AL52CO_cal_status'] < 1] = np.nan
        _groups.dropna(inplace=True)
        groups = d.groupby(_groups)

        for group in groups:
            start = group[1].index[0]
            end = group[1].index[-1] + datetime.timedelta(seconds=CAL_FLUSH_TIME)
            fdf.loc[start:end, IN_CAL_FLAG] = 1

        # Flag when counts are identically zero
        fdf.loc[d['AL52CO_counts'] == 0, ZERO_COUNTS_FLAG] = 1

        # Flag before the first calibration
        try:
            first_cal_start = d.loc[d['AL52CO_cal_status'] > 0].index[0]
        except IndexError:
            first_cal_start = d.index[-1]
        fdf.loc[d.index <= first_cal_start, NO_CAL_FLAG] = 1

        return fdf

    def process(self):
        """
        Entry point for the postprocessing module.
        """

        # Reduce the AL52 input from ca. 3 vals per sec to 1 Hz
        for var in self.inputs:
            if 'AL52CO' not in var:
                continue
            _df = self.dataset[var]._df
            if _df.index.size != _df.index.unique().size:
                self.dataset[var]._df = _df.groupby(_df.index).agg(
                    {var: 'first'}
                )

        self.get_dataframe()
        d = self.d

        # Skip a chunk of data at the start, where Weird Things(TM) sometimes
        # happen
        d.iloc[:INIT_SKIP] = np.nan

        # Mask erroneous values in the sensitivity
        d.loc[d['AL52CO_sens'] <= SENS_CUTOFF, 'AL52CO_sens'] = np.nan
        d['AL52CO_sens'].fillna(method='bfill', inplace=True)
        d['AL52CO_sens'].fillna(method='ffill', inplace=True)

        # Mask erroneous values in the zero
        d.loc[d['AL52CO_zero'] == 0, 'AL52CO_zero'] = np.nan
        d['AL52CO_zero'].fillna(method='bfill', inplace=True)

        # Build a flag indicating where the sensitivity and zero have changed
        # after a calibration, with a 2 sec safety buffer
        d['CAL_FLAG'] = d.AL52CO_sens.diff() != 0
        indicies = np.where(d.CAL_FLAG != 0)[0]
        indicies_p2 = indicies + 2
        d.loc[d.index[indicies_p2], 'CAL_FLAG'] = 1
        d.loc[d.index[indicies], 'CAL_FLAG'] = 0

        # Interpolate the zero
        flagged_avg(d, 'CAL_FLAG', 'AL52CO_zero', out_name='ZERO', interp=True)
        d.ZERO.fillna(method='bfill', inplace=True)

        # Interpolate the sensitivity
        flagged_avg(d, 'CAL_FLAG', 'AL52CO_sens', out_name='SENS', interp=True)
        d.SENS.fillna(method='bfill', inplace=True)

        # Calculate concentration using interpolated sens & zero
        d['CO_AERO'] = (d.AL52CO_counts - d.ZERO) / d.SENS

        # Flag build the qa flag dataframe
        flag_df = self.flag()

        # AL52CO output
        co_out = DecadesVariable(d['CO_AERO'], name='CO_AERO',
                                 flag=DecadesBitmaskFlag)

        # Add flagging to the output
        for mask in flag_df.columns:
            co_out.flag.add_mask(flag_df[mask].values, *mask)

        # Write output
        self.add_output(co_out)

import datetime
import numpy as np

from ..decades import DecadesVariable
from .base import PPBase
from ..utils import flagged_avg

INIT_SKIP = 100         # Number of datapoints to skip at the start
SENS_CUTOFF = 0         # Sensitivity vals at or below considered bad
CAL_FLUSH_TIME = 3      # Time for system to flush after a cal
CAL_PRESS_THRESH = 3.4  # Flag when cal_press is higher than this


class AL52CO(PPBase):
    """
    Process CO concentration from the AL52002 instrument.

    The instrument provides counts, concentration, sensitivity, and zero.
    However, the sensitivity and zero step change after calibrations. Here we
    assume that the sensitivity and zero drift linearly between calibrations,
    and interpolate across the step changes to produce smoother sensitivity and
    zero-offset curves.
    """

    inputs = [
        'AL52CO_sens',
        'AL52CO_zero',
        'AL52CO_calpress',
        'AL52CO_cal_status',
        'AL52CO_counts',
        'WOW_IND'
    ]

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

        Flag info:
            CO concentration < -10  --> 3
            In calibration & 3s after calibration --> 3
            CO counts identically zero --> 3
            Calibration gas press. > 3.4 --> 3
            Aircraft on ground --> 1
        """

        d = self.d
        flag_var = 'CO_AERO_FLAG'

        # In the processing, we nan out the start of the data, we need to
        # replace this so that the .shift()).cumsum() method works.
        d['AL52CO_cal_status'].fillna(method='bfill', inplace=True)
        d[flag_var] = 0
        d.loc[d['CO_AERO'] < -10, flag_var] = 3

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
            d.loc[start:end, flag_var] = 3

        # Flag when pressure in the calibration chamber is high
        d.loc[d['AL52CO_calpress'] > CAL_PRESS_THRESH, flag_var] = 3

        # Flag when counts are identically zero
        d.loc[d['AL52CO_counts'] == 0, flag_var] = 3

        # Flag when the aircraft is on the ground
        d.loc[d['WOW_IND'] != 0, flag_var] = 1

        first_cal_start = d.loc[d['AL52CO_cal_status'] > 0].first('1s').index[0]
        d.loc[d.index <= first_cal_start, flag_var] = 3

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

        # Mask erroneous values in the zero
        d.loc[d['AL52CO_zero'] == 0, 'AL52CO_zero'] = np.nan
        d['AL52CO_zero'].fillna(method='bfill', inplace=True)

        # Build a flag indicating where the sensitivity and zero have changed
        # after a calibration, with a 2 sec safety buffer
        d['CAL_FLAG'] = d.AL52CO_sens.diff() != 0
        indicies = np.where(d.CAL_FLAG != 0)[0]
        d.loc[d.index[indicies + 2], 'CAL_FLAG'] = 1
        d.loc[d.index[indicies], 'CAL_FLAG'] = 0

        # Interpolate the zero
        flagged_avg(d, 'CAL_FLAG', 'AL52CO_zero', out_name='ZERO', interp=True)
        d.ZERO.fillna(method='bfill', inplace=True)

        # Interpolate the sensitivity
        flagged_avg(d, 'CAL_FLAG', 'AL52CO_sens', out_name='SENS', interp=True)
        d.SENS.fillna(method='bfill', inplace=True)

        # Calculate concentration using interpolated sens & zero
        d['CO_AERO'] = (d.AL52CO_counts - d.ZERO) / d.SENS

        # Flag the output
        self.flag()

        # Write output
        self.add_output(
            DecadesVariable(d[['CO_AERO', 'CO_AERO_FLAG']], name='CO_AERO')
        )

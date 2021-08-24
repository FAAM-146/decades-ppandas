"""
This module provides a postprocessing module for the Applanix GIN. See the
class docstring for more info.
"""
# pylint: disable=invalid-name
import datetime
import numpy as np
import pandas as pd

from ..decades import DecadesVariable, DecadesBitmaskFlag
from .base import PPBase, register_pp
from .shortcuts import _l, _o, _z

gin_name = 'POS AV 410 GPS-aided Inertial Navigation unit'
gin_manufacturer = 'Applanix'
gin_model = 'POS AV 410'


@register_pp('core')
class Gin(PPBase):
    r"""
    This module provides variables from the Applanix POS AV 410 GPS-aided
    inertial navigation system (GIN). The GIN provides parameters at a
    frequency of 50 Hz; this module simply downsamples these parameters to 32
    Hz.

    The STATUS_GIN parameter gives the current solution status reported by the
    GIN. This is defined as

    * 0: Full Nav. (user accuracies met)
    * 1: Fine Align (heading RMS < 15 deg)
    * 2: GC CHI 2 (alignment w/ GPS, RMS heading error > 15 deg)
    * 3: PC CHI 2 (alignment w/o GPS, RMS heading error > 15 deg)
    * 4: GC GHI 1 (alignment w/ GPS, RMS heading error > 45 deg)
    * 5: PC CHI 1 (alignment w/o GPS, RMS heading error > 45 deg)
    * 6: Course levelling active
    * 7: Initial solution assigned
    * 8: No solution
    """

    inputs = [
        'GINDAT_lat', 'GINDAT_lon', 'GINDAT_alt', 'GINDAT_veln',
        'GINDAT_vele', 'GINDAT_veld', 'GINDAT_roll', 'GINDAT_ptch',
        'GINDAT_hdg', 'GINDAT_wand', 'GINDAT_trck', 'GINDAT_gspd',
        'GINDAT_rolr', 'GINDAT_pitr', 'GINDAT_hdgr', 'GINDAT_aclf',
        'GINDAT_acls', 'GINDAT_acld', 'GINDAT_status',
        ]

    @staticmethod
    def test():
        """
        Return some dummy input data for testing.
        """
        return {
            'GINDAT_lat': ('data', _l(60, 60.1, 100), 50),
            'GINDAT_lon': ('data', _l(0, .1, 100), 50),
            'GINDAT_alt': ('data', _l(0, 1000, 100), 50),
            'GINDAT_veln': ('data', 100 * _o(100), 50),
            'GINDAT_vele': ('data', 100 * _o(100), 50),
            'GINDAT_veld': ('data', _z(100), 50),
            'GINDAT_roll': ('data', _z(100), 50),
            'GINDAT_ptch': ('data', 6 * _o(100), 50),
            'GINDAT_hdg': ('data', 45 * _o(100), 50),
            'GINDAT_wand': ('data', -150 * _o(100), 50),
            'GINDAT_trck': ('data', 45 * _o(100), 50),
            'GINDAT_gspd': ('data', 120 * _o(100), 50),
            'GINDAT_rolr': ('data', _z(100), 50),
            'GINDAT_pitr': ('data', _z(100), 50),
            'GINDAT_hdgr': ('data', _z(100), 50),
            'GINDAT_aclf': ('data', _z(100), 50),
            'GINDAT_acls': ('data', _z(100), 50),
            'GINDAT_acld': ('data', _z(100), 50),
            'GINDAT_status': ('data', _o(100), 50)
        }

    def declare_outputs(self):
        """
        Declare module outputs.
        """

        self.declare(
            'LAT_GIN',
            units='degree_north',
            frequency=32,
            long_name=f'Latitude from {gin_name}',
            standard_name='latitude',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'LON_GIN',
            units='degree_east',
            frequency=32,
            long_name=f'Longitude from {gin_name}',
            standard_name='longitude',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'ALT_GIN',
            units='m',
            frequency=32,
            long_name=f'Altitude from {gin_name}',
            standard_name='altitude',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'VELN_GIN',
            units='m s-1',
            frequency=32,
            long_name=f'Aircraft velocity north from {gin_name}',
            standard_name=None,
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'VELE_GIN',
            units='m s-1',
            frequency=32,
            long_name=f'Aircraft velocity east from {gin_name}',
            standard_name=None,
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'VELD_GIN',
            units='m s-1',
            frequency=32,
            long_name=f'Aircraft velocity down from {gin_name}',
            standard_name=None,
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'ROLL_GIN',
            units='degree',
            frequency=32,
            long_name=f'Roll angle from {gin_name}',
            standard_name='platform_roll_angle',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'PTCH_GIN',
            units='degree',
            frequency=32,
            long_name=f'Pitch angle from {gin_name}',
            standard_name='platform_pitch_angle',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'HDG_GIN',
            units='degree',
            frequency=32,
            long_name=f'Heading from {gin_name}',
            standard_name='platform_yaw_angle',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model,
            circular=True
        )

        self.declare(
            'WAND_GIN',
            units='degree s-1',
            frequency=32,
            long_name=f'Wander angle from {gin_name}',
            standard_name=None,
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model,
            circular=True
        )

        self.declare(
            'TRCK_GIN',
            units='degree',
            frequency=32,
            long_name=f'Aircraft track angle from {gin_name}',
            standard_name='platform_course',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model,
            circular=True
        )

        self.declare(
            'GSPD_GIN',
            units='m s-1',
            frequency=32,
            long_name=f'Groundspeed from {gin_name}',
            standard_name='platform_speed_wrt_ground',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'ROLR_GIN',
            units='degree s-1',
            frequency=32,
            long_name=f'Rate-of-change of roll angle from {gin_name}',
            standard_name='platform_roll_rate',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'PITR_GIN',
            units='degree s-1',
            frequency=32,
            long_name=f'Rate-of-change of pitch angle from {gin_name}',
            standard_name='platform_pitch_rate',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'HDGR_GIN',
            units='degree s-1',
            frequency=32,
            long_name=f'Rate-of-change of heading from {gin_name}',
            standard_name='platform_yaw_rate',
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'ACLF_GIN',
            units='m s-2',
            frequency=32,
            long_name=('Acceleration along the aircraft longitudinal axis '
                       f'from {gin_name} (positive forward)'),
            standard_name=None,
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'ACLS_GIN',
            units='m s-2',
            frequency=32,
            long_name=('Acceleration along the aircraft transverse axis from '
                       f'{gin_name} (positive starboard)'),
            standard_name=None,
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'ACLD_GIN',
            units='m s-2',
            frequency=32,
            long_name=('Acceleration along the aircraft vertical axis from '
                       f'{gin_name} (positive down)'),
            standard_name=None,
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

        self.declare(
            'STATUS_GIN',
            frequency=32,
            units='1',
            long_name=f'Solution status from {gin_name}',
            standard_name=None,
            instrument_manufacturer=gin_manufacturer,
            instrument_model=gin_model
        )

    def process(self):
        """
        Processing entry hook.
        """
        start = self.dataset[self.inputs[0]].index[0].round('1S')
        end = self.dataset[self.inputs[0]].index[-1].round('1S')

        index = pd.date_range(start, end, freq=self.freq[32])

        self.get_dataframe(
            method='onto', index=index,
            circular=['GINDAT_hdg', 'GINDAT_trck', 'GINDAT_wand']
        )

        try:
            time_offset = self.dataset['GIN_TIME_OFFSET']
        except Exception:
            time_offset = 0

        # Timeshift correction for older data compatibility
        self.d.index = self.d.index + datetime.timedelta(seconds=time_offset)

        # Round the status message to a integer
        finite = np.isfinite(self.d['GINDAT_status'])
        self.d.loc[
            finite, 'GINDAT_status'
        ] = self.d.loc[finite, 'GINDAT_status'].astype(int)

        self.d['SOLUTION_FLAG'] = self.d['GINDAT_status'] == 8
        self.d['HEADING_FLAG'] = (
            (self.d['GINDAT_status'] > 0) & (self.d['GINDAT_status'] < 6)
        )

        self.d['ZERO_FLAG'] = (
            (self.d['GINDAT_lon'] == 0) & (self.d['GINDAT_lat'] == 0)
        )

        for declaration in self.declarations:
            input_name = 'GINDAT_{}'.format(declaration.split('_')[0].lower())

            self.d[declaration] = self.d[input_name]

            dv = DecadesVariable(self.d[declaration], flag=DecadesBitmaskFlag)

            if input_name != 'GINDAT_status':
                dv.flag.add_mask(
                    self.d.SOLUTION_FLAG, 'no solution',
                    ('The GIN status flag indicates no solution has been '
                     'obtained.')
                )

            if input_name == 'GINDAT_hdg':
                dv.flag.add_mask(
                    self.d.HEADING_FLAG, 'gin align',
                    ('Status flag indicates instrument is in align, which may '
                     'indicate an increased heading error')
                )

            dv.flag.add_mask(
                self.d.ZERO_FLAG, 'latlon identically zero',
                ('Latitude and Longitude are exactly zero. This '
                 'indicates an erroneous data message.')
            )

            self.add_output(dv)

        self.dataset.lon = 'LON_GIN'
        self.dataset.lat = 'LAT_GIN'

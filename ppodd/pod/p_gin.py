import numpy as np
import pandas as pd

from ..decades import DecadesVariable, DecadesBitmaskFlag
from .base import PPBase
from .shortcuts import _l, _o, _z


class Gin(PPBase):
    r"""
    This module provides variables from the Applanix POS AV 410 GPS-aided
    inertial navigation system (GIN). The GIN provides parameters at a
    frequency of 50 Hz; this module simply downsamples these parameters to 32
    Hz. Optionally, an offset can be applied to the GIN heading, throught the
    flight constants parameter \texttt{GIN\_HDG\_OFFSET}.
    """

    inputs = [
        'GINDAT_lat', 'GINDAT_lon', 'GINDAT_alt', 'GINDAT_veln',
        'GINDAT_vele', 'GINDAT_veld', 'GINDAT_roll', 'GINDAT_ptch',
        'GINDAT_hdg', 'GINDAT_wand', 'GINDAT_trck', 'GINDAT_gspd',
        'GINDAT_rolr', 'GINDAT_pitr', 'GINDAT_hdgr', 'GINDAT_aclf',
        'GINDAT_acls', 'GINDAT_acld', 'GINDAT_status'
        ]

    @staticmethod
    def test():
        return {
            'GINDAT_lat': ('data', _l(60, 60.1, 100)),
            'GINDAT_lon': ('data', _l(0, .1, 100)),
            'GINDAT_alt': ('data', _l(0, 1000, 100)),
            'GINDAT_veln': ('data', 100 * _o(100)),
            'GINDAT_vele': ('data', 100 * _o(100)),
            'GINDAT_veld': ('data', _z(100)),
            'GINDAT_roll': ('data', _z(100)),
            'GINDAT_ptch': ('data', 6 * _o(100)),
            'GINDAT_hdg': ('data', 45 * _o(100)),
            'GINDAT_wand': ('data', -150 * _o(100)),
            'GINDAT_trck': ('data', 45 * _o(100)),
            'GINDAT_gspd': ('data', 120 * _o(100)),
            'GINDAT_rolr': ('data', _z(100)),
            'GINDAT_pitr': ('data', _z(100)),
            'GINDAT_hdgr': ('data', _z(100)),
            'GINDAT_aclf': ('data', _z(100)),
            'GINDAT_acls': ('data', _z(100)),
            'GINDAT_acld': ('data', _z(100)),
            'GINDAT_status': ('data', _o(100))
        }

    def declare_outputs(self):
        self.declare(
            'LAT_GIN',
            units='degree_north',
            frequency=32,
            long_name='Latitude from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='latitude'
        )

        self.declare(
            'LON_GIN',
            units='degree_east',
            frequency=32,
            long_name='Longitude from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='longitude'
        )

        self.declare(
            'ALT_GIN',
            units='m',
            frequency=32,
            long_name='Altitude from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='altitude'
        )

        self.declare(
            'VELN_GIN',
            units='m s-1',
            frequency=32,
            long_name='Aircraft velocity north from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name=None
        )

        self.declare(
            'VELE_GIN',
            units='m s-1',
            frequency=32,
            long_name='Aircraft velocity east from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name=None
        )

        self.declare(
            'VELD_GIN',
            units='m s-1',
            frequency=32,
            long_name='Aircraft velocity down from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name=None
        )

        self.declare(
            'ROLL_GIN',
            units='degree',
            frequency=32,
            long_name='Roll angle from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='platform_roll_angle'
        )

        self.declare(
            'PTCH_GIN',
            units='degree',
            frequency=32,
            long_name='Pitch angle from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='platform_pitch_angle'
        )

        self.declare(
            'HDG_GIN',
            units='degree',
            frequency=32,
            long_name='Heading from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='platform_yaw_angle',
            circular=True
        )

        self.declare(
            'WAND_GIN',
            units='degree s-1',
            frequency=32,
            long_name='GIN wander angle',
            standard_name=None,
            circular=True
        )

        self.declare(
            'TRCK_GIN',
            units='degree',
            frequency=32,
            long_name='Aircraft track angle from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='platform_course',
            circular=True
        )

        self.declare(
            'GSPD_GIN',
            units='m s-1',
            frequency=32,
            long_name='Groundspeed from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='platform_speed_wrt_ground'
        )

        self.declare(
            'ROLR_GIN',
            units='degree s-1',
            frequency=32,
            long_name='Rate-of-change of GIN roll angle',
            standard_name='platform_roll_rate'
        )

        self.declare(
            'PITR_GIN',
            units='degree s-1',
            frequency=32,
            long_name='Rate-of-change of GIN pitch angle',
            standard_name='platform_pitch_rate'
        )

        self.declare(
            'HDGR_GIN',
            units='degree s-1',
            frequency=32,
            long_name='Rate-of-change of GIN heading',
            standard_name='platform_yaw_rate'
        )

        self.declare(
            'ACLF_GIN',
            units='m s-2',
            frequency=32,
            long_name='Acceleration along the aircraft longitudinal axis (GIN) (positive forward)',
            standard_name=None
        )

        self.declare(
            'ACLS_GIN',
            units='m s-2',
            frequency=32,
            long_name='Acceleration along the aircraft transverse axis (GIN) (positive starboard)',
            standard_name=None
        )

        self.declare(
            'ACLD_GIN',
            units='m s-2',
            frequency=32,
            long_name='Acceleration along the aircraft vertical axis (GIN) (positive down)',
            standard_name=None
        )

    def process(self):
        start = self.dataset[self.inputs[0]].index[0].round('1S')
        end = self.dataset[self.inputs[0]].index[-1].round('1S')

        index = pd.date_range(start, end, freq=self.freq[32])

        self.get_dataframe(
            method='onto', index=index,
            circular=['GINDAT_hdg', 'GINDAT_trck', 'GINDAT_wand']
        )

        try:
            self.d['GINDAT_hdg'] += self.dataset['GIN_HDG_OFFSET']
            self.d['GINDAT_hdg'] %= 360
        except KeyError:
            print('Warning: GIN_HDG_OFFSET not defined')

        flag = np.around(self.d['GINDAT_status'] / 3)

        self.d['STATUS_FLAG'] = self.d['GINDAT_status'] != 0
        self.d['ZERO_FLAG'] = (
            (self.d['GINDAT_lon'] == 0) & (self.d['GINDAT_lat'] == 0)
        )

        for declaration in self.declarations:
            input_name = 'GINDAT_{}'.format(declaration.split('_')[0].lower())

            self.d[declaration] = self.d[input_name]

            dv = DecadesVariable(self.d[declaration], flag=DecadesBitmaskFlag)

            dv.flag.add_mask(
                self.d.STATUS_FLAG, 'gin status nonzero',
                ('The GIN status indicator is non-zero, indicating a potential '
                 'issue.')
            )

            dv.flag.add_mask(
                self.d.ZERO_FLAG, 'latlon identically zero',
                ('Either the latitude or longitude is exactly zero. This most '
                 'probably indicates erroneous data')
            )

            self.add_output(dv)

        self.dataset.lon = 'LON_GIN'
        self.dataset.lat = 'LAT_GIN'

import numpy as np
import pandas as pd

from ..decades import DecadesVariable
from .base import PPBase


class Gin(PPBase):

    inputs = [
        'GINDAT_lat', 'GINDAT_lon', 'GINDAT_alt', 'GINDAT_veln',
        'GINDAT_vele', 'GINDAT_veld', 'GINDAT_roll', 'GINDAT_ptch',
        'GINDAT_hdg', 'GINDAT_wand', 'GINDAT_trck', 'GINDAT_gspd',
        'GINDAT_rolr', 'GINDAT_pitr', 'GINDAT_hdgr', 'GINDAT_aclf',
        'GINDAT_acls', 'GINDAT_acld', 'GINDAT_status'
        ]

    def declare_outputs(self):
        self.declare(
            'LAT_GIN',
            units='degree_north',
            frequency=32,
            long_name='Latitude from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='latitude',
            number=610
        )

        self.declare(
            'LON_GIN',
            units='degree_east',
            frequency=32,
            long_name='Longitude from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='latitude',
            number=611
        )

        self.declare(
            'ALT_GIN',
            units='m',
            frequency=32,
            long_name='Altitude from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='altitude',
            number=612
        )

        self.declare(
            'VELN_GIN',
            units='m s-1',
            frequency=32,
            long_name='Aircraft velocity north from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name=None,
            number=613
        )

        self.declare(
            'VELE_GIN',
            units='m s-1',
            frequency=32,
            long_name='Aircraft velocity east from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name=None,
            number=614
        )

        self.declare(
            'VELD_GIN',
            units='m s-1',
            frequency=32,
            long_name='Aircraft velocity down from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name=None,
            number=615
        )

        self.declare(
            'ROLL_GIN',
            units='degree',
            frequency=32,
            long_name='Roll angle from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='platform_roll_angle',
            number=616
        )

        self.declare(
            'PTCH_GIN',
            units='degree',
            frequency=32,
            long_name='Pitch angle from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='platform_pitch_angle',
            number=617
        )

        self.declare(
            'HDG_GIN',
            units='degree',
            frequency=32,
            long_name='Heading from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='platform_yaw_angle',
            number=618
        )

        self.declare(
            'WAND_GIN',
            units='deg s-1',
            frequency=32,
            long_name='GIN wander angle',
            standard_name=None,
            number=619
        )

        self.declare(
            'TRCK_GIN',
            units='degree',
            frequency=32,
            long_name='Aircraft track angle from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='platform_course',
            number=620
        )

        self.declare(
            'GSPD_GIN',
            units='m s-1',
            frequency=32,
            long_name='Groundspeed from POS AV 510 GPS-aided Inertial Navigation unit',
            standard_name='platform_speed_wrt_ground',
            number=621
        )

        self.declare(
            'ROLR_GIN',
            units='degree s-1',
            frequency=32,
            long_name='Rate-of-change of GIN roll angle',
            standard_name='platform_roll_rate',
            number=622
        )

        self.declare(
            'PITR_GIN',
            units='degree s-1',
            frequency=32,
            long_name='Rate-of-change of GIN pitch angle',
            standard_name='platform_pitch_rate',
            number=623
        )

        self.declare(
            'HDGR_GIN',
            units='degree s-1',
            frequency=32,
            long_name='Rate-of-change of GIN heading',
            standard_name='platform_yaw_rate',
            number=624
        )

        self.declare(
            'ACLF_GIN',
            units='m s-2',
            frequency=32,
            long_name='Acceleration along the aircraft longitudinal axis (GIN) (positive forward)',
            standard_name=None,
            number=625
        )

        self.declare(
            'ACLS_GIN',
            units='m s-2',
            frequency=32,
            long_name='Acceleration along the aircraft transverse axis (GIN) (positive starboard)',
            standard_name=None,
            number=626
        )

        self.declare(
            'ACLD_GIN',
            units='m s-2',
            frequency=32,
            long_name='Acceleration along the aircraft vertical axis (GIN) (positive down)',
            standard_name=None,
            number=627
        )

    def process(self):
        index = pd.date_range(
            self.dataset.inputs[0].index[0].round('1S'),
            self.dataset.inputs[1].index[-1].round('1S'),
            freq=self.freq[32]
        )

        self.get_dataframe(
            method='onto', index=index,
            circular=['GINDAT_hdg', 'GINDAT_trck']
        )

        flag = np.around(self.d['GINDAT_status'] / 3)
        flag.loc[self.d['GINDAT_lon'] == 0 & (flag < 2)] = 2
        flag.loc[self.d['GINDAT_lon'] == 0 & (flag < 2)] = 2

        for declaration in self.declarations:
            input_name = 'GINDAT_{}'.format(declaration.split('_')[0].lower())

            _df = pd.DataFrame()
            _df[declaration] = self.d[input_name]

            dv = DecadesVariable(_df, name=declaration)
            dv.add_flag(flag)

            self.add_output(dv)

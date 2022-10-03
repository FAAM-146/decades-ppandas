import datetime
import numpy as np
from .base import PPBase, register_pp
from ..utils import flagged_avg
from ..decades import DecadesVariable, DecadesBitmaskFlag, flags
from .shortcuts import _z, _c, _o

CAL_FLUSH_START = 3
CAL_FLUSH_END = 5
WOW_FLAG_BUFFER = 10
ZERO_FLAG_BUFFER = 10
CAL_FLAG_BUFFER = 220

@register_pp('core')
class TecoSO2V2(PPBase):
    r"""
    Calculate SO\ :math:`_2` concentration from the TECO 43 instrument.

    Zeros are taken whenever valve states V7 or V8 are 1, and are interpolated back to 1 Hz, 
    assuming a linear drift of the offsets between zeros. SO\ :math:`_2` concentration is
    then given by

    .. math::
        \left[\text{SO}_2\right] = \frac{\left[\text{SO}_{2|\text{INS}}\right] - Z}{S},

    where :math:`\left[\text{SO}_{2|\text{INS}}\right]` is the concentration reported
    by the instrument, :math:`Z` is the zero obtained from sampling scrubbed ambient air,
    and :math:`S` is the instrument sensitivity given in the flight constants.

    Flagging is based on valve states, weight on wheels, and the instrument status flag.
    """

    VALID_AFTER = datetime.date(2021, 9, 1)

    inputs = [
        'CHTSOO_concentration',
        'CHTSOO_flags',
        'CHTSOO_pmt_volt',
        'CHTSOO_MFM',
        'CHTSOO_MFC3_mass_flow',
        'CHTSOO_V6',
        'CHTSOO_V7',
        'CHTSOO_V8',
        'WOW_IND',
        'CHTSOO_SENS'
    ]

    @staticmethod
    def test():
        return {
            'CHTSOO_concentration': ('data', _o(100), 1),
            'CHTSOO_flags': ('data', [b'cc0000'] * 100, 1),
            'CHTSOO_pmt_volt': ('data', _o(100) * -700, 1),
            'CHTSOO_MFM': ('data', _o(100) * 1.2, 1),
            'CHTSOO_MFC3_mass_flow': ('data', _o(100) * 1.9, 1),
            'CHTSOO_V6': ('data', _c([_z(10), _o(10), _z(10), _o(70)]), 1),
            'CHTSOO_V7': ('data', _c([_z(10), _o(10), _z(10), _o(70)]), 1),
            'CHTSOO_V8': ('data', _c([_z(10), _o(10), _z(10), _o(70)]), 1),
            'WOW_IND': ('data', _c([_o(20), _z(70), _o(10)]), 1),
            'CHTSOO_SENS': ('const', 1)
        }

    def declare_outputs(self):
        self.declare(
            'SO2_TECO',
            units='ppb',
            frequency=1,
            long_name='Mole fraction of Sulphur Dioxide in air from TECO 43 instrument',
            standard_name='mole_fraction_of_sulfur_dioxide_in_air',
            instrument_manufacturer='Thermo Fisher Scientific, Inc.',
            instrument_model='43i Trace Level-Enhanced pulsed fluorescence SO2 spectrometer',
            instrument_serial_number='1505564557'
        )

        self.declare(
            'SO2_TECO_ZERO',
            units='ppb',
            frequency=1,
            long_name='TECO 43 SO2 interpolated zero',
            instrument_manufacturer='Thermo Fisher Scientific, Inc.',
            instrument_model='43i Trace Level-Enhanced pulsed fluorescence SO2 spectrometer',
            instrument_serial_number='1505564557',
            write=False
        )

    def get_wow_flag(self):
        d = self.d
        wow = self.d['WOW_IND'] == 1
        to = self.dataset.takeoff_time
        wow.loc[[to + datetime.timedelta(seconds=i) for i in range(WOW_FLAG_BUFFER)]] = 1
        d['wow_flag'] = wow

    def get_status_flag(self):
        self.d['status_flag'] = 0
        flags = self.d['CHTSOO_flags']
        volts = self.d['CHTSOO_pmt_volt']
        
        self.d.loc[flags != 40000000, 'status_flag'] = 1
        self.d.loc[volts > -690, 'status_flag'] = 1

    def get_mfm_flag(self):
        self.d['mfm_flag'] = 0
        self.d.loc[self.d['CHTSOO_MFM'] < 1.2, 'mfm_flag'] = 1

    def get_flow_flag(self):
        self.d['flow_flag'] = 0
        self.d.loc[self.d['CHTSOO_MFC3_mass_flow'] < 1.9, 'flow_flag'] = 1

    def get_cal_or_zero_flag(self):
        in_cal = self.d['CHTSOO_V8'] == 1
        in_zero = (self.d['CHTSOO_V6'] == 1) | (self.d['CHTSOO_V7'] == 1)

        leaving_zero = (in_zero - in_zero.shift()) == -1
        leaving_zero = leaving_zero[leaving_zero]
        
        for i in leaving_zero.index:
            in_zero.loc[i:i+datetime.timedelta(seconds=ZERO_FLAG_BUFFER)] = 1
            
        leaving_cal = (in_cal - in_cal.shift()) == -1
        leaving_cal = leaving_cal[leaving_cal]

        for i in leaving_cal.index:
            in_cal.loc[i:i+datetime.timedelta(seconds=CAL_FLAG_BUFFER)] = 1

        self.d['zero_cal_flag'] = in_zero | in_cal

    def flag(self, var):
        self.get_wow_flag()
        self.get_status_flag()
        self.get_mfm_flag()
        self.get_flow_flag()
        self.get_cal_or_zero_flag()
        
        flag_info = {
            'wow_flag': (
                'aircraft_on_ground',
                (f'The aircraft is on the ground, as indicated by the weight on '
                 f'wheels flag. An additional {WOW_FLAG_BUFFER} seconds are added'
                 f'after take-off to allow for instrument flushing.')
            ),
            'status_flag': (
                'instrument flagged',
                ('The instrument is reporting an alarm for one of the following: '
                 'internal or fluorescence cell temperatures, lamp intensity or '
                  'voltage, PMT voltage or fluorscence cell pressure.')
            ),
            'mfm_flag': (
                'inlet flow out of range',
                'The inlet sample overflow is out of range.'
            ),
            'flow_flag': (
                'sample flow out of range',
                'The spectrometer sample mass flow rate is out of range.'
            ),
            'zero_cal_flag': (
                'instrument in cal mode',
                'The instrument is being calibrated for either span or zero' 
            )

        }

        for col, (short, long) in flag_info.items():
            var.flag.add_mask(self.d[col], short, long)

    def process(self):
        
        self.get_dataframe()
        sens = self.dataset['CHTSOO_SENS']

        self.d['zero_flag'] = (self.d['CHTSOO_V7'] == 1)

        flagged_avg(self.d, 'zero_flag', 'CHTSOO_concentration', out_name='zero',
                    interp=True, skip_start=CAL_FLUSH_START, skip_end=CAL_FLUSH_END)

        self.d['zero'].ffill(inplace=True)

        conc = (self.d['CHTSOO_concentration'] - self.d['zero']) / sens

        so2 = DecadesVariable(conc, flag=DecadesBitmaskFlag, name='SO2_TECO')
        zero = DecadesVariable(self.d['zero'], flag=None, name='SO2_TECO_ZERO')

        self.flag(so2)

        self.add_output(so2)
        self.add_output(zero)
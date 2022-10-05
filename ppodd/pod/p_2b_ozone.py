import datetime

from vocal.schema_types import OptionalDerivedString

from ..decades import DecadesVariable
from ..decades.attributes import DocAttribute
from .base import PPBase, register_pp
from .shortcuts import _o, _z, _c

WOW_FLAG_BUFFER = 10


@register_pp('core')
class TwoBOzone(PPBase):
    r"""
    Provides ozone concentration from the 2B Tech Ozone instrument. The instrument
    natively provides an ozone concentration, however this may be zero adjusted
    and scaled by zero and senstivity parameters given in the flight constants
    as `TWBOZO_ZERO` and `TWBOZO_SENS`. 

    .. math::
        \text{O}_3 = \frac{\text{O}_3 - z}{S},

    where :math:`z` is the given zero offset and :math:`S` is the given
    sensitivity.
    
    Additional calibration information for metadata can be provided through the
    constant parameters `TWBOZO_CALINFO_DATE`, `TWBOZO_CALINFO_INFO`, and
    `TWBOZO_CALINFO_URL`.
    
    Flagging information is provided based on instrument and aircraft status.
    """

    inputs = [
        'TWBOZO_conc',
        'TWBOZO_MFM',
        'TWBOZO_ZERO',
        'TWBOZO_SENS',
        'WOW_IND'
    ]

    @staticmethod
    def test():
        return {
            'TWBOZO_conc': ('data', _o(100), 1),
            'TWBOZO_MFM': ('data', _o(100) * 1.2, 1),
            'TWBOZO_ZERO': ('const', 0.),
            'TWBOZO_SENS': ('const', 1.),
            'WOW_IND': ('data', _c([_o(20), _z(70), _o(10)]), 1),
            # Optional calibration info...
            'TWBOZO_CALINFO_DATE': ('const', DocAttribute(
                value=datetime.date(2000, 1, 1), doc_value=OptionalDerivedString
            )),
            'TWBOZO_CALINFO_INFO': ('const', DocAttribute(
                value='Calibrated in a lab', doc_value=OptionalDerivedString
            )),
            'TWBOZO_CALINFO_URL': ('const', DocAttribute(
                value='https://some.url', doc_value=OptionalDerivedString
            ))
        }


    def declare_outputs(self):
        self.declare(
            'O3_2BTECH',
            units='ppb',
            frequency=1,
            long_name=('Mole fraction of Ozone in air from the 2BTech '
                       'photometer'),
            standard_name='mole_fraction_of_ozone_in_air',
            instrument_manufacturer='2B Technologies Inc.',
            instrument_model='205',
            instrument_serial_number='1034DB',
            instrument_description='Motherboard PCB version "l"',
            calibration_information=self.dataset.lazy['TWBOZO_CALINFO_INFO'],
            calibration_date=self.dataset.lazy['TWBOZO_CALINFO_DATE'],
            calibration_url=self.dataset.lazy['TWBOZO_CALINFO_URL']
        )

    def get_wow_flag(self):
        d = self.d
        wow = self.d['WOW_IND'] == 1

        # Get the takeoff time - emulate this if we're in test mode.
        to = self.dataset.takeoff_time
        if self.test_mode:
            to = self.d.index[20]

        wow.loc[[to + datetime.timedelta(seconds=i) for i in range(WOW_FLAG_BUFFER)]] = 1
        d['wow_flag'] = wow

    def get_mfm_flag(self):
        self.d['mfm_flag'] = 0
        self.d.loc[self.d['TWBOZO_MFM'] < 1.2, 'mfm_flag'] = 1

    def get_flow_flag(self):
        pass

    def flag(self, var):
        self.get_wow_flag()
        self.get_flow_flag()
        self.get_mfm_flag()

        flag_info = {
            'wow_flag': (
                'aircraft_on_ground',
                (f'The aircraft is on the ground, as indicated by the weight on '
                 f'wheels flag. An additional {WOW_FLAG_BUFFER} seconds are added'
                 f'after take-off to allow for instrument flushing.')
            ),
            'mfm_flag': (
                'inlet flow out of range',
                'The inlet sample overflow is out of range.'
            ),
            'flow_flag': (
                'sample flow out of range',
                'The spectrometer sample mass flow rate is out of range.'
            ),
            'zero_flag': (
                'instrument in zero mode',
                'The instrument is being zeroed. This should be in preflight only' 
            )

        }

        for col, (short, long) in flag_info.items():
            var.flag.add_mask(self.d[col], short, long)

    def process(self):
        self.get_dataframe()
        zero = self.dataset['TWBOZO_ZERO']
        sens = self.dataset['TWBOZO_SENS']
        conc = (self.d['TWBOZO_conc'] - zero) / sens

        conc_out = DecadesVariable(conc, name='O3_2BTECH', flag=None)

        self.add_output(conc_out)
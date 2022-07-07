import datetime
from .base import PPBase, register_pp
from ..utils import flagged_avg
from ..decades import DecadesVariable, DecadesBitmaskFlag, flags
from .shortcuts import _z, _c, _o

CAL_FLUSH_START = 3
CAL_FLUSH_END = 5

@register_pp('core')
class TecoSO2V2(PPBase):
    r"""
    Calculate SO\ :math:`_2` concentration from the TECO 43 instrument. The
    instrument reports a nominal concentration, and valve state V7, indicating
    that the instrument is in zero mode.

    Zeros are taken whenever V7 is 1, and are interpolated back to 1 Hz, assuming
    a linear drift of the offsets between zeros. SO\ :math:`_2` concentration is
    then given by

    .. math::
        \left[\text{SO}_2\right] = \frac{\left[\text{SO}_{2|\text{INS}}\right] - Z}{S},

    where :math:`\left[\text{SO}_{2|\text{INS}}\right]` is the concentration reported
    by the instrument, :math:`Z` is the zero obtained from sampling scrubbed ambient air,
    and :math:`S` is the instrument sensitivity given in the flight constants.

    Flagging is based on valve states, weight on wheels, and the instrument status flag.
    """

    VALID_AFTER = datetime.date(2022, 1, 1)

    inputs = [
        'CHTSOO_concentration',
        'CHTSOO_V7',
        'CHTSOO_flags',
        'CHTSOO_SENS',
        'WOW_IND'
    ]

    @staticmethod
    def test():
        return {
            'CHTSOO_concentration': ('data', _o(100), 1),
            'CHTSOO_V7': ('data', _c([_z(10), _o(10), _z(10), _o(70)]), 1),
            'CHTSOO_flags': ('data', [b'cc0000'] * 100, 1),
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
            instrument_model='43i TLE pulsed fluorescence SO2 spectrometer',
            instrument_serial_number='1505564557'
        )

    def flag(self, var):
        wow_flag = self.d['WOW_IND'] == 1
        zero_flag = self.d['zero_flag']
        alarm = self.d['CHTSOO_flags'].apply(lambda x: str(x)[-4:-1] != '000')

        var.flag.add_mask(wow_flag, flags.WOW, 'Aircraft is on the ground')
        var.flag.add_mask(zero_flag, 'zero_mode', 'Instrument is in zero mode')
        var.flag.add_mask(
            alarm, 'instrument in alarm',
            'The instrument status flag indicates that the instrument is in alarm'
        )

    def process(self):
        
        self.get_dataframe()
        sens = self.dataset['CHTSOO_SENS']

        self.d['zero_flag'] = (self.d['CHTSOO_V7'] == 1)

        flagged_avg(self.d, 'zero_flag', 'CHTSOO_concentration', out_name='zero',
                    interp=True, skip_start=CAL_FLUSH_START, skip_end=CAL_FLUSH_END)

        self.d['zero'].ffill(inplace=True)

        conc = (self.d['CHTSOO_concentration'] - self.d['zero']) / sens

        so2 = DecadesVariable(conc, flag=DecadesBitmaskFlag, name='SO2_TECO')

        self.flag(so2)
        self.add_output(so2)
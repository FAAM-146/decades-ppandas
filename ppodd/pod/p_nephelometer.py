"""
Provides a processing module for the Nephelometer.
"""
# pylint: disable=invalid-name
import numpy as np

from ..decades import DecadesVariable
from ..decades.flags import DecadesBitmaskFlag
from .base import PPBase, register_pp
from .shortcuts import _z

INSTRUMENT_FAULT = 2


@register_pp('core')
class Nephelometer(PPBase):
    r"""
    Provides data and flagging information from the TSI nephelometer. The data
    from the nephelometer do not require any further processing, however this
    module parses the instrument status information to provide a QC flag for
    the data.

    .. note::
        Prior to software version 24.0.0, this module used classic flagging. From
        version 24.0.0, the module uses bitmask flagging, and further flagging may
        be added by standalone flagging modules.
    """

    inputs = [
        'AERACK_neph_total_blue',
        'AERACK_neph_total_green',
        'AERACK_neph_pressure',
        'AERACK_neph_temp',
        'AERACK_neph_backscatter_blue',
        'AERACK_neph_backscatter_green',
        'AERACK_neph_backscatter_red',
        'AERACK_neph_total_red',
        'AERACK_neph_humidity',
        'AERACK_neph_status',
        'AERACK_neph_mode'
    ]

    @staticmethod
    def test():
        """
        Provide dummy input data for testing.
        """

        return {
            'AERACK_neph_total_blue': ('data', _z(100), 1),
            'AERACK_neph_total_green': ('data', _z(100), 1),
            'AERACK_neph_pressure': ('data', _z(100), 1),
            'AERACK_neph_temp': ('data', _z(100), 1),
            'AERACK_neph_backscatter_blue': ('data', _z(100), 1),
            'AERACK_neph_backscatter_green': ('data', _z(100), 1),
            'AERACK_neph_backscatter_red': ('data', _z(100), 1),
            'AERACK_neph_total_red': ('data', _z(100), 1),
            'AERACK_neph_humidity': ('data', _z(100), 1),
            'AERACK_neph_status': ('data', _z(100), 1),
            'AERACK_neph_mode': ('data', _z(100), 1)
        }

    def declare_outputs(self):
        """
        Declare output variables.
        """

        # pylint: disable=unreachable
        neph_manufacturer='TSI'
        neph_model='3563'

        self.declare(
            'NEPH_PR',
            units='hPa',
            frequency=1,
            long_name='Internal sample pressure of the Nephelometer',
            instrument_manufacturer=neph_manufacturer,
            instrument_model=neph_model
        )

        self.declare(
            'NEPH_T',
            units='K',
            frequency=1,
            long_name='Internal sample temperature of the Nephelometer',
            instrument_manufacturer=neph_manufacturer,
            instrument_model=neph_model
        )

        self.declare(
            'NEPH_RH',
            units='%',
            frequency=1,
            long_name='Relative humidity from TSI 3563 Nephelometer',
            instrument_manufacturer=neph_manufacturer,
            instrument_model=neph_model
        )

        self.declare(
            'TSC_BLUU',
            units='m-1',
            frequency=1,
            long_name=('Uncorrected blue total scattering coefficient from '
                       'TSI 3563 Nephelometer'),
            instrument_manufacturer=neph_manufacturer,
            instrument_model=neph_model
        )

        self.declare(
            'TSC_GRNU',
            units='m-1',
            frequency=1,
            long_name=('Uncorrected green total scattering coefficient from '
                       'TSI 3563 Nephelometer'),
            instrument_manufacturer=neph_manufacturer,
            instrument_model=neph_model
        )

        self.declare(
            'TSC_REDU',
            units='m-1',
            frequency=1,
            long_name=('Uncorrected red total scattering coefficient from '
                       'TSI 3563 Nephelometer'),
            instrument_manufacturer=neph_manufacturer,
            instrument_model=neph_model
        )

        self.declare(
            'BSC_BLUU',
            units='m-1',
            frequency=1,
            long_name=('Uncorrected blue back scattering coefficient from '
                       'TSI 3563 Nephelometer'),
            instrument_manufacturer=neph_manufacturer,
            instrument_model=neph_model
        )

        self.declare(
            'BSC_GRNU',
            units='m-1',
            frequency=1,
            long_name=('Uncorrected green back scattering coefficient from '
                       'TSI 3563 Nephelometer'),
            instrument_manufacturer=neph_manufacturer,
            instrument_model=neph_model
        )

        self.declare(
            'BSC_REDU',
            units='m-1',
            frequency=1,
            long_name=('Uncorrected red back scattering coefficient from '
                       'TSI 3563 Nephelometer'),
            instrument_manufacturer=neph_manufacturer,
            instrument_model=neph_model
        )


    def flag(self):
        """
        Provide flagging information from the Neph status string.
        """
        d = self.d
        d.dropna()
        d['RH_FLAG'] = 0
        d['SC_FLAG'] = 0
        d['T_FLAG'] = 0
        d['P_FLAG'] = 0

        neph_status_list = list(d['AERACK_neph_status'].ffill())
        tmp = ['{0:04}'.format(int(x)) for x in neph_status_list]
        neph_status = [
            '{0:04b}{1:04b}{2:04b}{3:04b}'.format(
                int(t[0]), int(t[1]), int(t[2]), int(t[3])
            ) for t in tmp
        ]

        # Lamp Fault
        ix = np.array([int(flag[-1]) for flag in neph_status])
        d.loc[ix==1, 'SC_FLAG'] = INSTRUMENT_FAULT

        # Valve Fault
        ix = np.array([int(flag[-2]) for flag in neph_status])
        d.loc[ix==1, 'SC_FLAG'] = INSTRUMENT_FAULT

        # Chopper Fault
        ix = np.array([int(flag[-3]) for flag in neph_status])
        d.loc[ix==1, 'SC_FLAG'] = INSTRUMENT_FAULT

        # Shutter Fault
        ix = np.array([int(flag[-4]) for flag in neph_status])
        d.loc[ix==1, 'SC_FLAG'] = INSTRUMENT_FAULT

        # Heater unstable
        ix = np.array([int(flag[-5]) for flag in neph_status])
        d.loc[ix==1, 'SC_FLAG'] = INSTRUMENT_FAULT
        d.loc[ix==1, 'RH_FLAG'] = INSTRUMENT_FAULT
        d.loc[ix==1, 'T_FLAG'] = INSTRUMENT_FAULT

        # Pressure out of range
        ix = np.array([int(flag[-6]) for flag in neph_status])
        d.loc[ix==1, 'SC_FLAG'] = INSTRUMENT_FAULT
        d.loc[ix==1, 'P_FLAG'] = INSTRUMENT_FAULT

        # Sample T out of range
        ix = np.array([int(flag[-7]) for flag in neph_status])
        d.loc[ix==1, 'SC_FLAG'] = INSTRUMENT_FAULT
        d.loc[ix==1, 'T_FLAG'] = INSTRUMENT_FAULT

        # Inlet T out of range
        ix = np.array([int(flag[-8]) for flag in neph_status])
        d.loc[ix==1, 'SC_FLAG'] = INSTRUMENT_FAULT

        # RH out of range
        ix = np.array([int(flag[-9]) for flag in neph_status])
        d.loc[ix==1, 'RH_FLAG'] = INSTRUMENT_FAULT

        # All faults using the last 9 digits from the neph_status number
        ix = np.array([int(flag[-9:]) for flag in neph_status])
        d.loc[ix==111111111, 'RH_FLAG'] = INSTRUMENT_FAULT
        d.loc[ix==111111111, 'SC_FLAG'] = INSTRUMENT_FAULT
        d.loc[ix==111111111, 'T_FLAG'] = INSTRUMENT_FAULT
        d.loc[ix==111111111, 'P_FLAG'] = INSTRUMENT_FAULT


    def process(self):
        """
        Processing entry hook.
        """
        # Instrument is perm U/S, but we're going to hold on the the code until
        # we get a new one
        # pylint: disable=unreachable

        self.get_dataframe()
        self.flag()

        pairs = (
            ('AERACK_neph_total_blue', 'TSC_BLUU'),
            ('AERACK_neph_total_green', 'TSC_GRNU'),
            ('AERACK_neph_total_red', 'TSC_REDU'),
            ('AERACK_neph_backscatter_blue', 'BSC_BLUU'),
            ('AERACK_neph_backscatter_green', 'BSC_GRNU'),
            ('AERACK_neph_backscatter_red', 'BSC_REDU'),
            ('AERACK_neph_pressure', 'NEPH_PR'),
            ('AERACK_neph_temp', 'NEPH_T'),
            ('AERACK_neph_humidity', 'NEPH_RH')
        )

        for pair in pairs:
            tcp_name, out_name = pair

            dv = DecadesVariable(
                self.d[tcp_name], name=out_name, flag=DecadesBitmaskFlag
            )

            if 'SC_' in out_name:
                dv.data /= 1e6
                dv.flag.add_mask(
                    self.d['SC_FLAG'], 'instrument_flag_raised',
                    'A fault has been raised by the instrument'
                )

            elif '_PR' in out_name:
                dv.flag.add_mask(
                    self.d['P_FLAG'], 'instrument_flag_raised',
                    'A fault has been raised by the instrument'
                )

            elif '_T' in out_name:
                dv.flag.add_mask(
                    self.d['T_FLAG'], 'instrument_flag_raised',
                    'A fault has been raised by the instrument'
                )

            elif '_RH' in out_name:
                dv.flag.add_mask(
                    self.d['RH_FLAG'], 'instrument_flag_raised',
                    'A fault has been raised by the instrument'
                )

            self.add_output(dv)

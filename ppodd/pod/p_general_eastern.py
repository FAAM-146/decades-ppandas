"""
This module provides a prosprocessing module for the General Eastern chilled
mirror hygrometer. See class docstring for more information.
"""
# pylint: disable=invalid-name
import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..utils import get_range_flag
from ..utils.conversions import celsius_to_kelvin
from .base import PPBase, register_pp
from .shortcuts import _o

TDEW_VALID_RANGE = (195, 394)


@register_pp('core')
class GeneralEastern(PPBase):
    r"""
    Processing module to calculate Dew Point temperature from the General
    Eastern 1011B chilled mirror hygrometer. Counts from the core console are
    converted to dew point with a polynomial fit, using coefficients provided
    in the flight constants paramater ``CALGE``. The General Eastern
    provides a control signal voltage to indicate whether the instrument is
    controlling on a due point, and the data are flagged when outside this
    range. The valid range is provided through the flight constants parameter
    ``GELIMS``.
    """

    inputs = [
        'GELIMS',           #  Control sig. valid range (min, max) (Const)
        'CALGE',            #  Dewpoint calibrations (Const)
        'CORCON_ge_dew',    #  Gen. East. dewpoint (DLU)
        'CORCON_ge_cont'    #  Gen. East. control signal (DLU)
    ]

    @staticmethod
    def test():
        """
        Return dummy input data for testing.
        """
        return {
            'GELIMS': ('const', [7000, 5000]),
            'CALGE': ('const', [-80, 4e-3, -2.5e-10]),
            'GE_SN': ('const', '1234'),
            'CORCON_ge_dew': ('data', 125e2 * _o(100), 4),
            'CORCON_ge_cont': ('data', 5000 * _o(100), 4)
        }

    def declare_outputs(self):
        """
        Declare module outputs.
        """

        self.declare(
            'TDEW_GE',
            units='degK',
            frequency=4,
            long_name='Dew Point from the General Eastern instrument',
            standard_name='dew_point_temperature',
            instrument_manufacturer='General Eastern Instruments',
            instrument_model='1011B Chilled Mirror Hygrometer',
            instrument_serial_number=self.dataset.lazy['GE_SN'],
            calibration_information='This instrument cannot be calibrated'
        )

    def flag_control(self):
        """
        Calculate a flag based on the General Eastern control signal.

        Flag as 3 <=> Control signal is 0 or nan
        Flag as 2 <=> Control signal is out of range (given by flight
                      constant GELIMS).

        Sets 'CONTROL_FLAG' in the module dataframe, and returns the
        corresponding Series.
        """
        d = self.d

        control_lims = self.dataset['GELIMS']

        d['CONTROL_MISSING_FLAG'] = 0
        d['CONTROL_RANGE_FLAG'] = 0

        # Missing or 0 data flagged as 3
        d.loc[d['CORCON_ge_cont'] == 0, 'CONTROL_MISSING_FLAG'] = 1
        d.loc[np.isnan(d['CORCON_ge_cont']), 'CONTROL_MISSING_FLAG'] = 1

        # Out of range data flagged as 2
        d.loc[d['CORCON_ge_cont'] > control_lims[0], 'CONTROL_RANGE_FLAG'] = 1
        d.loc[d['CORCON_ge_cont'] < control_lims[1], 'CONTROL_RANGE_FLAG'] = 1

    def process(self):
        """
        Module processing entry point.
        """

        self.get_dataframe()
        d = self.d

        _cals = self.dataset['CALGE'][::-1]
        d['TDEW_GE'] = celsius_to_kelvin(
            np.polyval(_cals, d['CORCON_ge_dew'])
        )

        # Build flag arrays
        self.flag_control()
        range_flag = get_range_flag(d['TDEW_GE'], TDEW_VALID_RANGE, flag_val=1)

        # Create the output variable and add the flags
        tdew = DecadesVariable(d['TDEW_GE'], flag=DecadesBitmaskFlag)

        tdew.flag.add_mask(
            d['CONTROL_MISSING_FLAG'], 'control data missing',
            ('No control data is available. The instrument may or may not be '
             'controlling on a dew point')
        )

        tdew.flag.add_mask(
            d['CONTROL_RANGE_FLAG'], 'control out of range',
            ('The control signal is outside of the specified valid range '
             '[{}, {}]'.format(
                 self.dataset['GELIMS'][0], self.dataset['GELIMS'][1]
             ))
        )

        tdew.flag.add_mask(
            range_flag, 'dewpoint out of range',
            'Dew point outside valid range [{}, {}] K'.format(
                *TDEW_VALID_RANGE
            )
        )

        # Add the output to the parent Dataset
        self.add_output(tdew)

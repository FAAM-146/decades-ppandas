import numpy as np

from ..decades import DecadesVariable
from ..utils import get_range_flag
from ..utils.conversions import celsius_to_kelvin
from .base import PPBase

TDEW_VALID_RANGE = (195, 394)


class GeneralEastern(PPBase):
    """
    Processing module to calculate Dew Point temperature from the General
    Eastern 1011 chilled mirror hygrometer.
    """

    inputs = [
        'GELIMS',           #  Control sig. valid range (min, max) (Const)
        'CALGE',            #  Dewpoint calibrations (Const)
        'CORCON_ge_dew',    #  Gen. East. dewpoint (DLU)
        'CORCON_ge_cont'    #  Gen. East. control signal (DLU)
    ]

    def declare_outputs(self):
        """
        Declare module outputs.
        """

        self.declare(
            'TDEW_GE',
            units='degK',
            frequency=4,
            number=529,
            long_name='Dew Point from the General Eastern instrument',
            standard_name='dew_point_temperature'
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

        d['CONTROL_FLAG'] = 0

        # Missing or 0 data flagged as 3
        d.loc[d['CORCON_ge_cont'] == 0, 'CONTROL_FLAG'] = 3
        d.loc[np.isnan(d['CORCON_ge_cont']), 'CONTROL_FLAG'] = 3

        # Out of range data flagged as 2
        d.loc[d['CORCON_ge_cont'] > control_lims[0], 'CONTROL_FLAG'] = 2
        d.loc[d['CORCON_ge_cont'] < control_lims[1], 'CONTROL_FLAG'] = 2

        return d['CONTROL_FLAG']

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
        control_flag = self.flag_control()
        range_flag = get_range_flag(d['TDEW_GE'], TDEW_VALID_RANGE)

        # Create the output variable and add the flags
        tdew = DecadesVariable(d['TDEW_GE'])
        tdew.add_flag(control_flag)
        tdew.add_flag(range_flag)

        # Add the output to the parent Dataset
        self.add_output(tdew)
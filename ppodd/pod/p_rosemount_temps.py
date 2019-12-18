import numpy as np
import pandas as pd

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..utils.calcs import sp_mach, true_air_temp
from ..utils.conversions import celsius_to_kelvin
from .base import PPBase
from .shortcuts import *

class RosemountTemperatures(PPBase):
    """
    Calculate true air temperatures from the Rosemount temperature
    probes.
    """

    inputs = [
        'TRFCTR',                   #  Recovery factors (Const)
        'CALDIT',                   #  Deiced calibrations (Const)
        'CALNDT',                   #  Non deiced calibrations (Const)
        'NDTSENS',                  #  Non deiced sensor type (Const)
        'DITSENS',                  #  Deiced sensor type (Const)
        'PS_RVSM',                  #  Static pressure (derived)
        'Q_RVSM',                   #  Pitot-static pressure (derived)
        'CORCON_di_temp',           #  Deiced temperature counts (DLU)
        'CORCON_ndi_temp',          #  Non deiced temperature counts (DLU)
        'PRTAFT_deiced_temp_flag'   #  Deiced heater indicator flag (DLU)
    ]

    @staticmethod
    def test():
        return {
            'TRFCTR': ('const', [1., 1.]),
            'CALDIT': ('const', [0, 0, 0]),
            'CALNDT': ('const', [0, 0, 0]),
            'NDTSENS': ('const', 'ndi_serial'),
            'DITSENS': ('const', 'dit_serial'),
            'PS_RVSM': ('data', _a(1000, 300, -1)),
            'Q_RVSM': ('data', 250*(_o(700))),
            'CORCON_di_temp': ('data', _a(225, 245, .0286)*1000),
            'CORCON_ndi_temp': ('data', _a(225, 245, .0286)*1000),
            'PRTAFT_deiced_temp_flag': ('data', _c([_z(200), _o(300), _z(200)])
            ),
        }

    def declare_outputs(self):
        """
        Declare the outputs that are going to be written by this module.
        """

        self.declare(
            'TAT_DI_R',
            units='degK',
            frequency=32,
            long_name=('True air temperature from the Rosemount deiced '
                       'temperature sensor'),
            standard_name='air_temperature'
        )

        self.declare(
            'TAT_ND_R',
            units='degK',
            frequency=32,
            long_name=('True air temperature from the Rosemount non-deiced '
                       'temperature sensor'),
            standard_name='air_temperature'
        )

        self.declare(
            'IAT_DI_R',
            units='K',
            frequency=32,
            long_name=('Indicated air temperature from the Rosemount deiced '
                       'temperature sensor'),
            write=False
        )

        self.declare(
            'IAT_ND_R',
            units='K',
            frequency=32,
            long_name=('Indicated air temperature from the Rosemount '
                       'non-deiced temperature sensor'),
            write=False
        )

    def calc_mach(self):
        d = self.d

        d['MACHNO'], d['MACHNO_FLAG'] = sp_mach(
            d['Q_RVSM'], d['PS_RVSM'], flag=True
        )

        d.loc[d['MACHNO'] < 0.05, 'MACHNO'] = 0.05

    def calc_heating_correction(self):
        """
        Calculate a correction for heating from the deiced heater, which is
        required when PRTAFT_deiced_temp_flag = 1.

        The heating correction is required from the graphs of temperature
        vs Mach number in Rosemount Technical Reports 7597 and 7637.

        The required correction is stored in the HEATING_CORRECTION column
        of the instance dataframe.

        Requires MACHNO, PS_RVSM and Q_RVSM to be in the instance dataframe.
        """
        d = self.d

        # Heating correction is a function of Mach #, static pressure and
        # pitot-static pressure.
        corr = 0.1 * (
            np.exp(
                np.exp(
                    1.171 + (np.log(d['MACHNO']) + 2.738) *
                    (-0.000568 * (d['Q_RVSM'] + d['PS_RVSM']) - 0.452)
                )
            )
        )

        # Heating flag is at 1 Hz, so we need to fill the 32 Hz version
        heating_flag = (
            d['PRTAFT_deiced_temp_flag'].fillna(method='pad').fillna(0)
        )

        # Correction not required when heater is not on
        corr.loc[heating_flag == 0] = 0

        # Store in the instance dataframe
        d['HEATING_CORRECTION'] = corr

    def calc_ndi_iat(self):
        """
        Calculate the non-deiced indicated air temperature, in Kelvin.

        This is a quadratic calibration from counts (CORCON_ndi_temp),
        using the calibration coefficients in CALNDT.
        """

        d = self.d

        _cals = self.dataset['CALNDT'][::-1]
        d['IAT_ND_R'] = np.polyval(_cals, d['CORCON_ndi_temp'])

        # Convert to Kelvin
        d['IAT_ND_R'] = celsius_to_kelvin(d['IAT_ND_R'])

    def calc_di_iat(self):
        """
        Calculate the deiced indicated air temperature, in Kelvin.

        This is a quadratic calibration from counts (CORCON_di_temp),
        using the calibration coefficients in CALDIT.
        """

        d = self.d

        _cals = self.dataset['CALDIT'][::-1]
        d['IAT_DI_R'] = np.polyval(_cals, d['CORCON_di_temp'])

        # Convert to kelvin and apply heating correction
        d['IAT_DI_R'] = celsius_to_kelvin(d['IAT_DI_R'])
        d['IAT_DI_R'] -= d['HEATING_CORRECTION']

    def calc_ndi_tat(self):
        """
        Calculate the non-deiced true air temperature, using
        ppodd.utils.calcs.true_air_temp.

        Sets: TAT_ND_R
        """
        d = self.d
        d['TAT_ND_R'] = true_air_temp(
            d['IAT_ND_R'], d['MACHNO'], self.dataset['TRFCTR'][1]
        )

    def calc_di_tat(self):
        """
        Calculate the deiced true air temperature, using
        ppodd.utils.calcs.true_air_temp.

        Sets: TAT_DI_R
        """
        d = self.d
        d['TAT_DI_R'] = true_air_temp(
            d['IAT_DI_R'], d['MACHNO'], self.dataset['TRFCTR'][0]
        )

    def flag_delta_t(self, threshold=1):
        """
        Create a flag which highlights areas where |TAT_ND_R-TAT_DI_R| exceeds
        a specified threshold.

        Kwargs:
            threshold [1]: the threshold above which to flag TAT data.

        Sets:
            DT_FLAG
        """
        d = self.d
        d['DT_FLAG'] = 0
        d.loc[np.abs(d['TAT_DI_R'] - d['TAT_ND_R']) > threshold, 'DT_FLAG'] = 1

    def process(self):
        """
        Entry point for postprocessing.
        """
        self.get_dataframe()
        self.calc_mach()
        self.calc_heating_correction()
        self.calc_ndi_iat()
        self.calc_ndi_tat()
        self.calc_di_iat()
        self.calc_di_tat()
        self.flag_delta_t()

        tat_nd = DecadesVariable(self.d['TAT_ND_R'], flag=DecadesBitmaskFlag)
        tat_di = DecadesVariable(self.d['TAT_DI_R'], flag=DecadesBitmaskFlag)

        iat_nd = DecadesVariable(self.d['IAT_ND_R'], flag=DecadesBitmaskFlag)
        iat_di = DecadesVariable(self.d['IAT_DI_R'], flag=DecadesBitmaskFlag)

        for tat in (tat_nd, tat_di):
            tat.flag.add_mask(self.d['MACHNO_FLAG'], 'mach_out_of_range')
            tat.flag.add_mask(
                self.d['DT_FLAG'],
                'discrepancy_between_temperatures'
            )
            self.add_output(tat)

        for iat in (iat_nd, iat_di):
            iat.flag.add_mask(self.d['MACHNO_FLAG'], 'mach_out_of_range')
            self.add_output(iat)

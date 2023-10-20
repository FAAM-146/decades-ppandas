"""
This module provides a postprocessing module for the rosemount temperature
probes, when fitted with a platinum resistance thermometer. See class docstring
for more information.
"""
# pylint: disable=invalid-name
import datetime

import numpy as np

from vocal.types import DerivedString, OptionalDerivedString

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades.attributes import DocAttribute
from ..exceptions import EM_CANNOT_INIT_MODULE
from ..utils.calcs import sp_mach, true_air_temp_variable
from ..utils.conversions import celsius_to_kelvin
from .base import PPBase, register_pp
from .shortcuts import _a, _o, _c, _z

MACH_VALID_MIN = 0.05


@register_pp('core')
class PRTTemperatures(PPBase):
    r"""
    For further details see the `FAAM Met. Handbook <https://doi.org/10.5281/zenodo.5846962>`_.

    Calculate indicated and true (static)  air temperatures from the deiced and
    non-deiced Rosemount housings, when fitted with platinum resistance
    thermometer sensors. Indicated temperatures are calculated with a
    polynomial transformation of the DLU signal, using calibration factors in
    constants variables ``CALDIT`` and ``CALNDT``, which incorporates
    the DLU and sensor calibrations.

    The deiced indicated temperature is subject to a heating correction term
    when the heater is active, given by

    .. math::
        \Delta T_{\text{IAT}} = \frac{1}{10}\exp{\left(\exp{\left(a +
        \left(\log\left(M\right)+b\right)\left(c\left(q+P\right)+d\right)\right)
        }\right),}

    where :math:`M` is the Mach number, :math:`q` is the dynamic pressure and
    :math:`P` the static pressure. The parameters :math:`a`, :math:`b`,
    :math:`c`, and :math:`s` are

    .. math::
        \left[1.171, 2.738, -0.000568, -0.452\right].

    True air temperatures are a function of indicated temperatures, Mach number
    and housing recovery factor, and are given by

    .. math::
        T_\text{TAT} = \frac{T_\text{IAT}}{1 + \left(0.2 R_f M^2\right)},

    where :math:`M` is the Mach number and :math:`R_f` the recovery factor.
    Recovery factors are calculated in the processing module `RecoveryFactor`.

    A flag is applied to the data when the Mach number is out of range. **Further
    flags may be added by standalone flagging modules**.
    """

    inputs = [
        'RM_RECFAC',                #  Recovery factors (Const)
        'CALDIT',                   #  Deiced calibrations (Const)
        'CALNDT',                   #  Non deiced calibrations (Const)
        'NDTSENS',                  #  Non deiced sensor type (Const)
        'DITSENS',                  #  Deiced sensor type (Const)
        'SH_GAMMA',
        'MACH',
        'ETA_ND',
        'ETA_DI',
        'PS_RVSM',                  #  Static pressure (derived)
        'Q_RVSM',                   #  Pitot-static pressure (derived)
        'CORCON_di_temp',           #  Deiced temperature counts (DLU)
        'CORCON_ndi_temp',          #  Non deiced temperature counts (DLU)
        'PRTAFT_deiced_temp_flag'   #  Deiced heater indicator flag (DLU)
    ]

    @staticmethod
    def test():
        """
        Return some dummy input data for testing.
        """
        return {
            'RM_RECFAC': ('const', {'DI': 1., 'ND': 1.}),
            'CALDIT': ('const', [0, 0, 0]),
            'CALNDT': ('const', [0, 0, 0]),
            'NDTSENS': ('const', [
                DocAttribute(value='12345A', doc_value=DerivedString),
                DocAttribute(value='plate', doc_value=DerivedString)
            ]),
            'DITSENS': ('const', [
                DocAttribute(value='12345A', doc_value=DerivedString),
                DocAttribute(value='plate', doc_value=DerivedString)
            ]),
            'PS_RVSM': ('data', _a(1000, 300, -1), 32),
            'Q_RVSM': ('data', 250*(_o(700)), 32),
            'SH_GAMMA': ('data', _o(700), 32),
            'MACH': ('data', .5*_o(700), 32),
            'ETA_ND': ('data', _z(700), 32),
            'ETA_DI': ('data', _z(700), 32),
            'CORCON_di_temp': ('data', _a(225, 245, .0286)*1000, 32),
            'CORCON_ndi_temp': ('data', _a(225, 245, .0286)*1000, 32),
            'PRTAFT_deiced_temp_flag': (
                'data', _c([_z(200), _o(300), _z(200)]), 32
            ),
            # Optional calibration info...
            'RM_CALINFO_DI_DATE': ('const', DocAttribute(
                value=datetime.date(2000, 1, 1), doc_value=OptionalDerivedString
            )),
            'RM_CALINFO_DI_INFO': ('const', DocAttribute(
                value='Calibrated in a lab', doc_value=OptionalDerivedString
            )),
            'RM_CALINFO_DI_URL': ('const', DocAttribute(
                value='https://some.url', doc_value=OptionalDerivedString
            )),
            'RM_CALINFO_ND_DATE': ('const', DocAttribute(
                value=datetime.date(2000, 1, 1), doc_value=OptionalDerivedString
            )),
            'RM_CALINFO_ND_INFO': ('const', DocAttribute(
                value='Calibrated in a lab', doc_value=OptionalDerivedString
            )),
            'RM_CALINFO_ND_URL': ('const', DocAttribute(
                value='https://some.url', doc_value=OptionalDerivedString
            ))
        }

    def declare_outputs(self):
        """
        Declare the outputs that are going to be written by this module.
        """

        sampling = ('Sensor housed in Rosemount Aerospace Inc. Type 102 '
                    '{nddi} Total Temperature Housing')

        try:
            _ = self.dataset['DITSENS']
            _ = self.dataset['NDTSENS']
        except KeyError as err:
            message = EM_CANNOT_INIT_MODULE.format(
                module_name=self.__class__.__name__,
                constants='DITSENS, NDTSENS'
            )

            raise RuntimeError(message) from err

        if self.test_mode or self.dataset['DITSENS'][1].lower() != 'thermistor':
            self.declare(
                'TAT_DI_R',
                units='K',
                frequency=32,
                long_name=('True air temperature from the Rosemount deiced '
                           'temperature sensor'),
                standard_name='air_temperature',
                sensor_type=self.dataset.lazy['DITSENS'][1],
                sensor_serial_number=self.dataset.lazy['DITSENS'][0],
                calibration_date=self.dataset.lazy['RM_CALINFO_DI_DATE'],
                calibration_information=self.dataset.lazy['RM_CALINFO_DI_INFO'],
                calibration_url=self.dataset.lazy['RM_CALINFO_DI_URL'],
                comment=sampling.format(nddi='deiced')
            )

            self.declare(
                'IAT_DI_R',
                units='K',
                frequency=32,
                long_name=('Indicated air temperature from the Rosemount deiced '
                           'temperature sensor'),
                sensor_type=self.dataset.lazy['DITSENS'][1],
                sensor_serial_number=self.dataset.lazy['DITSENS'][0],
                calibration_date=self.dataset.lazy['RM_CALINFO_DI_DATE'],
                calibration_information=self.dataset.lazy['RM_CALINFO_DI_INFO'],
                calibration_url=self.dataset.lazy['RM_CALINFO_DI_URL'],
                comment=sampling.format(nddi='deiced'),
                write=False
            )

        if self.test_mode or self.dataset['NDTSENS'][1].lower() != 'thermistor':
            self.declare(
                'TAT_ND_R',
                units='K',
                frequency=32,
                long_name=('True air temperature from the Rosemount non-deiced '
                           'temperature sensor'),
                standard_name='air_temperature',
                sensor_type=self.dataset.lazy['NDTSENS'][1],
                sensor_serial_number=self.dataset.lazy['NDTSENS'][0],
                calibration_date=self.dataset.lazy['RM_CALINFO_ND_DATE'],
                calibration_information=self.dataset.lazy['RM_CALINFO_ND_INFO'],
                calibration_url=self.dataset.lazy['RM_CALINFO_ND_URL'],
                comment=sampling.format(nddi='non-deiced')
            )

            self.declare(
                'IAT_ND_R',
                units='K',
                frequency=32,
                long_name=('Indicated air temperature from the Rosemount '
                           'non-deiced temperature sensor'),
                sensor_type=self.dataset.lazy['NDTSENS'][1],
                sensor_serial_number=self.dataset.lazy['NDTSENS'][0],
                calibration_date=self.dataset.lazy['RM_CALINFO_ND_DATE'],
                calibration_information=self.dataset.lazy['RM_CALINFO_ND_INFO'],
                calibration_url=self.dataset.lazy['RM_CALINFO_ND_URL'],
                comment=sampling.format(nddi='non-deiced'),
                write=False
            )


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
                    1.171 + (np.log(d['MACH']) + 2.738) *
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
        d['TAT_ND_R'] = true_air_temp_variable(
            d.IAT_ND_R, d.MACH, d.ETA_ND, d.SH_GAMMA
        )

    def calc_di_tat(self):
        """
        Calculate the deiced true air temperature, using
        ppodd.utils.calcs.true_air_temp.

        Sets: TAT_DI_R
        """
        d = self.d
        d['TAT_DI_R'] = true_air_temp_variable(
            d.IAT_DI_R, d.MACH, d.ETA_DI, d.SH_GAMMA
        )

    def process(self):
        """
        Entry point for postprocessing.
        """

        proc_ndt = self.dataset['NDTSENS'][1] != 'thermistor'
        proc_dit = self.dataset['DITSENS'][1] != 'thermistor'

        self.get_dataframe()

        if proc_dit:
            self.calc_heating_correction()
            self.calc_di_iat()
            self.calc_di_tat()

        if proc_ndt:
            self.calc_ndi_iat()
            self.calc_ndi_tat()

        tats = []
        iats = []

        if proc_ndt:
            tat_nd = DecadesVariable(self.d['TAT_ND_R'], flag=DecadesBitmaskFlag)
            iat_nd = DecadesVariable(self.d['IAT_ND_R'], flag=DecadesBitmaskFlag)
            tats.append(tat_nd)
            iats.append(iat_nd)

        if proc_dit:
            tat_di = DecadesVariable(self.d['TAT_DI_R'], flag=DecadesBitmaskFlag)
            iat_di = DecadesVariable(self.d['IAT_DI_R'], flag=DecadesBitmaskFlag)
            tats.append(tat_di)
            iats.append(iat_di)

        for at in tats + iats:
            at.flag.add_mask(
                self.d.MACH < MACH_VALID_MIN, 'mach_out_of_range',
                f'Mach number is below acceptable minimum of {MACH_VALID_MIN}'
            )
            self.add_output(at)

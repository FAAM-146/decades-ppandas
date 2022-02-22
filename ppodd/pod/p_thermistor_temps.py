import numpy as np
import pandas as pd

from scipy import interpolate
from scipy.interpolate import CubicSpline

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..exceptions import EM_CANNOT_INIT_MODULE
from ..utils.calcs import sp_mach, true_air_temp_variable
from ..utils.conversions import celsius_to_kelvin
from .base import PPBase, register_pp
from .shortcuts import *

MACH_VALID_MIN = 0.5


@register_pp('core')
class ThermistorV1Temperatures(PPBase):
    r"""
    Calculate indicated and true (static) air temperatures from the Rosemount
    temperature probes, for the V1 (prototype) thermistor circuit. Note that
    when using this circuit, only one of the housings may host a thermistor
    type sensor.

    * Processes the NPL calibrations to produce a resistance to probe
      temperature relationship. The NPL calibration is performed with two
      different applied voltages so that self-heating can be calculated.
      This allows the probe temperature to be inferred by adding the
      calibration chamber temperature to the self-heating in order to
      produce a useable probe temperature to resistance calibration.}
    * Processes the NPL calibrations for a temperature to dissipation
      constant relationship. The dissipation constant measured in the
      calibration chamber for a given temperature is used later in the
      removal of self-heating in flight.}
    * Calculates the indicated temperature of the thermistor sensor by
      applying the NPL calibration and the DECADES counts to voltage
      calibration.
    * Calculates the dissipation constant at that indicated temperature
      based on the calibration dissipation constant and the flight
      dissipation multiplier (as described in FAAM013004A).}
    * Calculates the in-flight self-heating for each measurement, based
      on the voltage measurements and in-flight dissipation constants.}
    * Produces an indicated temperature corrected for self-heating by
      subtracting the self-heating from the temperature of the sensor.}

    The deiced indicated temperature is subject to a heating correction term
    when the heater is active, given by

    .. math::
        \Delta T_{\text{IAT}} = \frac{1}{10}\exp{\left(\exp{\left(a +
        \left(\log\left(M\right)+b\right)\left(c\left(q+P\right)+d\right)\right)
        }\right),}

    where :math:`M` is the Mach number, :math:`q` is the dynamic pressure and
    :math:`P` the static pressure. The parameters :math:`a`, :math:`b`,
    :math:`c`, and :math:`d` are

    .. math::
        \left[1.171, 2.738, -0.000568, -0.452\right].

    True air temperatures are a function of indicated temperatures, Mach number
    and housing recovery factor, and are given by

    .. math::
        T_\text{TAT} = \frac{T_\text{IAT}}{1 + \left(0.2 R_f M^2\right)},

    where :math:`M` is the Mach number and :math:`R_f` the recovery factor.
    Recovery factors are currently considered constant, and are specified in the
    flight constants parameters ``RM_RECFAC/DI`` and ``RM\_RECFAC/ND``.

    A flag is applied to the data when the Mach number is out of range. **Further
    flags may be added by standalone flagging modules.**
    """

    inputs = [
        'RM_RECFAC',                #  Recovery factors (Const)
        'NDTSENS',                  #  Non deiced sensor type (Const)
        'DITSENS',                  #  Deiced sensor type (Const)
        'TH_DISS_MUL',              #  Dissipation multiplier (Const)
        'TH_RESISTANCE',            #  Resistances (Const)
        'TH_DECADES',               #  DECADES calibration (Const)
        'TH_CAL_TEMPS',             #  Thermistor cal. temps (Const)
        'TH_CAL_TEMPS_HI',          #  Thermistor cal. temps hi (Const)
        'TH_CAL_TEMPS_LO',          #  Thermistor cal. temps lo (Const)
        'TH_HIGH_VIN',              #  Thermistor high vin (Const)
        'TH_LOW_VIN',               #  Thermistor low vin (Const)
        'TH_HIGH_VOUT',             #  Thermistor high vout (Const)
        'TH_LOW_VOUT',              #  Thermistor low vout (Const)
        'CORCON_fast_temp',         #  Deiced temperature counts (DLU)
        'CORCON_padding1',          #  Non deiced temperature counts (DLU)
        'PRTAFT_deiced_temp_flag',  #  Deiced heater indicator flag (DLU)
        'PS_RVSM',                  #  RVSM static pressure (derived)
        'Q_RVSM',                   #  RVSM dynamic pressure (derived)
        'SH_GAMMA',
        'MACH',
        'ETA_DI',
        'ETA_ND'
    ]

    @staticmethod
    def test():
        return {
            'RM_RECFAC': ('const', {'DI': 1., 'ND': 1.}),
            'NDTSENS': ('const', [lambda: 'xxx', 'thermistor']),
            'DITSENS': ('const', [lambda: 'xxx', 'thermistor']),
            'TH_DISS_MUL': ('const', {'ND': 1.8, 'DI': 1.8}),
            'TH_CIRCUIT_TYPE': ('const', 'V1'),
            'TH_RESISTANCE': ('const', {
                'ND': {'RF': 1E5, 'RF1': 1E5, 'RF2': 3E5, 'RB1': 1E5, 'RB2': 3E5},
                'DI': {'RF': 1E5, 'RF1': 1E5, 'RF2': 3E5, 'RB1': 1E5, 'RB2': 3E5}
            }),
            'TH_DECADES': ('const', {
                'CHANNEL1': [6E-7, 1E-3],
                'CHANNEL2': [6E-7, 1E-3]
            }),
            'TH_CAL_TEMPS': ('const', {
                'ND': _a(-60, 60, 10), 'DI': _a(-60, 60, 10),
            }),
            'TH_CAL_TEMPS_HI': ('const', {
                'ND': _a(-60, 60, 10), 'DI': _a(-60, 60, 10)
            }),
            'TH_CAL_TEMPS_LO': ('const', {
                'ND': _a(-60, 60, 10), 'DI': _a(-60, 60, 10)
            }),
            'TH_HIGH_VIN': ('const', {
                'ND': [5.1 + i / 10 for i in range(12)],
                'DI': [5.1 + i / 10 for i in range(12)],
            }),
            'TH_LOW_VIN': ('const', {
                'ND': [3.5 + i / 10 for i in range(12)],
                'DI': [3.5 + i / 10 for i in range(12)]
            }),
            'TH_HIGH_VOUT': ('const', {
                'ND': _l(5, .1, 12),
                'DI': _l(5, .1, 12)
            }),
            'TH_LOW_VOUT': ('const', {
                'ND': _l(3.2, .1, 12),
                'DI': _l(3.2, .1, 12)}
            ),
            'PS_RVSM': ('data', _a(1000, 300, -1), 32),
            'Q_RVSM': ('data', 250*(_o(700)), 32),
            'CORCON_fast_temp': ('data', _a(225, 245, .0286)*1000, 32),
            'CORCON_padding1': ('data', _a(225, 245, .0286)*1000, 32),
            'PRTAFT_deiced_temp_flag': ('data', _c([_z(200), _o(300),
                                                    _z(200)]), 1),
            'MACH': ('data', .5 * _o(700), 32),
            'SH_GAMMA': ('data', 1.6 * _o(700), 32),
            'ETA_DI': ('data', _o(700), 32),
            'ETA_ND': ('data', _o(700), 32),
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
            raise RuntimeError(EM_CANNOT_INIT_MODULE.format(
                module_name=self.__class__.__name__,
                constants='DITSENS, NDTSENS'
            )) from err

        if self.dataset['DITSENS'][1].lower() == 'thermistor':
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

        if self.dataset['NDTSENS'][1].lower() == 'thermistor':
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

    def calc_mach(self):
        d = self.d

        d['MACHNO'], d['MACHNO_FLAG'] = sp_mach(
            d['Q_RVSM'], d['PS_RVSM'], flag=True
        )

        d.loc[d['MACHNO'] < 0.05, 'MACHNO_FLAG'] = 1
        d.loc[~np.isfinite(d['MACHNO']), 'MACHNO_FLAG'] = 1
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


    def process(self):
        """
        Entry point for postprocessing.
        """
        if self.dataset['TH_CIRCUIT_TYPE'].lower() != 'v1':
            return

        if self.dataset['DITSENS'][1].lower() == 'thermistor':
            NDDI = 'DI'
            recovery_factor = self.dataset['RM_RECFAC']['DI']
        elif self.dataset['NDTSENS'][1].lower() == 'thermistor':
            NDDI = 'ND'
            recovery_factor = self.dataset['RM_RECFAC']['ND']
        else:
            return

        self.get_dataframe()
        self.calc_mach()
        self.calc_heating_correction()

        probe_temperature = np.zeros((12,))
        self_heating_out = np.zeros((12,))
        cal_temps_out = np.zeros((12,))
        resistance_out = np.zeros((12,))
        probe_temperature_n = np.zeros((12,))
        self_heating_out_n = np.zeros((12,))
        cal_temps_out_n = np.zeros((12,))
        resistance_out_n = np.zeros((12,))
        dissipation_out = np.zeros((12,))

        high_vin_ideal=5.0
        low_vin_ideal = high_vin_ideal / 2.0**0.5

        flight_dissipation_multiplier = self.dataset['TH_DISS_MUL']['ND']
        rf = self.dataset['TH_RESISTANCE'][NDDI]['RF']
        rf1 = self.dataset['TH_RESISTANCE'][NDDI]['RF1']
        rb1 = self.dataset['TH_RESISTANCE'][NDDI]['RB1']
        rf2 = self.dataset['TH_RESISTANCE'][NDDI]['RF2']
        rb2 = self.dataset['TH_RESISTANCE'][NDDI]['RB2']

        cal_temps = np.array(self.dataset['TH_CAL_TEMPS'][NDDI])
        cal_temps_high = np.array(self.dataset['TH_CAL_TEMPS_HI'][NDDI])
        cal_temps_low = np.array(self.dataset['TH_CAL_TEMPS_LO'][NDDI])
        high_vin = np.array(self.dataset['TH_HIGH_VIN'][NDDI])
        low_vin = np.array(self.dataset['TH_LOW_VIN'][NDDI])
        high_vout = np.array(self.dataset['TH_HIGH_VOUT'][NDDI])
        low_vout = np.array(self.dataset['TH_LOW_VOUT'][NDDI])

        adj_high_vout = high_vout * high_vin_ideal / high_vin
        adj_low_vout = low_vout * low_vin_ideal / low_vin

        padding_1_m = self.dataset['TH_DECADES']['CHANNEL1'][0]
        padding_1_c = self.dataset['TH_DECADES']['CHANNEL1'][1]
        corcon_fast_temp_m = self.dataset['TH_DECADES']['CHANNEL2'][0]
        corcon_fast_temp_c = self.dataset['TH_DECADES']['CHANNEL2'][1]

        # normalise all the points evaluted for the  two ideal exitation
        # voltages, 5 and 3.5355 V 
        normalised_adj_high_vout = adj_high_vout / 5.0
        normalised_adj_low_vout = adj_low_vout / 3.5355

        # Do a cubic spline to the normalised data, ie temperature on x axis,
        # normalised signal on y-coordinates. Do it every 1 K from -60 to 50 C
        invented_temperature = -65 + 120 * (np.arange(12000) / 12000.0)

        cs_high = CubicSpline(cal_temps_high, normalised_adj_high_vout, bc_type='natural')
        interpolated_high = cs_high(invented_temperature)

        cs_low = CubicSpline(cal_temps_low, normalised_adj_low_vout, bc_type='natural')
        interpolated_low = cs_low(invented_temperature)

        # Find the difference in normalised voltage for each temperature
        diff = interpolated_high - interpolated_low
        differential_high = cs_high(invented_temperature, 1)
        differential_low = cs_low(invented_temperature, 1)

        # Divide [differences in normalised signal volages] by [normalised
        # voltages] to get self heating. use differential_high as it's very
        # similar to differential_low
        self_heating_cal = 2.0 * diff / differential_high

        # Work out the resistances of the thermistors at high (5V) and low
        # (5 / root2 V)
        rt_high = 1.0 / (
            (high_vin_ideal - adj_high_vout) / (adj_high_vout * rf) - (1. /10.e6)
        )

        rt_low = 1.0 / (
            (low_vin_ideal - adj_low_vout) / (adj_low_vout * rf) - (1.0/10.0e6)
        )

        rt_interpolated = 1.0 / (
            (high_vin_ideal - 5.0 * interpolated_high) / (5.0 * interpolated_high * rf) - (1. / 10.0e6)
        )

        dissipation_constant = (interpolated_high * 5.0)**2.0 / (rt_interpolated * self_heating_cal)

        aaa = [
            np.where(np.abs(invented_temperature-i)<0.005)[0][0] for i in cal_temps
        ]

        for i in range(12):
            probe_temperature[i] = cal_temps[i] + self_heating_cal[aaa[i]]
            self_heating_out[i] = self_heating_cal[aaa[i]]
            cal_temps_out[i] = cal_temps[i]
            resistance_out[i] = rt_high[i]
            dissipation_out[i] = dissipation_constant[aaa[i]]

        fit_k0_t = np.polyfit(
            celsius_to_kelvin(probe_temperature[1:11]), dissipation_out[1:11], 3
        )

        k0_d = fit_k0_t[0]
        k0_c = fit_k0_t[1]
        k0_b = fit_k0_t[2]
        k0_a = fit_k0_t[3]

        corcon_padding1 = self.d['CORCON_padding1']
        corcon_fast_temp = self.d['CORCON_fast_temp']

        vout_ndi = (padding_1_m * corcon_padding1) + padding_1_c
        vout_di = (corcon_fast_temp_m * corcon_fast_temp) + corcon_fast_temp_c

        if NDDI == 'ND':
            vin = (vout_di * rf2 + rb2 * vout_di) / rb2
            r_therm = rf1 * rb1 / (
                rb1 * (vout_di / vout_ndi) * ((rf2 + rb2) / rb2) - (rb1 + rf1)
            )
            v2out_r = vout_ndi**2.0 / r_therm

        elif NDDI == 'DI':
            vin = (vout_ndi * rf1 + rb1 * vout_ndi) / rb1
            r_therm = rf2 * rb2 / (
                rb2 * (vout_ndi / vout_di) * ((rf1 + rb1) / rb1) - (rb2 + rf2)
            )
            v2out_r = vout_di**2.0 / r_therm

        else:
            raise ValueError('Invalid housing specification: {}'.format(NDDI))

        resistance_touse = resistance_out
        probetemp_touse = celsius_to_kelvin(probe_temperature)
        cs_touse = CubicSpline(
            resistance_touse[::-1],
            probetemp_touse[::-1],
            bc_type='natural'
        )

        it_therm = cs_touse(r_therm)

        # Not ideal to use it_therm, should use it_therm_ash, but this is
        # circular so can't do that, this won't make much difference
        k0_calc = k0_d * it_therm**3.0 + k0_c * it_therm**2.0 + k0_b * it_therm + k0_a
        dissipation_constant_odr = k0_calc * flight_dissipation_multiplier
        fitted_sh_odr = v2out_r / dissipation_constant_odr

        it_therm_ash_odr = it_therm - fitted_sh_odr

        mach = self.d['MACH']
        eta = self.d[f'ETA_{NDDI}']

        tt_therm_ash_odr = true_air_temp_variable(
            it_therm_ash_odr, mach, eta, self.d['SH_GAMMA']
        )

        NDDI = [NDDI]
        if self.test_mode:
            NDDI = ['ND', 'DI']

        for housing in NDDI:
            iat = DecadesVariable(
                it_therm_ash_odr, name='IAT_{}_R'.format(housing),
                flag=DecadesBitmaskFlag
            )

            tat = DecadesVariable(
                tt_therm_ash_odr, name='TAT_{}_R'.format(housing),
                flag=DecadesBitmaskFlag
            )

            for at in (iat, tat):
                at.flag.add_mask(
                    self.d['MACHNO_FLAG'], 'mach_out_of_range'
                )

            self.add_output(iat)
            self.add_output(tat)

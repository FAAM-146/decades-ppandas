"""
This module provides a postprocessing module for the Buck CR2 hygrometer,
BuckCR2. See the class docstring for further information.
"""
# pylint: disable=invalid-name

import warnings

import numpy as np
import pandas as pd

from scipy.optimize import fsolve

from ..decades import DecadesVariable
from .base import PPBase
from .shortcuts import _o, _z


class BuckCR2(PPBase):
    r"""
    This documentation adapted from FAAM document FAAM010015A (H. Price, 2016).

    The core processed data for the Buck CR2 includes the mirror temperature
    and the volume mixing ratio of water vapour. Prior to September 2016, the water
    vapour pressure was calculated using using the parameterisation given by Hardy
    (1998), which is based on the ITS-90 formulations. In September 2016, the data
    processing was updated to use the Murphy and Koop (2000) parameterisation for
    water vapour pressure. The vapour pressure over liquid water is now calculated
    according to

    .. math::
        \ln(p_{\text{liq}}) &= 54.842763 - \frac{6763.22}{T} - 4.21\ln{T} +0.000367 T\\
                &+ \tanh\left(0.0415\left(T - 218.8\right)\right)\left(53.878
        - \frac{331.22}{T} - 9.44523 \ln T + 0.014025 T \right),

    valid for :math:`123 < T < 332` K. The vapour pressure over ice is
    calculated as follows:

    .. math::
        \ln(p_\text{ice}) = 9.550426 - \frac{5723.265}{T} + 3.53068 \ln T
        -0.00728332 T,

    valid above 110 K. The water vapour pressure inside the instrument,
    :math:`p_{\text{H}_2\text{O,Buck}}` is calculated using either equation 1 or 2
    depending on whether a dew point or a frost point has been observed. Above
    273.15 K, a dew point is clearly observed. Below 243.15 K, we can be
    confident that a frost point is being measured. Between 243.15 and 273.15 K,
    a dew point is assumed if the mirror has not been below 243.15 K since it
    was last above 273.15 K and it has been below 273.15 K for less than ten
    minutes. If the mirror temperature is within the supercooled water regime
    but has been below 243.15 K since it was last at 273.15 K, or it has been
    below 273.15 K for more than ten minutes, a frost point is assumed. The fact
    that these are assumptions is reflected in the measurement uncertainty,
    described below.

    The vapour pressure is converted to volume mixing ratio,
    :math:`r_{\text{H}_2\text{O}}`, as follows:

    .. math::
        r_{\text{H}_2\text{O}} = \frac{e_f
        p_{\text{H}_2\text{O,Buck}}}{p_\text{Buck} - e_f
        p_{\text{H}_2\text{O,Buck}}},

    where :math:`e_f` is the enhancement factor given by Hardy (1998) and
    :math:`p_\text{Buck}` is the air pressure inside the instrument.

    A corrected dew or frost point is calculated, which is slightly different
    to the mirror temperature, correcting for the difference between the
    pressure inside the insturment and the static air pressure outside the
    aircraft. The water vapour pressure outside the aircraft is

    .. math::
        p_{\text{H}_2\text{o,outside}} = \frac{p_s r_{\text{H}_2\text{O}}}{e_f +
        e_f r_{\text{H}_2\text{O}}},

    where :math:`p_s` is the static air pressure. A dew or frost point is then
    calculated using the Murphy and Koop (2005) parameterisation. Frost points
    are calculated using an equation given in the paper:

    .. math::
        T_\text{frost,outside} =
        \frac{1.814625\ln{p_{\text{H}_2\text{O,outside}}} + 6190.134}{29.120 -
        \ln{p_{\text{H}_2\text{O,outside}}}}.

    In the the absence of an equation to calculate dew point, a numerical
    solving routine is used to find :math:`T_\text{dew,outside}` from Equation
    1.

    The uncertainty associated with the Buck CR2 measurements have several
    sources

    * The uncertainty associated with the calibration performed at NPL (where
      applicable). This is derived from the NPL expanded uncertainty and fit to
      a power law.
    * The repeatability of the calibration. This is derived from the NPL
      calibration measurements of different dew points and fit to a power law.
    * The response time of the instrument and the atmospheric variability. This
      is based on an assessment of the standard deviation of subsequent readings
      to give an indication of atmospheric variability.
    * The uncertainty involved in the interpolation of data between calibration
      datapoints. This is a function of temperature, increasing below 233.15 K.
    * The bias associated with the uncertainty about whether there is water or
      ice on the mirror between 243.15 and 273.15 K. This is calculated using a
      flagging scheme according to the temperature history of the mirror.

    These are combined to produce one uncertainty value for the mirror
    temperature, which may be propagated through to the volume mixing ratio
    and the pressure-corrected dew or frost point.
    """

    inputs = [
        'BUCK',
        'AERACK_buck_ppm',
        'AERACK_buck_mirr_temp',
        'AERACK_buck_pressure',
        'AERACK_buck_dewpoint_flag',
        'AERACK_buck_mirr_cln_flag',
        'PS_RVSM'
    ]

    @staticmethod
    def test():
        """
        Provide some dummy input data for testing purposes.
        """
        return {
            'BUCK': ('const', [0, 1]),
            'AERACK_buck_ppm': ('data', 2000 * _o(100), 1),
            'AERACK_buck_mirr_temp': ('data', -10 * _o(100), 1),
            'AERACK_buck_pressure': ('data', 800 * _o(100), 1),
            'AERACK_buck_dewpoint_flag': ('data', _o(100), 1),
            'AERACK_buck_mirr_cln_flag': ('data', _z(100), 1),
            'PS_RVSM': ('data', 800 * _o(100), 32)
        }

    def declare_outputs(self):
        """
        Declare the outputs produced by this module.
        """

        manufacturer = 'Buck Research Instruments'
        model = 'CR2 Chilled Mirror Hygrometer'

        self.declare(
            'VMR_CR2',
            units='ppmv',
            frequency=1,
            long_name=('Water vapour volume mixing ratio measured by the Buck '
                       'CR2'),
            instrument_manufacturer=manufacturer,
            instrument_model=model,
            instrument_serial_number=self.dataset.lazy['BUCK_SN']
        )

        self.declare(
            'VMR_C_U',
            units='ppmv',
            frequency=1,
            long_name=('Uncertainty estimate for water vapour volume mixing '
                       'ratio measured by the Buck CR2'),
            instrument_manufacturer=manufacturer,
            instrument_model=model,
            instrument_serial_number=self.dataset.lazy['BUCK_SN']
        )

        self.declare(
            'TDEW_CR2',
            units='degK',
            frequency=1,
            long_name='Mirror Temperature measured by the Buck CR2 Hygrometer',
            standard_name='dew_point_temperature',
            instrument_manufacturer=manufacturer,
            instrument_model=model,
            instrument_serial_number=self.dataset.lazy['BUCK_SN']
        )

        self.declare(
            'TDEW_C_U',
            units='degK',
            frequency=1,
            long_name='Uncertainty estimate for Buck CR2 Mirror Temperature',
            instrument_manufacturer=manufacturer,
            instrument_model=model,
            instrument_serial_number=self.dataset.lazy['BUCK_SN']
        )

        self.declare(
            'TDEWCR2C',
            units='degK',
            frequency=1,
            long_name=('Corrected dew point temperature measured by the Buck '
                       'CR2 Hygrometer'),
            standard_name='dew_point_temperature',
            instrument_manufacturer=manufacturer,
            instrument_model=model,
            instrument_serial_number=self.dataset.lazy['BUCK_SN']
        )

    @staticmethod
    def calc_uncertainty(buck_mirr_temp, buck_pressure,
                         buck_mirr_control):
        """
        Calculate and return the uncertainties from for the Buck parameters.

        Args:
            buck_mirr_temp: a timeseries of the buck mirror temperature
            buck_pressure: a timeseries of the internal pressure of the buck
            buck_mirr_control: a timeseries of the buck mirror control signal.

        Returns:
            buck_unc_k: the buck uncertainty.
        """
        # pylint: disable=too-many-locals, too-many-statements

        n = buck_mirr_temp.size
        buck_unc_c = np.zeros(n)
        buck_unc_r = np.zeros(n)
        buck_unc_t = np.zeros(n)
        buck_unc_i = np.zeros(n)
        buck_unc_b = np.zeros(n)
        buck_unc_k = np.zeros(n)

        buck_unc_temp = np.zeros(n)*np.nan

        for i in range(0, n):
            # Calibration Uncertainty
            Uc = 0.02 + 5E+27 * buck_mirr_temp[i]**-12.5
            buck_unc_c[i] = Uc

            # Repeatability
            Ur = 0.01 + 4E+19 * buck_mirr_temp[i]**-9.0
            buck_unc_r[i] = Ur

            if buck_mirr_temp[i] > 248.0:
                lag = 8
            else:
                lag = np.ceil(2e+29 * buck_mirr_temp[i]**-11.902)

            if not np.isfinite(lag):
                lag = 8

            lag = int(lag)

            if lag != np.nan:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fwdUt = np.nanstd(buck_mirr_temp[i:i+lag])
                    backUt = np.nanstd(buck_mirr_temp[i-lag:i])
            else:
                fwdUt, backUt = 0, 0

            if buck_pressure[i] > 0.0:
                Ut = np.max([fwdUt, backUt])
                buck_unc_t[i] = Ut
                buck_unc_temp[i] = Ut
            else:
                Ut = 0.0

            if buck_mirr_temp[i] > 233.15:
                Ui = 0.025
            else:
                Ui = -0.0044 * buck_mirr_temp[i] + 1.051
            buck_unc_i[i] = Ui

            # Bias uncertainty depending on knowledge of mirror state
            if buck_mirr_control[i] < 2:
                Ub = 0
                buck_unc_b[i] = 0

            if buck_mirr_control[i] == 2:

                lnesw = np.log(611.2) + (
                    17.62 * (buck_mirr_temp[i] - 273.15)
                ) / (243.12 + buck_mirr_temp[i] - 273.15)

                dpi = 273.15 + (
                    272.0 * (lnesw - np.log(611.2))
                    / (22.46 - (lnesw - np.log(611.2)))
                )

                buck_unc_b[i] = dpi-buck_mirr_temp[i]
                Ub = dpi-buck_mirr_temp[i]

            Uc = buck_unc_c[i]
            Ur = buck_unc_r[i]
            Ut = buck_unc_t[i]
            Ui = buck_unc_i[i]
            Ub = buck_unc_b[i]
            Uk = buck_unc_k[i]

            buck_unc_k[i] = 2 * np.sqrt(Uc**2 + Ur**2 + Ut**2 + Ui**2 + Ub**2)

        ix = np.where(buck_mirr_control[i] == 3)[0]
        buck_unc_k[ix] = np.nan

        return buck_unc_k

    @staticmethod
    def get_buck_mirror_ctl(buck_mirr_temp):
        """
        Calc the buck mirror control signal, given the mirror temperature.

        Args:
            buck_mirr_temp: a timeseries of the buck mirror temperature

        Returns:
            buck_mirror_control: the a derived mirror control signal.
        """
        # pylint: disable=too-many-branches, too-many-statements

        interval = 30
        recovery = 0
        mirrormin = 0
        mirrormax = 0
        DTmax = 0
        DTmin = 0
        DT = 0
        timing = 0

        buck_mirror_control = np.zeros(buck_mirr_temp.size, dtype=np.int)-9999
        for i in range(interval+1, buck_mirr_temp.size-interval-1):
            DT = np.mean(
                buck_mirr_temp[i:i+interval] - buck_mirr_temp[i-1:i+interval-1]
            )

            if buck_mirr_temp[i] < 220:
                DTmax = 1 / (220 * 0.0172438 - 3.6602) * 0.5
            else:
                DTmax = 1 / (buck_mirr_temp[i] * 0.0172438 - 3.6602) * 0.5

            if buck_mirr_temp[i] > 290:
                DTmin = 1 / (290 * 0.041044 - 12.232) * 0.5
            else:
                DTmin = 1 / (buck_mirr_temp[i] * 0.041044 - 12.232) * 0.5

            buck_mirror_control[i] = 2

            # Make a first cut at guessing the mirror state -
            #   0=water (above 273K),
            #   1=ice (when the mirror has been cold and then not above zero)
            #   2=not known.
            # these will be used to calculate the uncertainty
            # owing to mirror state.
            if buck_mirr_temp[i] > 273.15:
                buck_mirror_control[i] = 0

            if buck_mirr_temp[i] < 243.15:
                mirrormin = 1
                mirrormax = 1

            if buck_mirr_temp[i] > 273.15:
                timing = 0
            else:
                timing += 1

            if buck_mirr_temp[i] > 243.15:
                if mirrormin > 0:
                    if buck_mirr_temp[i] < 273.15:
                        mirrormax = 1
                    else:
                        mirrormin = 0
                        mirrormax = 0

            if mirrormin > 0:
                if mirrormax > 0:
                    buck_mirror_control[i] = 1
                else:
                    buck_mirror_control[i] = 2

            if timing > 600:
                buck_mirror_control[i] = 1

            # If Mirror Delta T outside acceptable range then flag as 3.
            # Start an 80s counter (320 4hz cycles)(recovery) following flag,
            # and only unflag when this has expired

            if DT > DTmax:
                buck_mirror_control[i] = 3
                recovery = 80
            else:
                if recovery > 0:
                    buck_mirror_control[i] = 3
                    recovery -= 1

            if DT < DTmin:
                buck_mirror_control[i] = 3
                recovery = 80
            else:
                if recovery > 0:
                    buck_mirror_control[i] = 3
                    recovery -= 1

        return buck_mirror_control

    @staticmethod
    def get_vp_coeff(buck_mirror_ctl):
        """
        Get the correct vapour pressure coefficients, for water or ice, given
        the derived mirror control signal.

        Args:
            buck_mirror_ctl: the derived buck mirror control signal

        Returns:
            result: coefficients for the calculation of vapour pressure with
                respect to either ice or water, depending on the control
                signal.
        """
        rows = buck_mirror_ctl.size
        result = np.zeros((rows, 11), dtype=np.float32)

        ice_coeff = [
            9.550426, -5723.265, 3.53068, -0.00728332,
            -9999.9, -9999.9, -9999.9, -9999.9, -9999.9, -9999.9, 0
        ]

        wat_coeff = [
            54.842763, -6763.22, -4.210, 0.000367, 0.0415, -218.8, 53.878,
            -1331.22, -9.44523, 0.014025, 1
        ]

        result[buck_mirror_ctl < 1, :] = wat_coeff
        result[buck_mirror_ctl >= 1, :] = ice_coeff
        result[buck_mirror_ctl > 2, :] = np.nan

        return result

    @staticmethod
    def get_enhance_coeff(buck_mirror_ctl):
        """
        Get the correct enhance coefficients, for water or ice, given
        the derived mirror control signal.

        Args:
            buck_mirror_ctl: the derived buck mirror control signal

        Returns:
            result: coefficients for the calculation of enhance with
                respect to either ice or water, depending on the control
                signal.
        """

        result = np.zeros((buck_mirror_ctl.size, 8), dtype=np.float32)

        ice_coeff = [
            -6.0190570E-2, 7.3984060E-4, -3.0897838E-6, 4.3669918E-9,
            -9.4868712E+1, 7.2392075E-1, -2.1963437E-3, 2.4668279E-6
        ]

        wat_0to100_coeff = [
            -1.6302041E-1, 1.8071570E-3, -6.7703064E-6, 8.5813609E-9,
            -5.9890467E+1, 3.4378043E-1, -7.7326396E-4, 6.3405286E-7
        ]

        wat_min50to0_coeff = [
            -5.5898100E-2, 6.7140389E-4, -2.7492721E-6, 3.8268958E-9,
            -8.1985393E+1, 5.8230823E-1, -1.6340527E-3, 1.6725084E-6
        ]

        result[buck_mirror_ctl < 1, :] = ice_coeff
        result[
            (buck_mirror_ctl >= 1) & (buck_mirror_ctl != 3), :
        ] = wat_0to100_coeff
        result[buck_mirror_ctl == 3, :] = np.nan

        return result

    def calc_vp(self, buck_mirr_temp, buck_mirror_ctl, buck_unc_k=None):
        """
        Calculate vapour pressure.

        Args:
            buck_mirr_temp: a timeseries of te buck mirror temperature
            buck_mirror_ctl: the derived mirror control signal

        Kwargs:
            buck_unc_k: buck uncertainty estimate.

        Returns:
            result: vapour pressure, calculated by ???
        """
        if not hasattr(buck_unc_k, 'size'):
            n = buck_mirr_temp.size
            buck_unc_k = np.zeros(n, dtype=np.float32)

        c = self.get_vp_coeff(buck_mirror_ctl)

        result = np.exp(
            c[:, 0] + c[:, 1] / (buck_mirr_temp + buck_unc_k)
            + c[:, 2] * np.log(buck_mirr_temp + buck_unc_k)
            + c[:, 3] * (buck_mirr_temp + buck_unc_k)
            + c[:, 10] * (np.tanh(c[:, 4] * (
                buck_mirr_temp + buck_unc_k + c[:, 5]
            )))
            * (c[:, 6] + c[:, 7] / (buck_mirr_temp + buck_unc_k)
                + c[:, 8] * np.log(buck_mirr_temp + buck_unc_k)
                + c[:, 9] * (buck_mirr_temp + buck_unc_k))
        )

        return result

    @staticmethod
    def calc_vmr(vp, enhance, buck_pressure):
        """
        Calculate volume mixing ration

        Args:
            vp: a timeseries of vapour pressure
            enhance: timeseries of enhancement factor
            buck_pressure: the internal pressure of the buck instrument.

        Returns:
            vmr: the volume mixing ratio of water in air.
        """
        vmr = vp / (buck_pressure * 100 - vp * enhance) * enhance * 10E5
        vmr[vmr < 0] = np.nan
        return vmr

    def calc_enhance_factor(self, vp_buck, buck_mirror_t, buck_pressure,
                            buck_mirror_ctl):
        """
        Calculate enhancement factors.

        Args:
            vp_buck: a timeseries of vapour pressure from the buck
            buck_mirror_t: a timeseries of mirror temperature from the buck
            buck_pressure: a timeseries of internal from the buck
            buck_mirror_ctl: the derived mirror control signal for the buck

        Returns:
            result: the enhancement factor.
        """

        c = self.get_enhance_coeff(buck_mirror_ctl)
        result = (np.exp(
            (1.0 - vp_buck / (buck_pressure * 100))
            * c[:, 0] + c[:, 1] * buck_mirror_t + c[:, 2] * buck_mirror_t**2
            + c[:, 3] * buck_mirror_t**3
        ) + ((buck_pressure * 100) / vp_buck - 1)
            * np.exp(
                c[:, 4] + c[:, 5] * buck_mirror_t + c[:, 6]
                * buck_mirror_t**2 + c[:, 7]*buck_mirror_t**3
            )
        )

        return result

    @staticmethod
    def get_flag(buck_mirr_flag, buck_dewpoint_flag):
        """
        Return flagging information of buck parameters.

        Args:
            buck_mirr_flag: a flag based on the state of the mirror
            buck_dewpoint_flag: a flag based on the dewpoint.

        Returns:
            flag: a composite flag built from the two inputs.
        """
        flag = np.zeros(buck_mirr_flag.size, dtype=np.int)
        flag[buck_dewpoint_flag == 0] = 1
        flag[buck_mirr_flag == 1] = 2
        flag[buck_dewpoint_flag == 2] = 3
        return flag

    @staticmethod
    def calc_tdew_corrected(buck_mirr_control, vmr_buck, ps_rvsm, enhance):
        """
        Calculate a corrected dewpoint temperature.

        Args:
            buck_mirr_control: the derived mirror control signal
            vmr_buck: volume mixing ratio from the buck
            ps_rvsm: static pressure.
            enhance: enhancement factors

        Returns:
            A dewpoint temperature corrected for pressure.
        """

        n = vmr_buck.size
        tfrost_corrected = np.zeros(n)
        result = np.zeros(n)
        vp_corrected = np.zeros(n)
        vp_corrected = 100 * ps_rvsm * vmr_buck / (
            enhance * 10e5 + enhance * vmr_buck
        )

        def tdew_function(tdew):
            """Dewpoint function."""
            return (
                54.842763 - 6763.22 / tdew - 4.210 * np.log(tdew)
                + 0.000367 * tdew + np.tanh(0.0415 * (tdew - 218.8))
                * (53.878 - 1331.22 / tdew - 9.44523 * np.log(tdew)
                   + 0.014025*tdew)
            ) - np.log(vp_here)

        tdew_corrected = np.zeros(n)
        vp_here = np.zeros(n)

        tdew_initial_guess = 300
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            for i in range(0, n):
                vp_here = vp_corrected[i]
                tdew_corrected[i] = fsolve(tdew_function, tdew_initial_guess)

        tfrost_corrected = (
            (1.814625 * np.log(vp_corrected) + 6190.134)
            / (29.120 - np.log(vp_corrected))
        )

        for j in range(0, n):
            if buck_mirr_control[j] < 1:
                result[j] = tdew_corrected[j]
            if buck_mirr_control[j] >= 1:
                result[j] = tfrost_corrected[j]
            if buck_mirr_control[j] > 2:
                result[j] = np.nan

        return result

    def process(self):
        """
        Processing entry hook.
        """
        # pylint: disable=too-many-locals

        self.get_dataframe(
            method='onto', index=self.dataset['AERACK_buck_ppm'].index
        )

        buck_mirr_temp = self.d['AERACK_buck_mirr_temp'].copy()
        buck_mirr_temp += 273.15
        buck_mirr_temp.loc[buck_mirr_temp == 273.15] = np.nan
        buck_mirr_temp.loc[buck_mirr_temp < 0] = np.nan

        ps_rvsm = self.d['PS_RVSM']

        p = np.poly1d(self.dataset['BUCK'][::-1])
        buck_mirr_temp = p(buck_mirr_temp)

        buck_pressure = self.d['AERACK_buck_pressure']
        buck_dewpoint_flag = self.d['AERACK_buck_dewpoint_flag']
        buck_mirr_cln_flag = self.d['AERACK_buck_mirr_cln_flag']

        buck_mirr_control = self.get_buck_mirror_ctl(buck_mirr_temp)

        vp_buck = self.calc_vp(buck_mirr_temp, buck_mirr_control)

        buck_unc_k = self.calc_uncertainty(
            buck_mirr_temp, buck_pressure, buck_mirr_control
        )

        vp_max = self.calc_vp(
            buck_mirr_temp, buck_mirr_control, buck_unc_k=buck_unc_k
        )

        enhance = self.calc_enhance_factor(
            vp_buck, buck_mirr_temp, buck_pressure, buck_mirr_control
        )

        vmr_buck = self.calc_vmr(vp_buck, enhance, buck_pressure)

        vmr_max = self.calc_vmr(vp_max, enhance, buck_pressure)

        vmr_unc = vmr_max - vmr_buck

        tdew_corrected = self.calc_tdew_corrected(
            buck_mirr_control, vmr_buck, ps_rvsm, enhance
        )

        # Get the flagging array
        flag = self.get_flag(buck_mirr_cln_flag, buck_dewpoint_flag)
        flag[~np.isfinite(buck_mirr_temp)] = 4

        _index = self.d.index

        # Create the Buck VMR output
        vmr_buck = DecadesVariable(
            pd.Series(vmr_buck, index=_index), name='VMR_CR2'
        )

        # Create the Buck VMR uncertainty output
        vmr_unc = DecadesVariable(
            pd.Series(vmr_unc, index=_index), name='VMR_C_U'
        )

        # Create the Buck TDEW output
        tdew_cr2 = DecadesVariable(pd.Series(
            buck_mirr_temp, index=_index), name='TDEW_CR2'
        )

        # Create the Buck TDEW uncertainty output
        tdew_c_u = DecadesVariable(
            pd.Series(buck_unc_k, index=_index), name='TDEW_C_U'
        )

        # Create the Buck corrected TDEW output
        tdewcr2c = DecadesVariable(
            pd.Series(tdew_corrected, index=_index), name='TDEWCR2C'
        )

        # Add flag to outputs and add to the dataset
        for dv in (vmr_buck, vmr_unc, tdew_cr2, tdew_c_u, tdewcr2c):
            dv.flag.add_meaning(0, 'data good')
            dv.flag.add_meaning(1, 'not controlling')
            dv.flag.add_meaning(2, 'mirror contaminated')
            dv.flag.add_meaning(3, 'in balance cycle')
            dv.flag.add_meaning(4, 'data missing')

            dv.flag.add_flag(flag)
            self.add_output(dv)

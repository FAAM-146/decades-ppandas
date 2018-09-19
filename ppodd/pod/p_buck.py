import warnings

import numpy as np
import pandas as pd

from scipy.optimize import fsolve

from ..decades import DecadesVariable
from .base import PPBase


class BuckCR2(PPBase):

    inputs = [
        'BUCK',
        'AERACK_buck_ppm',
        'AERACK_buck_mirr_temp',
        'AERACK_buck_pressure',
        'AERACK_buck_dewpoint_flag',
        'AERACK_buck_mirr_cln_flag',
        'PS_RVSM'
    ]

    def declare_outputs(self):

        self.declare(
            'VMR_CR2',
            units='ppmv',
            frequency=1,
            number=783,
            long_name=('Water vapour volume mixing ratio measured by the Buck '
                       'CR2'),
            standard_name='volume_mixing_ratio_of_water_in_air'
        )

        self.declare(
            'VMR_C_U',
            units='ppmv',
            frequency=1,
            number=784,
            long_name=('Uncertainty estimate for water vapour volume mixing '
                       'ratio measured by the Buck CR2')
        )

        self.declare(
            'TDEW_CR2',
            units='degK',
            frequency=1,
            number=785,
            long_name='Mirror Temperature measured by the Buck CR2 Hygrometer',
            standard_name='dew_point_temperature'
        )

        self.declare(
            'TDEW_C_U',
            units='degK',
            frequency=1,
            number=786,
            long_name='Uncertainty estimate for Buck CR2 Mirror Temperature'
        )

        self.declare(
            'TDEWCR2C',
            units='degK',
            frequency=1,
            long_name=('Corrected dew point temperature measured by the Buck '
                       'CR2 Hygrometer'),
            standard_name='dew_point_temperature'
        )

    def calc_uncertainty(self, buck_mirr_temp, buck_pressure,
                         buck_mirr_control):

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

            if (buck_pressure[i] > 0.0):
                Ut = np.max([fwdUt, backUt])
                buck_unc_t[i] = Ut
                buck_unc_temp[i] = Ut
            else:
                Ut = 0.0

            if(buck_mirr_temp[i] > 233.15):
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

    def get_buck_mirror_ctl(self, buck_mirr_temp):
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

    def get_vp_coeff(self, buck_mirror_ctl):
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

    def get_enhance_coeff(self, buck_mirror_ctl):
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

    def calc_vmr(self, vp, enhance, buck_pressure):
        vmr = vp / (buck_pressure * 100 - vp * enhance) * enhance * 10E5
        vmr[vmr < 0] = np.nan
        return vmr

    def calc_enhance_factor(self, vp_buck, buck_mirror_t, buck_pressure,
                            buck_mirror_ctl):

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

    def get_flag(self, buck_mirr_flag, buck_dewpoint_flag):
        flag = np.zeros(buck_mirr_flag.size, dtype=np.int)
        flag[buck_mirr_flag == 1] = 2
        flag[buck_dewpoint_flag == 2] = 3
        return flag

    def calc_tdew_corrected(self, buck_mirr_control, vmr_buck, ps_rvsm, enhance):
        n = vmr_buck.size
        tfrost_corrected = np.zeros(n)
        result = np.zeros(n)
        vp_corrected = np.zeros(n)
        vp_corrected = 100 * ps_rvsm * vmr_buck / (
            enhance * 10e5 + enhance * vmr_buck
        )

        def tdew_function(tdew):
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
        self.get_dataframe(
            method='onto', index=self.dataset['AERACK_buck_ppm'].index
        )

        buck_mirr_temp = self.d['AERACK_buck_mirr_temp']
        buck_mirr_temp += 273.15
        buck_mirr_temp[buck_mirr_temp == 273.15] = np.nan
        buck_mirr_temp[buck_mirr_temp < 0] = np.nan

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
        flag[~np.isfinite(buck_mirr_temp)] = 3

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
            dv.add_flag(flag)
            self.add_output(dv)

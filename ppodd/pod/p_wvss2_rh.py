"""
Provides a processing module which calculates a relative humidity from the
WVSS2 and Rosemount temperatures.
"""
# pylint: disable=invalid-name
import numpy as np
import pandas as pd

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..utils.constants import ZERO_C_IN_K
from .base import PPBase, register_pp
from .shortcuts import _o


@register_pp('core')
class WVSS2RH(PPBase):
    """
    Relative humidity from the WVSS2
    """

    inputs = [
        'PS_RVSM',
        'WOW_IND',
        'TAT_DI_R',
        'TAT_ND_R',
        'WVSS2F_VMR_C'
    ]

    @staticmethod
    def test():
        """
        Provide dummy input data for testing.
        """
        n = 100
        return {
            'PS_RVSM': ('data', 850 * _o(n), 32),
            'Q_RVSM': ('data', 75 * _o(n), 32),
            'CORCON_di_temp': ('data', _o(n), 32),
            'CORCON_ndi_temp': ('data', _o(n), 32)
        }

    def declare_outputs(self):
        """
        Declare outputs created by this module.
        """
        self.declare(
            'RH_LIQ',
            units='%',
            frequency=1,
            long_name=('Relative humidity wrt liquid water, derived from '
                       'corrected WVSS-II VMR and Rosemount temperatures'),
            standard_name='relative_humidity'
        )

        self.declare(
            'RH_ICE',
            units='%',
            frequency=1,
            long_name=('Relative humidity wrt ice water, derived from '
                       'corrected WVSS-II VMR and Rosemount temperatures'),
            standard_name='relative_humidity'
        )

    def process(self):
        """
        Processing entry hook.
        """
        self.get_dataframe()
        d = self.d

        temp = d['TAT_DI_R']
        press = d['PS_RVSM']
        wow = d['WOW_IND']
        wow.fillna(method='ffill', inplace=True)
        wow.fillna(method='bfill', inplace=True)
        d['WVSS2F_VMR_C'].interpolate(limit=32, inplace=True)

        temp_smooth = temp.rolling(64).mean()
        press_smooth = press.rolling(64).mean()

        temp_sens = self.dataset['DITSENS'][1]
        wvss2_lag = self.dataset['WVSS2_LAG'][temp_sens]
        wvss2_lag_std = self.dataset['WVSS2_LAG_STD'][temp_sens]

        lag_points = round(32 * wvss2_lag)
        lat_std_points = round(32 * wvss2_lag_std)

        vmr = np.roll(d['WVSS2F_VMR_C'], -lag_points)

        wvss2_vp = (
            ((vmr * 1e-6) * (press_smooth * 100)) / (1 + (vmr * 1e-6))
        )

        ice_coeff = [9.550426, -5723.265, 3.53068, -0.00728332, -9999.9,
                     -9999.9, -9999.9, -9999.9, -9999.9, -9999.9, 0.0]
        wat_coeff = [54.842763, -6763.22, -4.210,0.000367, 0.0415,- 218.8,
                     53.878, -1331.22, -9.44523, 0.014025, 1.0]

        sat_vp = lambda coeff, temp: np.exp(
            coeff[0] + coeff[1] / temp +
            coeff[2] * np.log(temp) +
            coeff[3] * temp +
            coeff[10] * (np.tanh(coeff[4] * (temp +
            coeff[5]))) * (coeff[6] + coeff[7] / temp +
            coeff[8] * np.log(temp) +
            coeff[9] * temp)
        )

        pure_sat_vp_liq = sat_vp(wat_coeff, temp_smooth)
        pure_sat_vp_ice = sat_vp(ice_coeff, temp_smooth)

        #### SOME UNCERTAINTY STUFF IN HERE ###

        ef_ice_coeff = [-6.0190570E-2, 7.3984060E-4, -3.0897838E-6,
                        4.3669918E-9, -9.4868712E+1, 7.2392075E-1,
                        -2.1963437E-3, 2.4668279E-6]

        ef_waterliq_coeff = [-1.6302041E-1, 1.8071570E-3, -6.7703064E-6,
                             8.5813609E-9, -5.9890467E+1, 3.4378043E-1,
                             -7.7326396E-4, 6.3405286E-7]

        ef_watersc_coeff = [-5.5898100E-2, 6.7140389E-4, -2.7492721E-6,
                            3.8268958E-9, -8.1985393E+1, 5.8230823E-1,
                            -1.6340527E-3, 1.6725084E-6]

        alpha_ice = np.polyval(ef_ice_coeff[:4][::-1], temp_smooth)
        beta_ice = np.exp(np.polyval(ef_ice_coeff[4:][::-1], temp_smooth))

        ef_sat_ice = np.exp(
            alpha_ice * (1. - pure_sat_vp_ice / (press_smooth * 100.))
            + beta_ice * ((press_smooth * 100.) / pure_sat_vp_ice - 1)
        )

        alpha_waterliq = np.polyval(ef_waterliq_coeff[:4][::-1], temp_smooth)
        beta_waterliq = np.exp(
            np.polyval(ef_waterliq_coeff[4:][::-1], temp_smooth)
        )

        ef_sat_waterliq = np.exp(
            alpha_waterliq * (1. - pure_sat_vp_liq / (press_smooth * 100.))
            + beta_waterliq*((press_smooth * 100.) / pure_sat_vp_liq - 1)
        )

        alpha_watersc = np.polyval(ef_watersc_coeff[:4][::-1], temp_smooth)
        beta_watersc = np.exp(
            np.polyval(ef_watersc_coeff[4:][::-1], temp_smooth)
        )

        ef_sat_watersc = np.exp(
            alpha_watersc * (1. - pure_sat_vp_liq / (press_smooth * 100.))
            + beta_watersc * ((press_smooth * 100.) / pure_sat_vp_liq - 1)
        )

        # Here we multiply the pure saturation vapour pressures by the
        # enhancement factors for each phase, to give the actual saturation
        # vapour pressures
        sat_vp_liq_tgt0 = ef_sat_waterliq *pure_sat_vp_liq
        sat_vp_liq_tlt0 = ef_sat_watersc *pure_sat_vp_liq
        sat_vp_ice = ef_sat_ice*pure_sat_vp_ice

        # Calculate the RH for each phase
        rh_liq_tgt0 = 100. * wvss2_vp / sat_vp_liq_tgt0
        rh_liq_tlt0 = 100. * wvss2_vp / sat_vp_liq_tlt0
        rh_ice = 100. * wvss2_vp / sat_vp_ice

        # Get rid of RHice where T > 0C, and all RH data on the ground
        rh_ice[temp_smooth > ZERO_C_IN_K] = np.nan
        rh_ice[wow == 1] = np.nan

        # Look at the temperature, if it's below 0 then the pure saturation
        # vapour pressure should be timesd by the supercooled water enhancement
        # factor (need to check this)
        rh_liq = np.nan * rh_ice
        mask_water = (temp_smooth > ZERO_C_IN_K)
        mask_ice = (temp_smooth <= ZERO_C_IN_K)
        rh_liq[mask_water] = rh_liq_tgt0[mask_water]
        rh_liq[mask_ice] = rh_liq_tlt0[mask_ice]
        rh_liq[wow == 1] = np.nan

        rh_ice_out = DecadesVariable(
            rh_ice.asfreq('1S'), name='RH_ICE', flag=DecadesBitmaskFlag
        )

        rh_liq_out = DecadesVariable(
            rh_liq.asfreq('1S'), name='RH_LIQ', flag=DecadesBitmaskFlag
        )

        self.add_output(rh_ice_out)
        self.add_output(rh_liq_out)


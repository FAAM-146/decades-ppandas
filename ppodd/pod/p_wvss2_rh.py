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
        'WVSS2F_VMR_C',
        'WVSS2F_VMR_C_CU'
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

        self.declare(
            'RH_LIQ_CU',
            units='%',
            frequency=1,
            long_name='Combined uncertainty estimate for RH_LIQ',
        )

        self.declare(
            'RH_ICE_CU',
            units='%',
            frequency=1,
            long_name='Combined uncertainty estimate for RH_ICE',
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
        lag_std_points = round(32 * wvss2_lag_std)

        vmr = np.roll(d['WVSS2F_VMR_C'], -lag_points)

        # ------------------------------------------------
        u_tat_core = 0.5
        u_pressure = 2.0

        roll_ve = pd.Series(np.roll(temp, -lag_std_points), index=temp.index)
        roll_va = pd.Series(np.roll(temp, lag_std_points), index=temp.index)
        u_tat_lag = np.maximum(
            (roll_ve.rolling(64).mean() - temp_smooth).abs(),
            (roll_va.rolling(64).mean() - temp_smooth).abs()
        )
        u_tat = (u_tat_lag**2 + u_tat_core**2)**0.5
        u_wvss2c = d['WVSS2F_VMR_C_CU']
        #=================================================

        wvss2_vp = (
            ((vmr * 1e-6) * (press_smooth * 100)) / (1 + (vmr * 1e-6))
        )

        # ----------------------------------------------------
        vmr_term = (
            (press_smooth * 100. / (1. + (vmr * 1e-6))**2)**2
            * (u_wvss2c * 1e-6)**2
        )

        psrvsm_term = ((vmr * 1e-6) / (1 + (vmr * 1e-6)))**2 * (u_pressure**2)**2
        delta_vp=(vmr_term+psrvsm_term)**0.5
        #=====================================================

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

        # ----------------------------------------------------
        x = temp_smooth.copy()
        u_liq = (
            wat_coeff[0] + wat_coeff[1] / x +
             wat_coeff[2] * np.log(x) + wat_coeff[3] * x +
             wat_coeff[10] * (
                 np.tanh(wat_coeff[4] * (x + wat_coeff[5]))
             ) * (wat_coeff[6] + wat_coeff[7] / x +
             wat_coeff[8] * np.log(x) + wat_coeff[9] * x)
        )

        dudx_liq = (
            - wat_coeff[1] + wat_coeff[2] * x
            + wat_coeff[3] * x**2.0
            + wat_coeff[10] * wat_coeff[ 4]* x
            * (1.0 / np.cosh(wat_coeff[4] * (wat_coeff[5] + x))**2)
            * (
                x * (wat_coeff[6] + wat_coeff[9] * x) + wat_coeff[7]
                + wat_coeff[8]* x *np.log(x)
            ) + wat_coeff[10] * np.tanh(wat_coeff[4] * (wat_coeff[5] + x))
            * (x * (wat_coeff[8] + wat_coeff[9] * x) - wat_coeff[7])
        ) / x**2

        u_ice = (
            + ice_coeff[0] + ice_coeff[1] / (temp_smooth)
            + ice_coeff[2] * np.log(temp_smooth) + ice_coeff[3] * temp_smooth
            + ice_coeff[10] * (
                np.tanh(ice_coeff[4] * (temp_smooth + ice_coeff[5]))
            ) * (
                ice_coeff[6] + ice_coeff[7] / temp_smooth
                + ice_coeff[8]*np.log(temp_smooth)+ ice_coeff[9]*(temp_smooth)
            )
        )

        dudx_ice = (
            - ice_coeff[1] + ice_coeff[2] * x + ice_coeff[3] * x**2.0
            + ice_coeff[10] * ice_coeff[4] * x * (
                1. / np.cosh(ice_coeff[4] * (ice_coeff[5] + x))**2.0
            ) * (
                x * (ice_coeff[6] + ice_coeff[9]* x)
                + ice_coeff[7] + ice_coeff[8] * x *np.log(x)
            ) + ice_coeff[10] * np.tanh(ice_coeff[4]* (ice_coeff[5] + x))
            * (x * (ice_coeff[8] + ice_coeff[9] * x) - ice_coeff[7])
        ) / x**2

        #d(pure saturation vapour pressure calculated for a liquid) / d (temp)
        dsatvdt_liq = dudx_liq * np.exp(u_liq)

        #d(pure saturation vapour pressure calculated for ice) / d (temp)
        dsatvdt_ice = dudx_ice * np.exp(u_ice)

        #these are the unceratinties in the saturation vapour pressures
        delta_satv_liq = dsatvdt_liq * u_tat
        delta_satv_ice = dsatvdt_ice * u_tat
        # ===========================================================

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

        #---------------------------------------------------------------
        delta_alpha_ice = (
            ef_ice_coeff[1] + 2.0 * ef_ice_coeff[2] * temp_smooth
            + 3.0 * ef_ice_coeff[3] * temp_smooth**2.0
        ) * u_tat

        delta_beta_ice = (
            ef_ice_coeff[5] +2.0 * ef_ice_coeff[6] * temp_smooth
            +3.0 * ef_ice_coeff[7] * temp_smooth**2.0
        ) * beta_ice * u_tat

        dfdalpha_ice = (
            1. - pure_sat_vp_ice / (press_smooth * 100.)
        ) * ef_sat_ice

        dfdbeta_ice = (
            (press_smooth * 100.) / pure_sat_vp_ice - 1
        ) * ef_sat_ice

        dfdes_ice = (
            -alpha_ice / (press_smooth * 100.0) - beta_ice
            * (press_smooth * 100.0) / (pure_sat_vp_ice**2)
        ) * ef_sat_ice

        dfdps_ice = (
            alpha_ice * pure_sat_vp_ice / ((press_smooth * 100)**2)
            + beta_ice / pure_sat_vp_ice
        ) * ef_sat_ice

        delta_ef_ice = (
            dfdalpha_ice**2 * delta_alpha_ice**2 + dfdbeta_ice**2
            * delta_beta_ice**2 + dfdes_ice**2 * delta_satv_ice**2
            + dfdps_ice**2 * (u_pressure*100)**2
        )**0.5

        delta_alpha_waterliq = (
            ef_waterliq_coeff[1] + 2. * ef_waterliq_coeff[2] * temp_smooth
            + 3.0 * ef_waterliq_coeff[3] * temp_smooth**2
        ) * u_tat

        delta_beta_waterliq = (
            ef_waterliq_coeff[5] + 2. * ef_waterliq_coeff[6] * temp_smooth
            + 3. * ef_waterliq_coeff[7] * temp_smooth**2
        ) * beta_waterliq * u_tat

        dfdalpha_waterliq = (
            1. - pure_sat_vp_liq / (press_smooth * 100.)
        ) * ef_sat_waterliq

        dfdbeta_waterliq = (
            (press_smooth * 100.) / pure_sat_vp_liq - 1.
        ) * ef_sat_waterliq

        dfdes_waterliq = (
            - alpha_waterliq / (press_smooth * 100.)
            - beta_waterliq * (press_smooth*100.0) / (pure_sat_vp_liq**2)
        ) * ef_sat_waterliq

        dfdps_waterliq = (
            alpha_waterliq * pure_sat_vp_liq / (
                (press_smooth * 100.)**2
            ) + beta_waterliq / pure_sat_vp_liq
        ) * ef_sat_waterliq

        delta_ef_waterliq = (
            dfdalpha_waterliq**2 * delta_alpha_waterliq**2
            + dfdbeta_waterliq**2 * delta_beta_waterliq**2
            + dfdes_waterliq**2 * delta_satv_liq**2
            + dfdps_waterliq**2 * (u_pressure * 100.)**2
        )**0.5

        delta_alpha_watersc = (
            ef_watersc_coeff[1] + 2. * ef_watersc_coeff[2] * temp_smooth
            + 3. * ef_watersc_coeff[3] * temp_smooth**2.
        ) * u_tat

        delta_beta_watersc = (
            ef_watersc_coeff[5] + 2. * ef_watersc_coeff[6] * temp_smooth +
            3. * ef_watersc_coeff[7] * temp_smooth**2.0
        ) * beta_watersc * u_tat

        dfdalpha_watersc = (
            1. - pure_sat_vp_liq / (press_smooth * 100.)
        ) * ef_sat_watersc

        dfdbeta_watersc = (
            (press_smooth * 100.) / pure_sat_vp_liq - 1.
        ) * ef_sat_watersc

        dfdes_watersc = (
            - alpha_watersc / (press_smooth * 100.) - beta_watersc
            * (press_smooth * 100.) / (pure_sat_vp_liq**2)
        ) * ef_sat_watersc

        dfdps_watersc = (
            alpha_watersc * pure_sat_vp_liq / ((press_smooth * 100.)**2.0)
            + beta_watersc / pure_sat_vp_liq
        ) * ef_sat_watersc

        delta_ef_watersc = (
            dfdalpha_watersc**2 * delta_alpha_watersc**2
            + dfdbeta_watersc**2. * delta_beta_watersc**2
            + dfdes_watersc**2. * delta_satv_liq**2.0
            + dfdps_watersc**2. * (u_pressure*100.0)**2.
        )**0.5
        #===============================================================

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

        #----------------------------------------------------------------
        # Water...
        vp_term_liq = (rh_liq / wvss2_vp)**2 * delta_vp**2
        f_term_liq_tgt0 = (rh_liq / ef_sat_waterliq)**2 * delta_ef_waterliq**2
        f_term_liq_tlt0 = (rh_liq / ef_sat_watersc)**2 * delta_ef_watersc**2
        es_term_liq = (rh_liq / pure_sat_vp_liq)**2 * delta_satv_liq**2
        u_rh_liq_tgt0 = (vp_term_liq + f_term_liq_tgt0 + es_term_liq)**0.5
        u_rh_liq_tlt0 = (vp_term_liq + f_term_liq_tlt0 + es_term_liq)**0.5
        u_rh_liq = np.nan * rh_liq
        u_rh_liq[mask_water] = u_rh_liq_tgt0[mask_water]
        u_rh_liq[mask_ice] = u_rh_liq_tlt0[mask_ice]
        u_rh_liq[wow == 1] = np.nan

        # Ice...
        delta_f_ice = ef_sat_ice - 1.
        vp_term_ice = (rh_ice / wvss2_vp)**2 * delta_vp**2
        f_term_ice = (rh_ice / ef_sat_ice)**2 * delta_ef_ice**2
        es_term_ice = (rh_ice / pure_sat_vp_ice)**2 * delta_satv_ice**2
        #this is the uncertainty in RHice
        u_rh_ice = (vp_term_ice + f_term_ice + es_term_ice)**0.5
        u_rh_ice[temp_smooth > ZERO_C_IN_K] = np.nan
        u_rh_ice[wow == 1] = np.nan
        #=================================================================

        rh_ice_out = DecadesVariable(
            rh_ice.asfreq('1S'), name='RH_ICE', flag=DecadesBitmaskFlag
        )

        rh_liq_out = DecadesVariable(
            rh_liq.asfreq('1S'), name='RH_LIQ', flag=DecadesBitmaskFlag
        )

        u_rh_ice_out = DecadesVariable(
            u_rh_ice.asfreq('1S'), name='RH_ICE_CU', flag=DecadesBitmaskFlag
        )

        u_rh_liq_out = DecadesVariable(
            u_rh_liq.asfreq('1S'), name='RH_LIQ_CU', flag=DecadesBitmaskFlag
        )


        self.add_output(rh_ice_out)
        self.add_output(rh_liq_out)
        self.add_output(u_rh_ice_out)
        self.add_output(u_rh_liq_out)

"""
Provides a processing module which calculates a relative humidity from the
WVSS2 and Rosemount temperatures.
"""
# pylint: disable=invalid-name
import numpy as np
import pandas as pd

from vocal.types import DerivedString

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..decades.attributes import DocAttribute
from ..utils.constants import ZERO_C_IN_K
from .base import PPBase, register_pp
from .shortcuts import _o

RH_VALID_MIN = 0
RH_VALID_MAX = 150


@register_pp('core')
class WVSS2RH(PPBase):
    """
    This module provides estimates of relative humidity with respect to both
    water and ice, along with their combined uncertainty estimates, derived from
    the WVSS2 water vapour mixing ratio and Rosemount temperatures.
    
    For full details of the methodology, see the
    `FAAM Met. Handbook <https://doi.org/10.5281/zenodo.5846962>`_.
    """

    inputs = [
        'PS_RVSM',
        'WOW_IND',
        'TAT_DI_R',
        'TAT_ND_R',
        'WVSS2F_VMR_C',
        'WVSS2F_VMR_C_CU',
        'DITSENS',
        'WVSS2_LAG',
        'WVSS2_LAG_STD'
    ]

    @staticmethod
    def test():
        """
        Provide dummy input data for testing.
        """
        n = 100
        return {
            'PS_RVSM': ('data', 850 * _o(n), 32),
            'WOW_IND': ('data', _o(n), 1),
            'TAT_DI_R': ('data', 250*_o(n), 32),
            'TAT_ND_R': ('data', 250*_o(n), 32),
            'WVSS2F_VMR_C': ('data', 5000*_o(n), 1),
            'WVSS2F_VMR_C_CU': ('data', 500*_o(n), 1),
            'DITSENS': ('const', [
                DocAttribute(value='1234', doc_value=DerivedString),
                'plate'
            ]),
            'WVSS2_LAG': ('const', {'plate': 1, 'loom': 1}),
            'WVSS2_LAG_STD': ('const', {'plate': .1, 'loom': .5}),
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
            coverage_content_type='auxiliaryInformation'
        )

        self.declare(
            'RH_ICE_CU',
            units='%',
            frequency=1,
            long_name='Combined uncertainty estimate for RH_ICE',
            coverage_content_type='auxiliaryInformation'
        )

    def process(self):
        """
        Processing entry hook.
        """
        self.get_dataframe()
        d = self.d

        try:
            nddi = self.dataset['WVSS2_RH_TEMP_SENSOR']
        except Exception:
            nddi = 'DI'

        temp = d[f'TAT_{nddi}_R']

        input_to_remove = 'ND' if nddi == 'DI' else 'DI'
        try:
            self.inputs.remove(f'TAT_{input_to_remove}_R')
        except ValueError:
            # The other temperature doesn't exist and therefore can't be removed.
            pass

        press = d['PS_RVSM']
        wow = d['WOW_IND']
        wow.fillna(method='ffill', inplace=True)
        wow.fillna(method='bfill', inplace=True)
        d['WVSS2F_VMR_C'].interpolate(limit=32, inplace=True)

        temp_smooth = temp.rolling(64).mean()
        press_smooth = press.rolling(64).mean()

        temp_sens = self.dataset[f'{nddi}TSENS'][1]
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

        psrvsm_term = ((vmr * 1e-6) / (1 + (vmr * 1e-6)))**2 * (u_pressure * 100)**2
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
        # Maximum expected uncertainty due to enhancement factors
        max_ef_unc = 0.00065

        delta_ef_watersc = max_ef_unc
        delta_ef_waterliq = max_ef_unc
        delta_ef_ice = max_ef_unc
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

        rh_liq = rh_liq.asfreq('1S')
        rh_ice = rh_ice.asfreq('1S')
        u_rh_liq = u_rh_liq.asfreq('1S')
        u_rh_ice = u_rh_ice.asfreq('1S')

        wow = d['WOW_IND'].fillna(method='bfill').fillna(method='ffill')
        wow = wow.reindex(rh_ice.index)

        rh_ice_out = DecadesVariable(
            rh_ice, name='RH_ICE', flag=DecadesBitmaskFlag
        )

        rh_liq_out = DecadesVariable(
            rh_liq, name='RH_LIQ', flag=DecadesBitmaskFlag
        )

        u_rh_ice_out = DecadesVariable(
            u_rh_ice, name='RH_ICE_CU', flag=None
        )

        u_rh_liq_out = DecadesVariable(
            u_rh_liq, name='RH_LIQ_CU', flag=None
        )

        on_ground = ('The aircraft is on the ground, as indicated by '
                     'weight-on-wheels')
        out_range = f'RH is outside the range [{RH_VALID_MIN}, {RH_VALID_MAX}]'

        rh_ice_out.flag.add_mask(wow, flags.WOW, on_ground)
        rh_liq_out.flag.add_mask(wow, flags.WOW, on_ground)

        liq_range_flag = (
            (rh_liq < RH_VALID_MIN) | (rh_liq > RH_VALID_MAX)
        )
        ice_range_flag = (
            (rh_ice < RH_VALID_MIN) | (rh_ice > RH_VALID_MAX)
        )

        rh_liq_out.flag.add_mask(liq_range_flag, flags.OUT_RANGE, out_range)
        rh_ice_out.flag.add_mask(ice_range_flag, flags.OUT_RANGE, out_range)

        self.add_output(rh_ice_out)
        self.add_output(rh_liq_out)
        self.add_output(u_rh_ice_out)
        self.add_output(u_rh_liq_out)

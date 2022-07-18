"""
This module provides a postprocessing module for the SEA WCM-2000 bulk water
content. See class docstring for more info.
"""
# pylint: disable=invalid-name, too-many-arguments, too-many-return-statements
# pylint: disable=too-many-locals
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..utils import slrs
from .base import PPBase, register_pp
from .shortcuts import _o, _r, _z

# Conversion from calories to joules [Woan00]_.
cal_to_J = 4.1868
J_to_cal = 1./cal_to_J

# Note that calories 20degC as used by Osborne et al. is 4.1819J
cal20_to_J = 4.1819
J_to_cal20 = 1./cal20_to_J

# Convert Kelvin to degrees C
C_to_K = lambda C: C + 273.15
K_to_C = lambda K: K - 273.15


def get_instr_fault_mask(el_temps=None, temp_limits=(100, 180)):
    """
    Return an instrument fault mask, based on upper and lower limits.

    Kwargs:
        el_temps - a pandas dataframe consisting of one or more columns of
                   element temperatures.
        temp_limits - a 2-tuple containing the minimum and maximum limits of
                      the valid temperature range.

    Returns:
        a single column pandas dataframe containing a boolean mask, which is
        True when any element is outside the valid temperature range.
    """

    return el_temps[
        (el_temps < temp_limits[0]) | (el_temps > temp_limits[1])
    ].fillna(0).astype(bool).any(axis=1)



def get_cloud_mask(el_temperature, var_thresh=0.6, set_temp=140, var_temp=2):
    """
    Return a cloud mask, indicating whether the aircraft is in or out of cloud.
    Assumned in cloud whenever the temp range in a rolling window exceeds a
    specified value, or where the temperature in that window exceeds a
    specified low or high value.

    Args:
        el_temperature - a pd.Series of element temperature

    Kwargs:
        var_thresh - the variability of el_temperature allowable in the window
                     before being marked as cloud.
        set_temp - the element temperature set on the SEA controller.
        var_temp - the deviation from set_temp allowable before being marked as
                   in cloud

    Returns:
        _mask - a single column dataframe containing a cloud mask, where True
                indicates in-cloud.
    """

    t_lo = set_temp - var_temp
    t_hi = set_temp + var_temp

    _ptp = el_temperature.rolling(20).apply(np.ptp, raw=True) >= var_thresh
    _tmin = el_temperature.rolling(20).min() < t_lo
    _tmax = el_temperature.rolling(20).max() > t_hi

    _mask = (_ptp | _tmin | _tmax).astype(bool)

    return _mask


def get_slr_mask(wow, ps, roll):
    """
    Return a mask indicating whether or not the aircraft is in a straight and
    level run.

    Args:
        wow - a pd.Series containing the  weight on wheels flag.
        ps - a pd.Series containing static pressure.
        roll - a pd.Series containing aircraft roll.

    Returns:
        mask - a boolean mask, with True values indicating that the aircraft is
               in a straight and level run.
    """

    _slrs = slrs(wow, ps, roll, min_length=10, roll_lim=1, ps_lim=.2)
    mask = pd.Series(0, index=wow.index)

    for slr in _slrs:
        mask.loc[(mask.index >= slr[0]) & (mask.index <= slr[-1])] = 1

    return mask.astype(bool)


def dryair_calc(p_sense, T, ts, ps, tas, cloud_mask=None, rtn_func=False,
                rtn_goodness=False):
    """
    Calculate dry air power term by fitting constants for 1st principles.
    The calculation of the dry air power term is based on method three
    as described on page 58 of the WCM-2000 manual. This uses a fit
    between the theoretical and measured (in cloud-free conditions)
    sense powers to find the fitting constants k1 and k2 (k2~0.5).

    Psense,dry = k1 * (T - ts) * (ps * tas)**k2
    Psense,total = Psense,dry + Psense,wet
    =>
    Psense,total - Psense,dry = 0 in cloud-free conditions

    Args:
        Psense: sense element power (W)
        T: temperature of sense element (deg C)
        ts: ambient static temperature (deg C)
        ps: ambient static air pressure (mbar)
        tas: true air speed (m/s)

    Kwargs:
        cloud_mask: Array of True/False or 1/0 for in/out of cloud
                    Default is None for no cloud.
        rtn_func: If False [default] array of dry air powers for given
                  sense element is returned. If True then the lambda function to
                  calculate the dry air power from Pcomp is returned.
        rtn_goodness: If True then also return the quality of the fit based
                      on the covariance. Default is False.

    Returns:
        result: Array of dry air power terms calculated for each value of
                Psense. Will return None if the fitting routine fails.
    """

    if (cloud_mask is None) or np.all(cloud_mask == False): # pylint: disable=singleton-comparison
        cloud = np.zeros_like(p_sense).astype(bool)
    else:
        cloud = np.ma.make_mask(cloud_mask)

    nan_mask = ((~np.isfinite(p_sense)) | (~np.isfinite(T)) |
                (~np.isfinite(ts)) | (~np.isfinite(ps)) | (~np.isfinite(tas)))

    mask = np.logical_or(cloud, nan_mask)

    if np.all(mask):
        return None

    _p_sense = p_sense[~mask]
    _T = T[~mask]
    _ts = ts[~mask]
    _ps = ps[~mask]
    _tas = tas[~mask]

    func1 = lambda x, a, b: (a * (_T - _ts) * (_ps * _tas)**b) - x
    func2 = lambda a, b: a * (_T - _ts) * (_ps * _tas)**b
    func3 = lambda a, b: a * (T - ts) * (ps * tas)**b

    try:
        # pylint: disable=unbalanced-tuple-unpacking
        popt, pcov = curve_fit(
            func1, _p_sense, np.zeros_like(_p_sense), p0=(2.5e-3, 0.5),
            method='trf'
        )
    except (TypeError, ValueError, RuntimeError):
        # Incompatible inputs or minimization failure
        pcov = np.inf

    if np.any(np.isinf(pcov)):
        # No convergence
        # TODO: This should RAISE an error instead?
        if rtn_goodness:
            return None, -np.inf
        return None

    # Calculate standard deviation of fitting parameters
    perr = np.sqrt(np.diag(pcov))
    opt_err = np.dstack((popt, perr)).ravel()

    # Calculate the coefficient of determination (r^2) to get goodness of fit metric
    # https://en.wikipedia.org/wiki/Coefficient_of_determination
    ss_res = np.sum((p_sense[~mask] - func2(*popt)) ** 2)
    ss_tot = np.sum((p_sense[~mask] - np.mean(p_sense[~mask])) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    if rtn_func & rtn_goodness:
        # This doesn't actually make a lot of sense due to constants
        return lambda x: func3(*popt), r2

    if rtn_func:
        return lambda x: func3(*popt)

    if rtn_goodness:
        return func3(*popt), r2

    return func3(*popt)


def dryair_calc_comp(p_sense, p_comp, cloud_mask=None, rtn_func=False,
                     rtn_goodness=False):
    """
    Calculate dry air power term from compensation element measurements.
    The calculation of the dry air power term (DAT) is based on the use of
    the compensation element as described on page 56 of the WCM-2000 manual.
    This finds the slope and offset for conversion of the compensation
    power to dry air sense element power.

    Psense,dry = P0 + K * Pcomp
    Psense,total = P0 + K * Pcomp when in clear air

    This will be TAS and Pambient dependent (possibly Tambient). This
    function determines optimum fitting parameters for the entire dataset,
    thus any baseline drift shall be lost.

    Args:
        Psense: Array of powers of sense element.
        Pcomp: Array of powers of compensation element for same
               times. Shape of two input arrays must be the same.

    Kwargs:
        cloud_mask: Array of True/False or 1/0 for in/out of cloud
                    Default is None for no cloud.
        rtn_func: If False [default] array of dry air powers for given
                  sense element is returned. If True then the lambda function to
                  calculate the dry air power from Pcomp is returned.
        rtn_goodness: If True then also return the quality of the fit based
                      on the covariance. Default is False.

    Returns:
        Array of dry air powers if rtn_func is False, if rtn_func is
        True then returns the fitting function. If there is no data to fit
        as all data is masked then an array of NaN is returned
    """

    # Linear fitting function
    func = lambda x, a, b: a*x + b

    # Create mask based on cloud_mask
    # This step is to cope with different types of binary elements
    # np.all statement so that cloud does not become False in make_mask step
    if cloud_mask is None or np.all(cloud_mask == False): # pylint: disable=singleton-comparison
        cloud = np.array([False] * len(p_comp))
    else:
        cloud = np.ma.make_mask(cloud_mask, shrink=False)

    # Remove any nan's from input arrays by wrapping up into cloud mask
    nan_mask = np.logical_or(~np.isfinite(p_comp), ~np.isfinite(p_sense))
    mask = np.logical_or(cloud, nan_mask)

    # Check whether entire array is masked
    if np.all(mask):
        arr = np.empty_like(mask) * np.nan
        if rtn_goodness:
            return arr, -inf
        return arr

    # Fit compensation power to sense power
    # Interpolations don't accept masked arrays so delete masked elements
    # pylint: disable=unbalanced-tuple-unpacking
    try:
        popt, pcov = curve_fit(func, p_comp[~mask], p_sense[~mask])
    except (TypeError, ValueError, RuntimeError):
        # Incompatible inputs or minimization failure
        pcov = np.inf

    if np.any(np.isinf(pcov)):
        # No convergence
        # TODO: This should RAISE an error instead?
        if rtn_goodness:
            return np.empty_like(mask) * np.nan, -np.inf

        return np.empty_like(mask) * np.nan

    # Calculate standard deviation of fitting parameters
    perr = np.sqrt(np.diag(pcov))
    opt_err = np.dstack((popt, perr)).ravel()

    # Calculate the coefficient of determination (r^2) to get goodness of fit metric
    # https://en.wikipedia.org/wiki/Coefficient_of_determination
    ss_res = np.sum((p_sense[~mask] - func(p_comp[~mask], *popt))**2)
    ss_tot = np.sum((p_sense[~mask] - np.mean(p_sense[~mask]))**2)
    r2 = 1 - (ss_res / ss_tot)

    if rtn_func & rtn_goodness:
        return lambda x: func(x,*popt), r2

    if rtn_func:
        return lambda x: func(x,*popt)

    if rtn_goodness:
        return func(p_comp,*popt), r2

    return func(p_comp,*popt)


def T_check(V, I, Tset, R100, dTdR, Twarn=None):
    """
    Compare calculated element temperature to setpoint temperature
    This function calculates the element resistance from the voltage and
    current readings and with the R100 and dT/dR calibration data converts
    this into element temperature. This is compared to the setpoint
    temperature.

    From WCM-2000 manual page 62.

    Args:
        V: voltage across element (V)
        I: current through element (I)
        Tset: setpoint temperature (dec C)
        R100: calibrated element resistance at 100deg C
        dTdR: calibrated element differential temp resistance ratio

    Kwargs:
        Twarn: Temperature difference from setpoint at which to
               trigger a warning. Default is None and defined in function.

    Returns:
        Array of differences between setpoint and calculated temperature.
        Masked for differences greater than set warning level
    """

    # +/- temperature difference (deg C) at which to trigger warning
    if not Twarn:
        Twarn = 5

    # Calculate resistance. Make 64bit so can cope with I -> 0
    R = np.divide(V, I, dtype='f8')

    # Calculate element temperature and difference
    Tcalc = 100 + (R - R100) * dTdR
    Tdiff = Tcalc - Tset

    return np.ma.masked_outside(Tdiff, -abs(Twarn), abs(Twarn))


def energy_liq(ps):
    """
    Calculate the evaporative temperature and latent heat of evaporation.
    Based on WCM-2000 empirical equations on page 64. No references found.
    Valid for pressures 100-1050 mb.

    Korolev [KoSI98] includes the evaporative temperature in equation 8
    although this includes element efficiency.

    Args:
        ps: ambient static air pressure (mbar)

    Returns:
        (Tevap, Levap):
            Tevap: Temperature of evaporation (deg C)
            Levap: Latent heat of evaporation (J/g)
    """

    # Ensure array so equations work correctly for all inputs
    ps = np.asarray(ps)

    # Calculate evaporative temperature of liquid water with pressure
    Tevap = (
        32.16 + (0.1801 * ps) - (2.391e-4 * ps**2.)
        + (1.785e-7 * ps**3.) - (5.19e-11 * ps**4.)
    )

    Levap = 594.4 - (0.484 * Tevap) - (7.0e-4 * Tevap**2.)

    Levap *= cal_to_J

    return Tevap, Levap


def calc_L(T, ps):
    """
    Calculate the specific energies for melting and/or evaporation
    The specific energy expended to evaporate water of a given temperature,
    T, is given by L^*_liq. The specific energy expended to melt then evaporate
    ice of a given temperature, T, is given by L^*_ice. The ratio of L*_i to
    L*_l is designated as k and is used to calculate LWC and IWC.

    This is described in Korolev 1998 and 2003.

    Args:
        T: Ambient temperature (degree C). Function attempts to intercept
           temperatures in kelvin and convert
        ps: Ambient static pressure (mb)

    Returns:
        (Lstar_liq, Lstar_ice):
            Lstar_liq: The specific energy expended for liquid water (J/g)
            Lstar_ice: The specific energy expended for ice (J/g)
    """


    # Convert temperatures from kelvin to celsius if necessary
    if np.any(T<0):
        # T must be in celcius
        pass

    elif np.any(T>170):
        # Assume is in kelvin
        T = K_to_C(T.copy())

    # Latent heat of fusion for ice (J/g)
    # 333.5J/g == 79.71cal/g from Osborne, 1939.
    # (apparently, I can't find it now!)
    L_ice = 333.5

    # Specific heat of water from 0-100degC
    # From Osborne et al. 1939, page 227 and checked against Table 6
    C_liq = lambda t: (
        4.169828  + (0.000364 * (t+100)**5.26) * 1e-10
        + 0.046709*10**(-0.036*t)
    )

    # Specific heat of ice
    # from http://www.kayelaby.npl.co.uk/general_physics/2_3/2_3_6.html
    # Have pinned values for T>0 to C_ice(0)
    C_ice_T = K_to_C(np.array([77., 173., 273.]))
    C_ice_data = np.array([0.686, 1.372, 2.097])
    C_ice = lambda t: np.interp(t, C_ice_T, C_ice_data)

    # Specific heat of ice from Dickinson, 1915 as a comparison to above.
    # Has a slightly larger gradient with a cross-over at ~-33degC.
    C_ice_d = lambda t: (0.5057 + 0.001863 * t) * cal20_to_J

    # Obtain evaporation temperature and latent heat for ambient pressure
    # Note that T_sens > T_e > T
    T_e, L_liq = energy_liq(ps)

    # Calculate specific energy of liquid water
    # from Korolev et al. 2003. eq 5
    Lstar_liq = C_liq(T_e - T) + L_liq

    # Equation divided into melting (up to 0C) and evaporation (0C -> T_e)
    # from Korolev et al. 2003. eq 6
    Lstar_ice = C_ice(T) + L_ice + C_liq(T_e) + L_liq

    return Lstar_liq, Lstar_ice


def calc_sense_wc(Psense, Levap, Tevap, tat, tas, sens_dim):
    """
    Calculate the sense element-measured water content (either LWC or IWC).
    Use equation as given on pages 61,65 of the WCM-2000 manual. This is
    the same as eqs 3,4 from Korolev 2003 but for calories instead of joules.
    This calculation does not include any element efficiencies

    The same L*liq is used for both liquid and total water contents.
    This is fine as the difference between liquid and ice is accounted for
    by including the appropriate efficiencies in calc_lwc() and calc_iwc().

    Args:
        Psense: Sense element wet power (total-DAT) (W)
        Levap: Latent heat of evaporation (cal/gm)
        Tevap: Evaporative temperature in (degC)
        tat: True air temperature (degC)
        tas: True air speed (m s-1)
        sens_dim: (length, width) both in mm

    Returns:
        wc: Water content (liquid or total) as measured by the
            sense element (g/m**3)
    """

    wc = (
        (Psense * 2.389*(10**5)) /
        ((Levap + (Tevap - tat)) * tas * sens_dim[0] * sens_dim[1])
    )

    return wc


def calc_lwc(W_twc, W_lwc, k, e_liqT=1, e_iceT=1, e_liqL=1, beta_iceL=0):
    """
    Calculate liquid water content from the measured LWC and TWC.

    Args:
        W_twc: array of as-measured total water content from TWC element
               in g/m**3.
        W_lwc: array of as-measured total water content from 083 or 021
               LWC element in g/m**3.
        k: Ratio of expended specific energies of water evaporation and ice
           sublimation

    Kwargs:
        e_liqT: Collection efficiency of the TWC sensor for liquid droplets.
                Default is 1.
        e_iceT: Collection efficiency of the TWC sensor for ice particles.
                Default is 1.
        e_liqL: Collection efficiency of the LWC sensor for liquid droplets.
                Default is 1.
        beta_iceL: Collection efficiency of the LWC sensor for ice
                   particles. Default is 0.

    Returns:
        lwc: The calculated liquid water content (g/m**3).
    """

    with warnings.catch_warnings():
        # Often get RuntimeWarning for invalid values for nans etc.
        warnings.simplefilter('ignore')
        lwc = np.divide(
            np.asfarray(beta_iceL) * W_twc - k * np.asfarray(e_iceT) * W_lwc,
            np.asfarray(beta_iceL) * np.asfarray(e_liqT)
            - np.asfarray(e_liqL) * k * np.asfarray(e_iceT)
        )

    return lwc


def calc_iwc(W_twc, W_lwc, k, e_liqT=1, e_iceT=1, e_liqL=1, beta_iceL=0):
    """
    Calculate ice water content from the measured LWC and TWC

    Args:
        W_twc: array of as-measured total water content from TWC element
               in g/m**3.
        W_lwc: array of as-measured total water content from 083 or 021
               LWC element in g/m**3.
        k: Ratio of expended specific energies of water evaporation and ice
           sublimation

    Kwargs:
        e_liqT: Collection efficiency of the TWC sensor for liquid droplets.
                Default is 1.
        e_iceT: Collection efficiency of the TWC sensor for ice particles.
                Default is 1.
        e_liqL: Collection efficiency of the LWC sensor for liquid droplets.
                Default is 1.
        beta_iceL: Collection efficiency of the LWC sensor for ice
                   particles. Default is 0.

    Returns:
        lwc: The calculated liquid water content (g/m**3).
    """

    iwc = np.divide(
        np.asfarray(e_liqL) * W_twc - np.asfarray(e_liqT) * W_lwc,
        np.asfarray(e_liqL) * k * np.asfarray(e_iceT)
        - np.asfarray(beta_iceL) * np.asfarray(e_liqT)
    )

    return iwc

def calc_wc(W_twc, W_lwc, k, e_liqT=1, e_iceT=1, e_liqL=1, beta_iceL=0):
    """
    Calculate total, liquid and ice water contents.

    Args:
        W_twc: array of as-measured total water content from TWC element
               in g/m**3.
        W_lwc: array of as-measured total water content from 083 or 021
               LWC element in g/m**3.
        k: Ratio of expended specific energies of water evaporation and ice
           sublimation

    Kwargs:
        e_liqT: Collection efficiency of the TWC sensor for liquid droplets.
                Default is 1.
        e_iceT: Collection efficiency of the TWC sensor for ice particles.
                Default is 1.
        e_liqL: Collection efficiency of the LWC sensor for liquid droplets.
                Default is 1.
        beta_iceL: Collection efficiency of the LWC sensor for ice
                   particles. Default is 0.

    Returns:
        (lwc, iwc, twc):
            The calculated liquid, ice and total water content (g/m**3),
            respectively.
    """

    lwc = calc_lwc(W_twc, W_lwc, k, e_liqL, e_liqT, e_iceT, beta_iceL)
    iwc = calc_iwc(W_twc, W_lwc, k, e_liqL, e_liqT, e_iceT, beta_iceL)
    twc = lwc + iwc

    return lwc, iwc, twc


@register_pp('core')
class SeaProbe(PPBase):
    """
    Calculates bulk water contents from the SEA WCM-2000 sensor.
    """

    inputs = [
        'WOW_IND',
        'PS_RVSM',
        'TAS_RVSM',
        'ROLL_GIN',
        'TAT_DI_R',
        'SEAPROBE_d0_021',
        'SEAPROBE_d0_021_A',
        'SEAPROBE_d0_021_T',
        'SEAPROBE_d0_021_V',
        'SEAPROBE_d0_083',
        'SEAPROBE_d0_083_A',
        'SEAPROBE_d0_083_T',
        'SEAPROBE_d0_083_V',
        'SEAPROBE_d0_CMP',
        'SEAPROBE_d0_CMP_A',
        'SEAPROBE_d0_CMP_T',
        'SEAPROBE_d0_CMP_V',
        'SEAPROBE_d0_DCE',
        'SEAPROBE_d0_DCE_A',
        'SEAPROBE_d0_DCE_T',
        'SEAPROBE_d0_DCE_V',
        'SEAPROBE_d0_TWC',
        'SEAPROBE_d0_TWC_A',
        'SEAPROBE_d0_TWC_T',
        'SEAPROBE_d0_TWC_V',
        'SEAPROBE_c0_TWC_l',
        'SEAPROBE_c0_TWC_w',
        'SEAPROBE_c0_083_l',
        'SEAPROBE_c0_083_w',
        'SEAPROBE_c0_021_l',
        'SEAPROBE_c0_021_w',
        'SEAPROBE_c0_CMP_l',
        'SEAPROBE_c0_CMP_w',
        'SEA_EFF_TWC',
        'SEA_EFF_021',
        'SEA_EFF_083',
        'SEA_TEMP_LIMS',
        'SEA_SETPOINT_TEMP',
        'SEA_SN'
    ]

    @staticmethod
    def test():
        """
        Provide some dummy input data for testing.
        """
        n = 900
        _d = {
            'SEA_EFF_TWC': ('const', [1, 1]),
            'SEA_EFF_021': ('const', [1, 0]),
            'SEA_EFF_083': ('const', [1, 0]),
            'SEA_SETPOINT_TEMP': ('const', 120),
            'SEA_TEMP_LIMS': ('const', [100, 180]),
            'SEA_SN': ('const', 'XXXX'),
            'WOW_IND': ('data', _z(n), 1),
            'PS_RVSM': ('data', 800 * _o(n), 32),
            'TAS_RVSM': ('data', 200 * _o(n), 32),
            'ROLL_GIN': ('data', _z(n), 32),
            'TAT_DI_R': ('data', 200 * _o(n), 32)
        }

        _all_ele = ('021', '083', 'CMP', 'DCE', 'TWC')
        for ele in _all_ele:
            try:
                _d[f'SEAPROBE_d0_{ele}'] = ('data', int(ele) * _o(n), 20)
            except ValueError:
                _d[f'SEAPROBE_d0_{ele}'] = ('data', n * [ele], 20)

            if ele == 'DCE':
                _d[f'SEAPROBE_d0_{ele}_T'] = ('data', 50 * _o(n) + .2 * _r(n), 20)
                _d[f'SEAPROBE_d0_{ele}_V'] = ('data', 25 * _o(n) + .2 * _r(n), 20)
            else:
                _d[f'SEAPROBE_d0_{ele}_T'] = ('data', 120 * _o(n) + .2 * _r(n), 20)
                _d[f'SEAPROBE_d0_{ele}_V'] = ('data', .4 * _o(n) + .02 * _r(n), 20)

        _d['SEAPROBE_d0_CMP_A'] = ('data', 4.5 * _o(n) + .1 * _r(n), 20)
        _d['SEAPROBE_d0_DCE_A'] = ('data', 9 * _o(n) + .2 * _r(n), 20)
        _d['SEAPROBE_d0_021_A'] = ('data', 12 * _o(n) + .2 * _r(n), 20)
        _d['SEAPROBE_d0_083_A'] = ('data', 20 * _o(n) + .2 * _r(n), 20)
        _d['SEAPROBE_d0_TWC_A'] = ('data', 17 * _o(n) + .2 * _r(n), 20)

        _d['SEAPROBE_c0_TWC_l'] = ('data', 23.139 * _o(n), 20)
        _d['SEAPROBE_c0_TWC_w'] = ('data', 2.108 * _o(n), 20)
        _d['SEAPROBE_c0_083_l'] = ('data', 22.555 * _o(n), 20)
        _d['SEAPROBE_c0_083_w'] = ('data', 2.108 * _o(n), 20)
        _d['SEAPROBE_c0_021_l'] = ('data', 21.184 * _o(n), 20)
        _d['SEAPROBE_c0_021_w'] = ('data', 0.533 * _o(n), 20)
        _d['SEAPROBE_c0_CMP_l'] = ('data', 16.764 * _o(n), 20)
        _d['SEAPROBE_c0_CMP_w'] = ('data', 0.2794 * _o(n), 20)

        return _d

    def declare_outputs(self):
        manufacturer = 'Science Engineering Associates, Inc.'
        model = 'WCM-2000'

        self.declare(
            'SEA_TWC_021',
            units='g m-3',
            frequency=20,
            long_name=('Total water content from the SEA WCM-2000 probe, '
                       'element 021'),
            instrument_manufacturer=manufacturer,
            instrument_model=model,
            instrument_serial_number=self.dataset.lazy['SEA_SN']
        )

        self.declare(
            'SEA_TWC_083',
            units='g m-3',
            frequency=20,
            long_name=('Total water content from the SEA WCM-2000 probe, '
                       'element 083'),
            instrument_manufacturer=manufacturer,
            instrument_model=model,
            instrument_serial_number=self.dataset.lazy['SEA_SN']
        )


        self.declare(
            'SEA_LWC_021',
            units='g m-3',
            frequency=20,
            long_name=('Liquid water content from the SEA WCM-2000 probe, '
                       'element 021'),
            standard_name='mass_concentration_of_liquid_water_in_air',
            instrument_manufacturer=manufacturer,
            instrument_model=model,
            instrument_serial_number=self.dataset.lazy['SEA_SN']
        )

        self.declare(
            'SEA_LWC_083',
            units='g m-3',
            frequency=20,
            long_name=('Liquid water content from the SEA WCM-2000 probe, '
                       'element 083'),
            standard_name='mass_concentration_of_liquid_water_in_air',
            instrument_manufacturer=manufacturer,
            instrument_model=model,
            instrument_serial_number=self.dataset.lazy['SEA_SN']
        )

    def process(self):
        self.get_dataframe(
            method='onto', index=self.dataset['SEAPROBE_d0_TWC_T'].index
        )
        df = self.d

        # 1 Hz variable requires filling
        df['WOW_IND'] = df['WOW_IND'].fillna(method='ffill').astype(bool)

        key = lambda e, s: f'SEAPROBE_d0_{e}_{s}'

        def _wc(_el, _Levap, _Tevap, _l, _w):
            _key = lambda s: f'SEAPROBE_d0_{_el}_{s}'

            # Calculate dry air terms using compensation element
            # Note that these generally give a better DAT than 1st principles
            df[_key('DATcomp')], DATcomp_r2 = dryair_calc_comp(
                df[_key('P')].values, df['SEAPROBE_d0_CMP_P'].values,
                df['dryair_mask'].values, rtn_goodness=True
            )

            # Calculate dry air terms using first principles method
            df[_key('DAT')], DAT_r2 = dryair_calc(
                df[_key('P')].values, df[_key('T')].values,
                df['TAT_DI_R'].values, df['PS_RVSM'].values,
                df['TAS_RVSM'].values, cloud_mask=df['dryair_mask'],
                rtn_goodness=True
            )

            # Determine the best DAT to use based on r squared value of fit
            # Note that there is currently no meta record of which DAT is used
            if DATcomp_r2 > 0.75 * DAT_r2:
                DAT = df[_key('DATcomp')]
            else:
                DAT = df[_key('DAT')]

            df[_key('WC')] = calc_sense_wc(
                df[_key('P')] - DAT, _Levap, _Tevap, df['TAT_DI_R'],
                df['TAS_RVSM'], (_l, _w)
            )

        # calculate the powers for the sensors
        for el in ['TWC','083','021','CMP','DCE']:
            df[key(el,'P')] = df[key(el,'A')] * df[key(el,'V')]

        # Get straight-and-level mask
        slr_mask = get_slr_mask(
            df['WOW_IND'],
            df['PS_RVSM'],
            df['ROLL_GIN']
        )

        instr_mask = get_instr_fault_mask(
            el_temps=df[[
                'SEAPROBE_d0_TWC_T',
                'SEAPROBE_d0_083_T',
                'SEAPROBE_d0_021_T'
            ]],
            temp_limits=self.dataset['SEA_TEMP_LIMS']
        )

        # Put masks into dataframe
        df['slr_mask'] = slr_mask
        df['instr_mask'] = instr_mask

        df['cloud_mask'] = get_cloud_mask(
            df['SEAPROBE_d0_TWC_T'],
            **{'var_thresh': 1.5,
               'set_temp': self.dataset['SEA_SETPOINT_TEMP'],
               'var_temp': 2.}
        )

        # Mask for dry air calcs: must be slr, out of cloud, and instruments in
        # range
        df['dryair_mask'] = (~slr_mask) | (df.cloud_mask) | (instr_mask)

        # Calculate the Temp of evaporation and latent heat of evaporation
        Tevap, Levap = energy_liq(df['PS_RVSM'].values)

        # Calculate the specific energies for sublimation and evaporation
        L_liq, L_ice = calc_L(df['TAT_DI_R'].values, df['PS_RVSM'].values)
        k = L_ice / L_liq

        for el in ['TWC','083','021']:
            # Get element width and length from the c0 message
            l = df[f'SEAPROBE_c0_{el}_l'].dropna().iloc[0]
            w = df[f'SEAPROBE_c0_{el}_w'].dropna().iloc[0]

            _wc(el,Levap,Tevap,l,w)

        # Get element efficiencies for TWC/083 and TWC/021 calcs from flight
        # constants
        eff_TWC083  = np.array([
            np.array(self.dataset['SEA_EFF_TWC']),
            np.array(self.dataset['SEA_EFF_083'])
        ]).ravel()

        eff_TWC021  = np.array([
            np.array(self.dataset['SEA_EFF_TWC']),
            np.array(self.dataset['SEA_EFF_021'])
        ]).ravel()

        df['LWC_083'] = calc_lwc(
            df['SEAPROBE_d0_TWC_WC'].values,
            df['SEAPROBE_d0_083_WC'].values,
            k, *eff_TWC083
        )

        df['LWC_021'] = calc_lwc(
            df['SEAPROBE_d0_TWC_WC'].values,
            df['SEAPROBE_d0_021_WC'].values,
            k, *eff_TWC021
        )

        df['IWC_083'] = calc_iwc(
            df['SEAPROBE_d0_TWC_WC'].values,
            df['SEAPROBE_d0_083_WC'].values,
            k, *eff_TWC083
        )

        df['IWC_021'] = calc_iwc(
            df['SEAPROBE_d0_TWC_WC'].values,
            df['SEAPROBE_d0_021_WC'].values,
            k, *eff_TWC021
        )

        # Total water is sum of liquid and ice
        df['TWC_083'] = df['LWC_083'] + df['IWC_083']
        df['TWC_021'] = df['LWC_021'] + df['IWC_021']

        # Define output variables
        outputs = {
            'twc_021': DecadesVariable(
                df['TWC_021'], name='SEA_TWC_021', flag=DecadesBitmaskFlag
            ),
            'twc_083': DecadesVariable(
                df['TWC_083'], name='SEA_TWC_083', flag=DecadesBitmaskFlag
            ),
            'lwc_201': DecadesVariable(
                df['LWC_021'], name='SEA_LWC_021', flag=DecadesBitmaskFlag
            ),
            'lwc_083': DecadesVariable(
                df['LWC_083'], name='SEA_LWC_083', flag=DecadesBitmaskFlag
            )
        }

        # Set flagging info and add output to dataset
        for var in outputs.values():
            var.flag.add_mask(
                df.instr_mask, 'element temperature out of range'
            )
            var.flag.add_mask(df.WOW_IND, flags.WOW)
            self.add_output(var)

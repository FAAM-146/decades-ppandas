"""
This module provides a postprocessing module for the Nevzorov vane in both the
'old' (1T1L2R) and 'new' (1T2L1R) configurations. See class docstring for more
info.
"""
# pylint: disable=invalid-name, too-many-arguments, too-many-locals
# pylint: disable=too-many-statements
import warnings
import logging

from enum import Enum, auto

import numpy as np
import pandas as pd

from vocal.types import DerivedString

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..decades.attributes import DocAttribute
from ..utils import slrs
from .base import PPBase, register_pp
from .shortcuts import _o, _z


MINIMUM_IAS_CORRELATION = 0.8
BASELINE_DEVIATION_LIMIT = 0.02

logger = logging.getLogger(__name__)

LinearFit = tuple[float, float]


class CORRECTION_STATUS(Enum):
    """
    Enum to represent the status of the baseline correction.
    """

    UNINITIALISED = auto()
    FROM_FLIGHT_DATA = auto()
    FROM_CONSTANTS = auto()
    FAILED = auto()


def get_baseline_flag(
        col_p: pd.Series, ref_p: pd.Series, k: pd.Series | float, mask: pd.Series
    ) -> pd.Series:
    """
    Get a flag indicating when the parameterized baseline correction is
    known to be poor.

    Args:
        col_p: Collector power.
        ref_p: Reference power.
        mask: Clear air mask.

    Returns:
        A mask indicating when the Nevzorov is in clear air.
    """

    flag = col_p * 1

    if isinstance(k, float):
        k = pd.Series(k, index=col_p.index)

    smoothed_diff = (
        (col_p / ref_p).rolling(64, center=True).mean()
        - k.rolling(64, center=True).mean() # type: ignore #  we're confident k is a series
    )

    flag.loc[mask == 0] = 0
    flag.loc[smoothed_diff.abs() < BASELINE_DEVIATION_LIMIT] = 0

    return flag


def get_water_content_comment(status: CORRECTION_STATUS) -> str:
    """
    Get a variable comment indicating how the baseline correction was
    determined.
    """

    match status:

        case CORRECTION_STATUS.FROM_FLIGHT_DATA:
            return "Automatically baselined using flight data."
        
        case CORRECTION_STATUS.FROM_CONSTANTS:
            return "Automatically baselined using flight constants."
        
        case (CORRECTION_STATUS.UNINITIALISED, CORRECTION_STATUS.FAILED):
            return "Failed to baseline correct Nevzorov."
        
        case _:
            return "Unknown baseline correction status."


def get_k_ias_fit(
        col_p: pd.Series, ref_p: pd.Series, ias: pd.Series, mask: pd.Series,
        nominal_k: float, runs: list[pd.DatetimeIndex]
    ) -> LinearFit:
    """
    Get the linear fit for the IAS correction.

    Args:
        col_p: Collector power.
        ref_p: Reference power.
        ias: Indicated air speed.
        mask: Clear air mask.
        nominal_k: The nominal baseline correction.
        runs: list of indices for each run.

    Returns:
        The linear fit.
    """

    ms = []
    cs = []

    for run in runs:
        run_mask = mask[run]

        run_k = (col_p / ref_p)[run] - nominal_k
        run_k.loc[run_mask==0] = np.nan

        run_rias = 1 / ias[run]
        run_rias.loc[run_mask==0] = np.nan

        cc = np.corrcoef(run_rias, run_k)[0, 1]
        if np.isnan(cc):
            cc = 0
        if cc < MINIMUM_IAS_CORRELATION:
            continue

        (x, y) = np.polyfit(run_rias, run_k, 1)
        ms.append(x)
        cs.append(y)

    fit = (np.median(ms), np.median(cs))

    return fit


def get_k_ps_fit(
        col_p: pd.Series, ref_p: pd.Series, ps: pd.Series, mask: pd.Series,
        k_ias: pd.Series, nominal_k: float, runs: list[pd.DatetimeIndex]
    ) -> LinearFit:
    """
    Get the linear fit for the PS correction.

    Args:
        col_p: Collector power.
        ref_p: Reference power.
        ps: Static pressure.
        mask: Clear air mask.
        k_ias: The IAS correction.
        nominal_k: The nominal baseline correction.
        runs: list of indices for each run.

    Returns:
        The linear fit.
    """

    pss = []
    rats = []

    measured_k = (col_p / ref_p)
    ps = ps.reindex(measured_k.index).interpolate().bfill()

    corrected_k = measured_k - k_ias

    for run in runs:
        ri = corrected_k[run].loc[mask==1] - nominal_k
        psi = np.log10(ps[run].loc[mask==1])
        rim, psim = ri.median(), psi.median()
        if np.isnan(rim) or np.isnan(psim):
            continue

        rats.append(rim)
        pss.append(psim)

    fit = np.polyfit(pss, rats, 1)

    return fit


def get_k_ps_from_fit(fit: LinearFit, ps: pd.Series) -> pd.Series:
    """
    Return the k value for a given set of PS values and a linear fit.

    Args:
        fit: The linear fit.
        ps: The static pressure.

    Returns:
        The k value.
    """
    return fit[0] * np.log10(ps) + fit[1]


def get_k_ias_from_fit(fit: LinearFit, ias: pd.Series) -> pd.Series:
    """
    Return the k value for a given set of IAS values and a linear fit.

    Args:
        fit: The linear fit.
        ias: The indicated air speed.

    Returns:
        The k value.
    """
    return fit[0] * (1 / ias) + fit[1]


def get_parameterized_k(
        ias_fit: LinearFit, ps_fit: LinearFit, ias: pd.Series, ps: pd.Series,
        nominal_k: float
    ) -> pd.Series:
    """
    Return the parameterized k value for a given set of IAS and PS values.

    Args:
        ias_fit: The linear fit for the IAS correction.
        ps_fit: The linear fit for the PS correction.
        ias: The indicated air speed.
        ps: The static pressure.
        nominal_k: The nominal baseline correction.

    Returns:
        The parameterized k value.
    """

    return (
        get_k_ias_from_fit(ias_fit, ias) + get_k_ps_from_fit(ps_fit, ps) + nominal_k
    )


def get_no_cloud_mask(twc_col_p, wow, window_secs=3, min_period=5, freq=64):
    """
    Create a mask giving times when we are
    a) in clear air
    b) not on the ground.
    This is determined by looking at the range of the total water collector
    power in a given window. Range in cloud should be significantly higher than
    the range in clear air.

    Args:
        twc_col_p: Total Water Collector power, as a pd.Series.
        wow: The weighton-wheels flag

    Kwargs:
        window_secs: the size of the window, in secs
        min_period: the minimum total time, in secs, that we must be in/out of
                    cloud to flip the flag value.
        freq: The frequency of the Nevzorov signal.

    Returns:
        mask: The clear air mask. 1 indicates inflight/no cloud, 0 indicates on
              ground/in cloud.
    """

    range_limits = (1E-12, 0.1)

    vals = np.array(twc_col_p.values)
    out = np.zeros_like(vals)

    # Window sizes
    window_size = window_secs * freq
    min_length = min_period * freq
    _half_window = int(np.floor(window_size / 2))

    # Calculate range through a sliding window
    for i in range(0, len(vals) - _half_window):
        _range = np.ptp(vals[i:i+window_size])
        out[i + _half_window] = range_limits[0] < _range < range_limits[1]

    out = out.astype(int)

    # Ensure that sectors marked as clear air are not too short
    _split = np.split(out, np.where(np.abs(np.diff(out)) != 0)[0] + 1)
    for _group in _split:
        if len(_group) < min_length:
            _group[:] = 0

    out = np.concatenate(_split)

    # Don't allow clear air calculations when we're on the ground
    out[wow == 1] = 0

    # Return mask as a Pandas object.
    return pd.Series(out, index=twc_col_p.index)


@register_pp('core')
class Nevzorov(PPBase):
    r"""
    Post processing for liquid and total water from the Nevzorov Vane. Works
    with both ``1T1L2R`` and ``1T2L1R`` vanes, which should be specified in
    the flight constants as ``VANETYPE``.

    The Nevzorov hot-wire probe measures total and liquid water content by
    recording the powers required to hold exposed and sheltered wires at a
    constant temperature.

    The water content, :math:`W`, measured by a collector is given by

    .. math::
        W = \frac{P_c - K P_r}{V_t A L},

    where :math:`P_c` is the collector power, :math:`P_r` is the reference
    power, :math:`K` is the baseline, the ratio of :math:`P_c` and :math:`P_r`
    in clear air, :math:`V_t` is the true air speed, :math:`A` is the
    forward-facing area of the collector, and :math:`L` is the energy required
    to melt and then evaporate the water impacted on the sensor, specified in
    the flight constants as ``CALNVL``.

    The baseline, :math:`K`, is not a true constant, but varies with the ambient
    conditions. `Abel et al. (2014) <https://doi.org/10.5194/amt-7-3007-2014>`_
    parameterise :math:`K` as a function of indicated air speed,
    :math:`V_\text{IAS}` and ambient pressure, :math:`P`,

    .. math::
        K = \alpha_\text{IAS}\frac{1}{V_\text{IAS}} + \alpha_P\log_{10}(P).

    If, for any reason, the fitting above fails, then only the uncorrected
    outputs, using a constant :math:`K` specified in the flight constants, are
    written to file.

    The outputs listed produced by this module depend on the type of Nevzorov
    vane fitted to the aircraft. If an old-style ``1T1L2R`` vane is fitted,
    then the default outputs are 
    `NV_TWC_C`, `NV_LWC_C`, `NV_TWC_COL_P`, `NV_LWC_COL_P`, `NV_TWC_REF_P`, `NV_LWC_REF_P`.
    If a new-style ``1T2L1R`` vane is fitted, then the default outputs are 
    `NV_TWC_C`, `NV_LWC1_C`, `NV_LWC2_C`, `NV_TWC_COL_P`, `NV_LWC1_COL_P`, 
    `NV_LWC2_COL_P`, `NV_REF_P`.

    .. note::

        Prior to software version 24.6.0 this module output uncorrected water
        content data (`NV_TWC_U`, `NV_LWC_U`, `NV_LWC1_U`, `NV_LWC2_U`), instead
        of the element powers. These outputs are no longer produced.
    """

    TEST_SETUP = {'VANETYPE': 'all'}

    inputs = [
        'CORCON_nv_lwc_vcol',
        'CORCON_nv_lwc_icol',
        'CORCON_nv_lwc_vref',
        'CORCON_nv_lwc_iref',
        'CORCON_nv_twc_vcol',
        'CORCON_nv_twc_icol',
        'CORCON_nv_twc_vref',
        'CORCON_nv_twc_iref',
        'TAS_RVSM',
        'IAS_RVSM',
        'PS_RVSM',
        'WOW_IND',
        'ROLL_GIN',
        'CLWCIREF', 'CLWCVREF', 'CLWCICOL',
        'CLWCVCOL',
        'CTWCIREF', 'CTWCVREF', 'CTWCICOL',
        'CTWCVCOL',
        'CALNVTWC',
        'CALNVLWC1',
        'CALNVLWC2',
        'CALNVL'
    ]

    instruments = {
        '1t1l2r': ('twc', 'lwc'),
        '1t2l1r': ('twc', 'lwc1', 'lwc2'),
        'all': ('twc', 'lwc', 'lwc1', 'lwc2')
    }

    @staticmethod
    def test():
        """
        Return dummy input data for testing.
        """
        return {
            'VANETYPE': ('const', DocAttribute(value='1t2l1r', doc_value=DerivedString)),
            'VANE_SN': ('const', DocAttribute(value='SN123', doc_value=DerivedString)),
            'CLWCIREF': ('const', [-5.8e-2, 3.3e-4, 5e-1]),
            'CLWCVREF': ('const', [-5.8e-2, 3.3e-4, 2.0]),
            'CLWCICOL': ('const', [-5.8e-2, 3.3e-4, 5e-1]),
            'CLWCVCOL': ('const', [-5.8e-2, 3.3e-4, 2.0]),
            'CTWCIREF': ('const', [-5.8e-2, 3.3e-4, 5e-1]),
            'CTWCVREF': ('const', [-5.8e-2, 3.3e-4, 2.0]),
            'CTWCICOL': ('const', [-5.8e-2, 3.3e-4, 5e-1]),
            'CTWCVCOL': ('const', [-5.8e-2, 3.3e-4, 2.0]),
            'CALNVTWC': ('const', [0.70, 0.5e-4]),
            'CALNVLWC': ('const', [1.35, 0.25e-4]),
            'CALNVLWC1': ('const', [1.35, 0.258e-4]),
            'CALNVLWC2': ('const', [1.95, 0.381e-4]),
            'CALNVL': ('const', 2589.0),
            'CALRSL': ('const', [0.75E-4, 110]),
            'CALRST': ('const', [1.2E-4, 110]),
            'CORCON_nv_lwc_vcol': ('data', 7500 * _o(100), 64),
            'CORCON_nv_lwc_icol': ('data', 1e4 * _o(100), 64),
            'CORCON_nv_lwc_vref': ('data', 6e4 * _o(100), 64),
            'CORCON_nv_lwc_iref': ('data', 9e3 * _o(100), 64),
            'CORCON_nv_twc_vcol': ('data', 5e3 * _o(100), 64),
            'CORCON_nv_twc_icol': ('data', 8e3 * _o(100), 64),
            'CORCON_nv_twc_vref': ('data', 8e3 * _o(100), 64),
            'CORCON_nv_twc_iref': ('data', 12e3 * _o(100), 64),
            'TAS_RVSM': ('data', 250 * _o(100), 32),
            'IAS_RVSM': ('data', 200 * _o(100), 32),
            'PS_RVSM': ('data', 500 * _o(100), 32),
            'WOW_IND': ('data', _z(100), 1),
            'ROLL_GIN': ('data', 0.0 * _o(100), 32)
        }

    def _declare_outputs_common(self):
        """
        Declare the outputs that are common between both Nevz. vanes.
        """

        self.declare(
            'NV_TWC_C',
            units='gram m-3',
            frequency=64,
            long_name=('Corrected total condensed water content from the '
                       'Nevzorov probe'),
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN']
        )

        self.declare(
            'NV_TWC_COL_P',
            units='W',
            frequency=64,
            long_name='TWC collector power',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN'],
        )

        self.declare(
            'NV_CLEAR_AIR_MASK',
            units=None,
            frequency=64,
            long_name=('Clear air mask based on Nevzorov Total Water power '
                       'variance'),
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN'],
            write=False

        )

    def _declare_outputs_1t1l2r(self):
        """
        Declare the outputs that are only valid for the 1t1l2r vane type
        """

        self.declare(
            'NV_LWC_C',
            units='gram m-3',
            frequency=64,
            long_name='Corrected liquid water content from the Nevzorov probe',
            standard_name='mass_concentration_of_liquid_water_in_air',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN']
        )

        self.declare(
            'NV_TWC_REF_P',
            units='W',
            frequency=64,
            long_name='TWC reference power',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN'],

        )

        self.declare(
            'NV_LWC_COL_P',
            units='W',
            frequency=64,
            long_name='LWC collector power',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN'],
        )

        self.declare(
            'NV_LWC_REF_P',
            units='W',
            frequency=64,
            long_name='LWC reference power',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN']
        )

    def _declare_outputs_1t2l1r(self):
        """
        Declare the outputs that are only valid for the 1t2l1r vane type.
        """

        self.declare(
            'NV_LWC1_C',
            units='gram m-3',
            frequency=64,
            long_name=('Corrected liquid water content from the Nevzorov probe'
                       ' (1st collector)'),
            standard_name='mass_concentration_of_liquid_water_in_air',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN']
        )

        self.declare(
            'NV_LWC2_C',
            units='gram m-3',
            frequency=64,
            long_name=('Corrected liquid water content from the Nevzorov probe'
                       ' (2nd collector)'),
            standard_name='mass_concentration_of_liquid_water_in_air',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN']
        )

        self.declare(
            'NV_REF_P',
            units='W',
            frequency=64,
            long_name='Reference power',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN'],
        )

        self.declare(
            'NV_LWC1_COL_P',
            units='W',
            frequency=64,
            long_name='LWC1 collector power',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN'],
        )

        self.declare(
            'NV_LWC2_COL_P',
            units='W',
            frequency=64,
            long_name='LWC2 collector power',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN']
        )

    def declare_outputs(self):
        """
        Declare the module outputs, which are dependent on the type of vane
        fitted to the aircraft.
        """

        self._declare_outputs_common()

        try:
            _vanetype = self.dataset['VANETYPE'].lower()
        except KeyError:
            warnings.warn('VANETYPE not given, setting to default 1t2l1r')
            self.dataset.constants['VANETYPE'] = '1t2l1r'
            _vanetype = '1t2l1r'

        if self.test_mode:
            _vanetype = 'all'

        if _vanetype in ('1t1l2r', 'all'):
            self._declare_outputs_1t1l2r()
        if _vanetype in ('1t2l1r', 'all'):
            self._declare_outputs_1t2l1r()
        else:
            raise ValueError(
                'Unknown Nevz. vane type: {}'.format(_vanetype)
            )

    def _remap_1t2l1r(self):
        """
        The variables in DECADES are named for the old Nevzorov vane type,
        which had 1 total, 1 liquid and 2 reference sensors. When running with
        the new vane type, which has 1 total, 2 liquid and 1 reference sensors,
        we need to map the old variable names onto new ones which correspont to
        the new vane design.
        """

        _vanetype = self.dataset['VANETYPE'].lower()
        if self.test_mode:
            _vanetype = 'all'

        if _vanetype not in ('1t2l1r', 'all'):
            return

        var_map = (
            ('CORCON_nv_lwc1_vcol', 'CORCON_nv_lwc_vcol'),
            ('CORCON_nv_lwc1_icol', 'CORCON_nv_lwc_icol'),
            ('CORCON_nv_lwc1_vref', 'CORCON_nv_lwc_vref'),
            ('CORCON_nv_lwc1_iref', 'CORCON_nv_lwc_iref'),
            ('CORCON_nv_lwc2_vcol', 'CORCON_nv_twc_vref'),
            ('CORCON_nv_lwc2_icol', 'CORCON_nv_twc_iref'),
            ('CORCON_nv_lwc2_vref', 'CORCON_nv_lwc_vref'),
            ('CORCON_nv_lwc2_iref', 'CORCON_nv_lwc_iref'),
            ('CORCON_nv_twc_vref',  'CORCON_nv_lwc_vref'),
            ('CORCON_nv_twc_iref',  'CORCON_nv_lwc_iref'),
            ('CLWC1ICOL', 'CLWCICOL'),
            ('CLWC1VCOL', 'CLWCVCOL'),
            ('CLWC1IREF', 'CLWCIREF'),
            ('CLWC1VREF', 'CLWCVREF'),
            ('CLWC2ICOL', 'CTWCIREF'),
            ('CLWC2VCOL', 'CTWCVREF'),
            ('CLWC2IREF', 'CLWCIREF'),
            ('CLWC2VREF', 'CLWCVREF'),
            ('CTWCICOL',  'CTWCICOL'),
            ('CTWCVCOL',  'CTWCVCOL'),
            ('CTWCIREF',  'CLWCIREF'),
            ('CTWCVREF',  'CLWCVREF')
        )

        for var in var_map:
            try:
                self.d[var[0]] = self.d[var[1]]
            except KeyError:
                self.dataset.constants[var[0]] = self.dataset[var[1]]

    def get_fitted_k(
            self, col_p: pd.Series, ref_p: pd.Series, no_cloud_mask: pd.Series,
            nominal_k: float
        ) -> tuple[pd.Series, tuple[LinearFit, LinearFit]]:
        """
        Fit the baseline correction to the Nevzorov data.

        Args:
            col_p: Collector power.
            ref_p: Reference power.
            no_cloud_mask: Mask indicating clear air.
            nominal_k: The nominal baseline correction.

        Returns:
            The fitted baseline correction and the linear fits for the IAS and
            PS dependencies.
        """

        wow = self.dataset['WOW_IND'].reindex(col_p.index).interpolate().bfill()
        ps = self.dataset['PS_RVSM'].reindex(col_p.index).interpolate().bfill()
        roll = self.dataset['ROLL_GIN'].reindex(col_p.index).interpolate().bfill()
        ias = self.dataset['IAS_RVSM'].reindex(col_p.index).interpolate().bfill()
        mask = no_cloud_mask.reindex(col_p.index).ffill().bfill()

        s = slrs(wow, ps, roll, min_length=60, max_length=60)

        k_ias_fit = get_k_ias_fit(col_p, ref_p, ias, mask, nominal_k, s)
        k_ias = get_k_ias_from_fit(k_ias_fit, ias)

        k_ps_fit = get_k_ps_fit(col_p, ref_p, ps, mask, k_ias, nominal_k, s)

        parameterized_k = get_parameterized_k(
            k_ias_fit, k_ps_fit, ias, ps, nominal_k
        )

        return parameterized_k, (k_ias_fit, k_ps_fit)


    def process(self):
        """
        Main processing routine.
        """

        # Get all required variables at the Nevz. sampling frequency.
        self.get_dataframe(index=self.dataset['CORCON_nv_twc_vcol'].index,
                           method='onto', limit=63)

        # Chuck out Nevz data on the ground: it's no use to anyone
        try:
            self.d = self.d.loc[self.d.index > self.dataset.takeoff_time]
        except TypeError:
            # Dataset has no takeoff_time - probably no PRTAFT data
            pass

        try:
            self.d = self.d.loc[self.d.index < self.dataset.landing_time]
        except TypeError:
            # Dataset has no landing_time - probably no PRTAFT data
            pass

        # Create Nevz flag, currently only based on weight on wheels
        self.d['flag'] = 0
        self.d.loc[self.d['WOW_IND'] == 1, 'flag'] = 3

        # Remap variable names iff we're running a new vane type
        self._remap_1t2l1r()

        _vanetype = self.dataset['VANETYPE'].lower()
        if self.test_mode:
            _vanetype = 'all'

        # Measurements are collector current, collector voltage, reference
        # current, reference voltage
        measurements = ('icol', 'vcol', 'iref', 'vref')

        # Energy required to melt/evaporate consensed water
        nvl = self.dataset['CALNVL']

        for ins in self.instruments[_vanetype]:

            # Sensor area and cp / rp ratio in constants file
            _calconst = 'CALNV{ins}'.format(ins=ins.upper())
            _calfit = 'FITNV{ins}'.format(ins=ins.upper())
            area = self.dataset[_calconst][1]
            K = self.dataset[_calconst][0]
            fit_status = CORRECTION_STATUS.UNINITIALISED

            for meas in measurements:
                # Get the raw sensor reading from DECADES
                _var = 'CORCON_nv_{ins}_{meas}'.format(ins=ins, meas=meas)
                raw = self.d[_var]

                # Calibrate raw DECADES reading to Amps / Volts
                _calconst = 'C{ins}{meas}'.format(ins=ins.upper(),
                                                  meas=meas.upper())
                _cals = self.dataset[_calconst]

                _outvar = 'NV_{ins}_{meas}'.format(ins=ins.upper(),
                                                   meas=meas.upper())

                self.d[_outvar] = (_cals[0] + _cals[1] * raw) * _cals[2]

            # Calibrated collector and reference variables for the current
            # sensor
            _col_i_name = 'NV_{ins}_ICOL'.format(ins=ins.upper())
            _col_v_name = 'NV_{ins}_VCOL'.format(ins=ins.upper())
            _ref_i_name = 'NV_{ins}_IREF'.format(ins=ins.upper())
            _ref_v_name = 'NV_{ins}_VREF'.format(ins=ins.upper())

            # Power (W) = I * V
            col_p = self.d[_col_i_name] * self.d[_col_v_name]
            ref_p = self.d[_ref_i_name] * self.d[_ref_v_name]

            if ins == 'twc':
                # Cloud mask is based on variance of power from the total water
                # sensor.
                clear_air = get_no_cloud_mask(col_p, self.d.WOW_IND)

            try:
                fitted_K, (ias_fit, ps_fit) = self.get_fitted_k(
                    col_p, ref_p, clear_air, K
                )
                fit_status = CORRECTION_STATUS.FROM_FLIGHT_DATA
                logger.info(f"Fit from flight data for {ins}: IAS: {ias_fit}, PS: {ps_fit}")

            except Exception:
                # If the fit has failed, we only want to write
                # uncorrected variables
                if self.test_mode:
                    fit_status = CORRECTION_STATUS.FROM_FLIGHT_DATA
                    fitted_K = K
                else:
                    logger.warning('Failed to baseline correct Nevzorov from flight data', exc_info=True)
                    fitted_K = 0
                    
            if fit_status == CORRECTION_STATUS.UNINITIALISED:
                try:
                    (ias_fit, ps_fit) = self.dataset[_calfit]
                    fitted_K = get_parameterized_k(
                        ias_fit, ps_fit, self.d['IAS_RVSM'], self.d['PS_RVSM'], K
                    )
                except Exception:
                    logger.error('Failed to baseline correct Nevzorov from constants', exc_info=True)
                    fit_status = CORRECTION_STATUS.FAILED
                    fitted_K = 0

            # Create and write output variables
            w_c = DecadesVariable(
                (col_p - fitted_K * ref_p) / (self.d.TAS_RVSM * area * nvl),
                name='NV_{ins}_C'.format(ins=ins.upper()),
                flag=DecadesBitmaskFlag,
                comment=get_water_content_comment(fit_status)
            )

            if fit_status in (CORRECTION_STATUS.UNINITIALISED, CORRECTION_STATUS.FAILED):
                w_c.write = False

            self.dataset.add_constant('NV_{ins}_K'.format(ins=ins.upper()), fitted_K)

            col_power = DecadesVariable(
                col_p, name='NV_{ins}_COL_P'.format(ins=ins.upper()),
                flag=DecadesBitmaskFlag
            )

            for _var in (w_c, col_power):
                _var.flag.add_mask(
                    self.d['flag'], flags.WOW, 'The aircraft is on the ground'
                )
                self.add_output(_var)

            if fit_status != CORRECTION_STATUS.FAILED:
                w_c.flag.add_mask(
                    get_baseline_flag(col_p, ref_p, fitted_K, clear_air),
                    'poor clear air baseline',
                    ('The Nevzorov baseline correction is poor in clear air '
                    f'(|dk| > {BASELINE_DEVIATION_LIMIT})')
                )

            if _vanetype in ('1t1l2r', 'all'):
                if ins in ('lwc1', 'lwc2'):
                    continue

                ref_power = DecadesVariable(
                    ref_p, name='NV_{ins}_REF_P'.format(ins=ins.upper()),
                    flag=DecadesBitmaskFlag
                )

                _var = ref_power
                _var.flag.add_mask(
                    self.d['flag'], flags.WOW, 'The aircraft is on the ground'
                )
                self.add_output(_var)

            if _vanetype in ('1t2l1r', 'all'):
                if ins == 'twc':
                    _var = DecadesVariable(
                    ref_p, name='NV_REF_P', flag=DecadesBitmaskFlag
                    )
                    _var.flag.add_mask(
                        self.d['flag'], flags.WOW, 'The aircraft is on the ground'
                    )
                    self.add_output(_var)

        self.add_output(
            DecadesVariable(clear_air, name='NV_CLEAR_AIR_MASK',
            flag=DecadesBitmaskFlag),
        )

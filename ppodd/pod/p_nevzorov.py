"""
This module provides a postprocessing module for the Nevzorov vane in both the
'old' (1T1L2R) and 'new' (1T2L1R) configurations. See class docstring for more
info.
"""
# pylint: disable=invalid-name, too-many-arguments, too-many-locals
# pylint: disable=too-many-statements
import warnings
import logging

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit

from vocal.types import DerivedString

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..decades.attributes import DocAttribute
from .base import PPBase, register_pp
from .shortcuts import _o, _z

logger = logging.getLogger(__name__)


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


def get_fitted_k(col_p, ref_p, ias, ps, no_cloud_mask, k):
    """
    The Nevzorov baseline is not constant, but varies as a function of
    indicated air speed (IAS_RVSM) and static air pressure (PS_RVSM).
    Abel et al. (2014) provide a fitting formula in Appendix A to correct
    the K value (ratio between collector and reference power, when outside of
    clouds) to remove the zero offset of the liquid and total water
    measurements.

    Reference:
        S J Abel, R J Cotton, P A Barrett and A K Vance. A comparison of ice
        water content measurement techniques on the FAAM BAe-146 aircraft.
        Atmospheric Measurement Techniques 7(5):4815--4857, 2014.

    Args:
        col_p: Nevz. collector power (W), pd.Series
        ref_p: Nevz. reference power (W), pd.Series
        ias: Indicated airspeed (m s-1), pd.Series
        ps: Static Pressure (hPa), pd.Series
        no_cloud_mask: mask indicating in (0) or out (1) of cloud, pd.Series
        k: K value given in the flight constants.

    Returns:
        a tuple (K, A), where K is the fitted K value (pd.Series) and A is a
        tuple containing the fit parameters.

    """

    def fit_func(x, a, b):
        """
        (col_pow / ref_pow) - k - [ a(1/ias) + b log_10(Ps) ]
        """
        return (
            x[0, :] / x[1, :] - k - (a * (1 / x[2, :]) + b * np.log10(x[3, :]))
        )

    # Generate a mask that is out of cloud and everywhere-finite for all input
    # variables
    mask = ((no_cloud_mask == 1) & (np.isfinite(col_p)) & (np.isfinite(ref_p))
            & (np.isfinite(ias)) & (np.isfinite(ps)))

    # Create input data array for fit_func
    xdata = np.vstack([
        col_p.loc[mask].values,
        ref_p.loc[mask].values,
        ias.loc[mask].values,
        ps.loc[mask].values
    ])

    # linter complains that this may be an unbalances unpacking, however the
    # docs indicate that this isn't the case.
    # pylint: disable=unbalanced-tuple-unpacking
    popt, _ = curve_fit(fit_func, xdata, xdata[0, :] * 0.0)

    return (k + (popt[0] * (1. / ias) + popt[1] * np.log10(ps)), popt)


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
    `NV_TWC_U`, `NV_TWC_C`, `NV_LWC_U`, `NV_LWC_C`. If a new-style ``1T2L1R``
    vane is fitted, then the default outputs are `NV_TWC_U`, `NV_TWC_C`,
    `NV_LWC1_U`, `NV_LWC1_C`, `NV_LWC2_U`, `NV_LWC2_C`.
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
            'WOW_IND': ('data', _z(100), 1)
        }

    def _declare_outputs_common(self):
        """
        Declare the outputs that are common between both Nevz. vanes.
        """

        self.declare(
            'NV_TWC_U',
            units='gram m-3',
            frequency=64,
            long_name=('Uncorrected total condensed water content from the '
                       'Nevzorov probe'),
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN']
        )

        self.declare(
            'NV_TWC_C',
            units='gram m-3',
            frequency=64,
            long_name=('Corrected total condensed water content from the '
                       'Nevzorov probe'),
            comment='Automatically baselined. May require further processing.',
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
            write=False
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
            'NV_LWC_U',
            units='gram m-3',
            frequency=64,
            long_name=('Uncorrected liquid water content from the Nevzorov '
                       'probe'),
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN']
        )

        self.declare(
            'NV_LWC_C',
            units='gram m-3',
            frequency=64,
            long_name='Corrected liquid water content from the Nevzorov probe',
            standard_name='mass_concentration_of_liquid_water_in_air',
            comment='Automatically baselined. May require further processing.',
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
            write=False
        )

        self.declare(
            'NV_LWC_COL_P',
            units='W',
            frequency=64,
            long_name='LWC collector power',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN'],
            write=False
        )

        self.declare(
            'NV_LWC_REF_P',
            units='W',
            frequency=64,
            long_name='LWC reference power',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN'],
            write=False
        )

    def _declare_outputs_1t2l1r(self):
        """
        Declare the outputs that are only valid for the 1t2l1r vane type.
        """

        self.declare(
            'NV_LWC1_U',
            units='gram m-3',
            frequency=64,
            long_name=('Uncorrected liquid water content from the Nevzorov '
                       'probe (1st collector)'),
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN']
        )

        self.declare(
            'NV_LWC1_C',
            units='gram m-3',
            frequency=64,
            long_name=('Corrected liquid water content from the Nevzorov probe'
                       ' (1st collector)'),
            standard_name='mass_concentration_of_liquid_water_in_air',
            comment='Automatically baselined. May require further processing.',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN']
        )

        self.declare(
            'NV_LWC2_U',
            units='gram m-3',
            frequency=64,
            long_name=('Uncorrected liquid water content from the Nevzorov '
                       'probe (2nd collector)'),
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
            comment='Automatically baselined. May require further processing.',
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
            write=False
        )

        self.declare(
            'NV_LWC1_COL_P',
            units='W',
            frequency=64,
            long_name='LWC1 collector power',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN'],
            write=False
        )

        self.declare(
            'NV_LWC2_COL_P',
            units='W',
            frequency=64,
            long_name='LWC2 collector power',
            instrument_manufacturer='Sky Phys Tech Inc.',
            instrument_model=self.dataset.lazy['VANETYPE'],
            instrument_serial_number=self.dataset.lazy['VANE_SN'],
            write=False
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
            area = self.dataset[_calconst][1]
            K = self.dataset[_calconst][0]

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
                fitted_K, _ = get_fitted_k(
                    col_p, ref_p, self.d.IAS_RVSM, self.d.PS_RVSM, clear_air, K
                )

                fit_success = True
            except (RuntimeError, ValueError) as e:
                # If the fit has failed, we only want to write
                # uncorrected variables
                if self.test_mode:
                    fit_success = True
                    fitted_K = K
                else:
                    logger.warning('Failed to baseline correct Nevzorov')
                    logger.warning(str(e))
                    fitted_K = 0
                    fit_success = False

            # Create and write output variables
            w_c = DecadesVariable(
                (col_p - fitted_K * ref_p) / (self.d.TAS_RVSM * area * nvl),
                name='NV_{ins}_C'.format(
                    ins=ins.upper()
                ), flag=DecadesBitmaskFlag
            )
            if not fit_success:
                w_c.write = False

            w_u = DecadesVariable(
                (col_p - K * ref_p) / (self.d.TAS_RVSM * area * nvl),
                name='NV_{ins}_U'.format(ins=ins.upper()),
                flag=DecadesBitmaskFlag
            )

            col_power = DecadesVariable(
                col_p, name='NV_{ins}_COL_P'.format(ins=ins.upper()),
                flag=DecadesBitmaskFlag
            )

            for _var in (w_c, w_u, col_power):
                _var.flag.add_mask(
                    self.d['flag'], flags.WOW, 'The aircraft is on the ground'
                )
                self.add_output(_var)

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

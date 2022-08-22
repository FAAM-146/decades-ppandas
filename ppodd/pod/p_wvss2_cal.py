"""
Provides a processing module which calculates a mach number for moist
air.
"""
# pylint: disable=invalid-name
import numpy as np
import pandas as pd

from vocal.schema_types import DerivedString

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..decades.attributes import DocAttribute
from ..utils.calcs import sp_mach
from .base import PPBase, register_pp
from .shortcuts import _o, _z


@register_pp('core')
class WVSS2Calibrated(PPBase):
    r"""
    For further details see the `FAAM Met. Handbook <https://doi.org/10.5281/zenodo.5846962>`_.
    
    This module provides a calibrated WVSS2 volume mixing ratio, along with its
    uncertainty estimate, from the flush mounted instrument. The calibration is
    a polynomial in VMR, with coefficients provided in the constants parameter
    ``WVSS2_F_CAL``. The calibration is assumed to only be valid within a
    certain VMR range, specified in the constants parameter
    ``WVSS2_F_CAL_RANGE``. Outside of this range, data are set to NaN.

    The uncertainty estimate, :math:`U_c`, is given by

    .. math::

        U_c = \sqrt{\sigma_f^2 + \sigma_r^2 + \sigma_b^2},

    where

    .. math::

        \sigma_f &= \sigma_0 + \sigma_1 V + ... + \sigma_n V^n\\
        \sigma_b &= \frac{l_0 + l_1 V}{2}\\
        \sigma_r &= V\left(p_0 V^{p_1}\right)

    where :math:`V` is the uncorrected volume mixing ratio, :math:`\sigma_0,
    ..., \sigma_n` are specified in the constants parameter
    ``WVSS2_F_UNC_FITPARAMS``, :math:`l_0 \text{ and} l_1` are specified in the
    constants parameter ``WVSS2_F_UNC_BUCK`` and :math:`p_0 \text{ and} p_1`
    are specified in the constants parameter ``WVSS2_F_UNC_FITRES``.
    """

    inputs = [
        'WVSS2F_VMR_U',
        'WOW_IND',
        'WVSS2_F_CAL',
        'WVSS2_F_CAL_RANGE',
        'WVSS2_F_SN',
        'WVSS2_F_UNC_FITPARAMS',
        'WVSS2_F_UNC_BUCK',
        'WVSS2_F_UNC_FITRES'
    ]

    @staticmethod
    def test():
        """
        Provide dummy input data for testing.
        """
        n = 100
        return {
            'WVSS2F_VMR_U': ('data', 1E5 * _o(n), 1),
            'WOW_IND': ('data', _z(n), 1),
            'WVSS2_F_CAL': ('const', [1, 0, 0]),
            'WVSS2_F_CAL_RANGE': ('const', [0, 20000]),
            'WVSS2_F_SN': ('const', DocAttribute(value='1234', doc_value=DerivedString)),
            'WVSS2_F_UNC_FITPARAMS': ('const', [1, 0, 0, 0, 0, 0]),
            'WVSS2_F_UNC_BUCK': ('const', [5, 0]),
            'WVSS2_F_UNC_FITRES': ('const', [8, 0, 1])
        }

    def declare_outputs(self):
        """
        Declare outputs created by this module.
        """
        self.declare(
            'WVSS2F_VMR_C',
            units='ppm',
            frequency=1,
            long_name=('Calibrated volume mixing ratio from WVSS2F'),
            instrument_manufacturer='SpectraSensors',
            instrument_serial_number=self.dataset.lazy['WVSS2_F_SN']
        )

        self.declare(
            'WVSS2F_VMR_C_CU',
            units='ppm',
            frequency=1,
            long_name=('Uncertainty estimate for calibrated volume mixing '
                       'ratio from WVSS2F'),
            instrument_manufacturer='SpectraSensors',
            instrument_serial_number=self.dataset.lazy['WVSS2_F_SN'],
            coverage_content_type='auxiliaryInformation',
            flag=None
        )



    def get_corrected_vmr(self):
        d = self.d
        fit = self.dataset['WVSS2_F_CAL'][::-1]
        vmr_unc = d.WVSS2F_VMR_U
        vmr_corr = pd.Series(np.polyval(fit, vmr_unc), index=vmr_unc.index)

        out_range = self.get_out_range()
        vmr_corr[out_range] = np.nan

        return vmr_corr

    def get_uncertainty(self):
        d = self.d
        vmr_u = d.WVSS2F_VMR_U
        popt_quintic_sigma_f = self.dataset['WVSS2_F_UNC_FITPARAMS'][::-1]
        popt_linear = self.dataset['WVSS2_F_UNC_BUCK'][::-1]
        popt_power = self.dataset['WVSS2_F_UNC_FITRES']

        sigma_f_data = np.polyval(popt_quintic_sigma_f, vmr_u)
        sigma_b_data = 0.5 * np.polyval(popt_linear, vmr_u)
        sigma_r_data = vmr_u * (popt_power[0] * vmr_u**popt_power[1])

        u_wvss2c = (sigma_f_data**2. + sigma_r_data**2. + sigma_b_data**2.)**0.5

        return u_wvss2c

    def get_out_range(self):
        d = self.d
        vmr_unc = d.WVSS2F_VMR_U
        val_range = self.dataset['WVSS2_F_CAL_RANGE']
        out_range = (vmr_unc < val_range[0]) | (vmr_unc > val_range[1])
        return out_range

    def process(self):
        """
        Processing entry hook.
        """
        self.get_dataframe()
        d = self.d
        wow = d.WOW_IND.fillna(method='bfill').fillna(method='ffill')
        vmr_corr = self.get_corrected_vmr()
        vmr_corr_cu = self.get_uncertainty()

        vmr_corr_out = DecadesVariable(
            vmr_corr, name='WVSS2F_VMR_C',
            flag=DecadesBitmaskFlag
        )

        vmr_corr_cu_out = DecadesVariable(
            vmr_corr_cu, name='WVSS2F_VMR_C_CU', flag=None
        )

        out_range = self.get_out_range()
        val_range = self.dataset['WVSS2_F_CAL_RANGE']

        vmr_corr_out.flag.add_mask(
            out_range, flags.OUT_RANGE,
            (f'VMR is outside calibration range '
                f'[{val_range[0]} {val_range[1]}]')
        )

        vmr_corr_out.flag.add_mask(
            wow, flags.WOW,
            ('Aircraft is on the ground, as indicated by the '
                'weight-on-wheels indicator.')
        )

        for var in (vmr_corr_out, vmr_corr_cu_out):
            self.add_output(var)

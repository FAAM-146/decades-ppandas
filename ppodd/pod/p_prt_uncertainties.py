"""
Provides a processing module which calculates uncertainties for indicated and
true air temperatures measured with PRTs.
"""
# pylint: disable=invalid-name
import numpy as np

from functools import reduce

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..utils.calcs import sp_mach
from .base import PPBase, register_pp
from .shortcuts import _o

PRT_SENSORS = ('plate', 'loom')


@register_pp('core')
class PRTTemperatureUncertainties(PPBase):
    r"""
    This module calculates combined uncertainty estimates for indicated and
    true air temperatures, when these are recorded with platinum resistance
    thermometers.

    Uncertainties in the indicated temperatures are derived from:

    * Uncertainty in the thermometer calibration from NPL
    * Drift of sensors between NPL calibrations
    * Resolution uncertainty in the DECADES DLU
    * DECADES calibration uncertainty
    * Noise in the system when a fixed resistor is fitted
    * Keithley calibration
    * Keithley stability
    * DECADES calibration (counts to resistance) residual
    * NPL calibration (resistance to temperature) residual.

    Uncertainties in the true air temperatures are derived from:

    * Uncertainty of the corresponding indicated air temperature
    * Uncertainty in the Mach number
    * Uncertainty in the ratio of specific heats
    * Uncertainty in the variable housing recovery factor.
    """

    inputs = [
        'TAT_DI_R',
        'TAT_ND_R',
        'IAT_DI_R',
        'IAT_ND_R',
        'IT_PRT_UNC_NPLCAL',
        'ITDI_PRT_UNC_DRIFT',
        'ITND_PRT_UNC_DRIFT',
        'IT_UNC_DECADESRES',
        'ITDI_UNC_DECADES',
        'ITND_UNC_DECADES',
        'IT_UNC_NOISE',
        'IT_UNC_DVMCAL',
        'IT_UNC_DVMSPEC',
        'IT_UNC_DVMRES',
        'ITDI_UNC_RESIDUALC2R',
        'ITND_UNC_RESIDUALC2R',
        'ITDI_UNC_RESIDUALR2T',
        'ITND_UNC_RESIDUALR2T',
        'IT_UNC_NPLRES',
        'DITSENS',
        'NDTSENS',
        'MACH',
        'MACH_CU',
        'SH_GAMMA',
        'SH_GAMMA_CU',
        'ETA_ND',
        'ETA_ND_CU',
        'ETA_DI',
        'ETA_DI_CU'
    ]

    @staticmethod
    def test():
        """
        Provide dummy input data for testing.
        """
        n = 100
        return {
            'TAT_DI_R': ('data', 200 * _o(n), 32),
            'TAT_ND_R': ('data', 200 * _o(n), 32),
            'IAT_DI_R': ('data', 200 * _o(n), 32),
            'IAT_ND_R': ('data', 200 * _o(n), 32),
            'IT_PRT_UNC_NPLCAL': ('const', [[[0, 100], .1], [[100, 400], .2]]),
            'ITDI_PRT_UNC_DRIFT': ('const', {'plate': .1, 'loom': .1}),
            'ITND_PRT_UNC_DRIFT': ('const', {'plate': .1, 'loom': .1}),
            'IT_UNC_DECADESRES': ('const', .05),
            'ITDI_UNC_DECADES': ('const', .1),
            'ITND_UNC_DECADES': ('const', .1),
            'IT_UNC_NOISE': ('const', .005),
            'IT_UNC_DVMCAL': ('const', .05),
            'IT_UNC_DVMSPEC': ('const', 0.01),
            'IT_UNC_DVMRES': ('const', 0.001),
            'ITDI_UNC_RESIDUALC2R': ('const', 0.01),
            'ITND_UNC_RESIDUALC2R': ('const', 0.01),
            'ITDI_UNC_RESIDUALR2T': ('const', {'plate': 0.01, 'loom': 0.01}),
            'ITND_UNC_RESIDUALR2T': ('const', {'plate': 0.01, 'loom': 0.01}),
            'IT_UNC_NPLRES': ('const', 0.01),
            'DITSENS': ('const', ['xxxx', 'plate']),
            'NDTSENS': ('const', ['xxxx', 'plate']),
            'MACH': ('data', .5 * _o(n), 32),
            'MACH_CU': ('data', 0.001 * _o(n), 32),
            'SH_GAMMA': ('data', 1.4 * _o(n), 32),
            'SH_GAMMA_CU': ('data', .01 * _o(n), 32),
            'ETA_ND': ('data', .01 * _o(n), 32),
            'ETA_ND_CU': ('data', .001 * _o(n), 32),
            'ETA_DI': ('data', .01 * _o(n), 32),
            'ETA_DI_CU': ('data', .001 * _o(n), 32)
        }

    def declare_outputs(self):
        """
        Declare outputs created by this module.
        """
        nd_sens = self.dataset['NDTSENS'][1]
        di_sens = self.dataset['DITSENS'][1]

        if di_sens in PRT_SENSORS:
            self.declare(
                'IAT_DI_R_CU',
                units='K',
                frequency=32,
                long_name=('Combined uncertainty estimate for IAT_DI_R'),
                write=False
            )

            self.declare(
                'TAT_DI_R_CU',
                units='K',
                frequency=32,
                long_name=('Combined uncertainty estimate for TAT_DI_R'),
            )


        if nd_sens in PRT_SENSORS:
            self.declare(
                'IAT_ND_R_CU',
                units='K',
                frequency=32,
                long_name=('Combined uncertainty estimate for IAT_ND_R'),
                write=False
            )

            self.declare(
                'TAT_ND_R_CU',
                units='K',
                frequency=32,
                long_name=('Combined uncertainty estimate for TAT_ND_R'),
            )

    def _combine_unc(self, uncs):
        return sum([i**2 for i in uncs])**.5

    def get_it_unc(self, nddi):
        d = self.d

        if nddi == 'DI':
            sens_type = self.dataset['DITSENS'][1]
        elif nddi == 'ND':
            sens_type = self.dataset['NDTSENS'][1]
        else:
            raise ValueError(f'Unknown sensor type {nddi}')

        u_it_npltemp = 0. * d[f'IAT_{nddi}_R'].copy()
        for _range, val in self.dataset['IT_PRT_UNC_NPLCAL']:
            mask = (
                (d[f'IAT_{nddi}_R'] >= _range[0])
              & (d[f'IAT_{nddi}_R'] < _range[1])
            )
            u_it_npltemp[mask] = val

        u_itdi_drift = self.dataset[f'IT{nddi}_PRT_UNC_DRIFT'][sens_type]
        u_it_res = self.dataset['IT_UNC_DECADESRES']
        u_itdi_decades = self.dataset[f'IT{nddi}_UNC_DECADES']
        u_it_noise = self.dataset['IT_UNC_NOISE']
        u_it_dvmcal = self.dataset['IT_UNC_DVMCAL']
        u_it_dvmspec = self.dataset['IT_UNC_DVMSPEC']
        u_it_dvmres = self.dataset['IT_UNC_DVMRES']
        u_itdi_residualc2r = self.dataset[f'IT{nddi}_UNC_RESIDUALC2R']
        u_itdi_residualr2t = self.dataset[f'IT{nddi}_UNC_RESIDUALR2T'][sens_type]
        u_it_nplres = self.dataset['IT_UNC_NPLRES']

        uncs = [u_itdi_drift, u_it_res, u_itdi_decades, u_it_noise,
                u_it_dvmcal, u_it_dvmspec, u_it_dvmres, u_itdi_residualc2r,
                u_itdi_residualr2t, u_it_nplres, u_it_npltemp]

        return self._combine_unc(uncs)

    def get_itdi_unc(self):
        return self.get_it_unc('DI')

    def get_itnd_unc(self):
        return self.get_it_unc('ND')

    def get_tat_unc(self, nddi):
        d = self.d
        eta = d[f'ETA_{nddi}']
        mach_moist = d['MACH']
        gamma = d['SH_GAMMA']
        it = d[f'IAT_{nddi}_R']
        u_eta = d[f'ETA_{nddi}_CU']
        u_mach = d['MACH_CU']
        u_gamma = d['SH_GAMMA_CU']
        u_it = d[f'IAT_{nddi}_R_CU']

        dTsdTi = 1. / ((1. - eta) * (0.5 * mach_moist**2 * (gamma - 1.) + 1.))
        dTsdeta = (2. * it) / ((1. - eta)**2 * (mach_moist**2 * (gamma - 1.) + 2.))
        dTsdM = (4 * it * (gamma - 1.) * mach_moist) / ((eta - 1.) * ((gamma - 1.) * mach_moist**2 + 2.)**2)
        dTsdgamma = (2 * it * mach_moist**2) / ((eta - 1.) * (mach_moist**2 * (gamma - 1.) + 2.)**2)

        Titerm = dTsdTi**2 * u_it**2
        etaterm = dTsdeta**2 * u_eta**2
        Mterm = dTsdM**2 * u_mach**2
        gammaterm = dTsdgamma**2 * u_gamma**2

        return (Titerm + etaterm + Mterm + gammaterm)**0.5

    def _process_sensor(self, sensor):
        """
        Process the uncertainties for a specific sensor.

        Args:
            sensor: The sensor to process. Either 'ND' for the non-deiced
                    sensor, or 'DI' for the deiced sensor.
        """
        fn_map = {
            'ND': self.get_itnd_unc,
            'DI': self.get_itdi_unc
        }

        unc_it = fn_map[sensor]()
        self.d[f'IAT_{sensor}_R_CU'] = unc_it

        unc_tat = self.get_tat_unc(sensor)

        self.add_output(
            DecadesVariable(unc_it, name=f'IAT_{sensor}_R_CU',
                            flag=None)
        )

        self.add_output(
            DecadesVariable(unc_tat, name=f'TAT_{sensor}_R_CU',
                            flag=None)
        )


    def process(self):
        """
        Processing entry hook.
        """
        self.get_dataframe()
        d = self.d

        nd_sens = self.dataset['NDTSENS'][1]
        di_sens = self.dataset['DITSENS'][1]

        if nd_sens in PRT_SENSORS:
            self._process_sensor('ND')

        if di_sens in PRT_SENSORS:
            self._process_sensor('DI')

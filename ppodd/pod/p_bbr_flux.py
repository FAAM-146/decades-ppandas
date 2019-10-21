import numpy as np

from .base import PPBase
from ..decades import DecadesVariable

import matplotlib.pyplot as plt

class BBRFlux(PPBase):

    inputs = [
        'CALCUCF', 'CALCURF', 'CALCUIF', 'CALCLCF', 'CALCLRF', 'CALCLIF',
        'UP1S', 'UP2S', 'UIRS', 'UP1Z', 'UP2Z', 'UIRZ', 'UP1T', 'UP2T', 'UIRT',
        'LP1S', 'LP2S', 'LIRS', 'LP1Z', 'LP2Z', 'LIRZ', 'LP1T', 'LP2T', 'LIRT',
        'SOL_AZIM', 'SOL_ZEN', 'ROLL_GIN', 'PTCH_GIN', 'HDG_GIN'
    ]

    def declare_outputs(self):

        self.declare(
            'SW_DN_C',
            units='W m-2',
            frequency=1,
            number=1019,
            standard_name='downwelling_shortwave_flux_in_air',
            long_name='Corrected downward short wave irradiance, clear dome'
        )

        self.declare(
            'RED_DN_C',
            units='W m-2',
            frequency=1,
            number=1020,
            long_name='Corrected downward short wave irradiance, red dome'
        )

        if False:
            # Masked as we now only run KippZonen Pyrgeometers
            self.declare(
                'IR_DN_C',
                units='W m-2',
                frequency=1,
                number=1021,
                long_name='Corrected downward long wave irradiance'
            )

        self.declare(
            'SW_UP_C',
            units='W m-2',
            frequency=1,
            number=1019,
            standard_name='upwelling_shortwave_flux_in_air',
            long_name='Corrected upward short wave irradiance, clear dome'
        )

        self.declare(
            'RED_UP_C',
            units='W m-2',
            frequency=1,
            number=1020,
            long_name='Corrected upward short wave irradiance, red dome'
        )

        if False:
            # Masked as we now only run KippZonen Pyrgeometers
            self.declare(
                'IR_UP_C',
                units='W m-2',
                frequency=1,
                number=1021,
                long_name='Corrected upward long wave irradiance'
            )

    def corr_thm(self, therm):
        """
        Correct thermistors for linearity. TODO: find a reference for this
        """

        rcon = -0.774
        v = 6.08E-02
        w = 2.47E-03
        x = -6.29E-05
        y = -8.78E-07
        z = 1.37E-08

        rt = therm - 273.15

        therm_c = (
            rt + (rcon + rt * (v + rt * (w + rt * (x + rt * (y + rt * z)))))
        )

        return therm_c

    def process(self):

        self.get_dataframe(method='onto', index=self.dataset['UP1S'].index,
                           circular='HDG_GIN')
        d = self.d

        deg2rad = 360. / (2 * np.pi)

        d['ZENRAD'] = d.SOL_ZEN / deg2rad
        d['AZMRAD'] = d.SOL_AZIM / deg2rad
        d['HDGRAD'] = d.HDG_GIN / deg2rad
        d['SUNHDG'] = d.AZMRAD - d.HDGRAD

        d['FCRIT'] = 920. * (np.cos(d.ZENRAD))**1.28

        ceff = np.array(
            [1.010, 1.005, 1.005, 1.005, 1.000, 0.995, 0.985, 0.970,
             0.930, 0.930]
        )

        fdir = np.array([.95] * 10)

        # Create 2s running means for GIN pitch and roll
        for gin in ('PTCH', 'ROLL'):
            d['{}_GIN_rmean'.format(gin)] = (
                d['{}_GIN'.format(gin)].rolling(2, center=True).mean()
            )

        for dome in ('P1', 'P2'):
            for pos in ('U', 'L'):

                # Create 10s running mean series of the instrument zero offsets
                _label = '{pos}{dome}Z'.format(pos=pos, dome=dome)
                d['{}_rmean'.format(_label)] = d[_label].rolling(10, center=True).mean()

                _therm_label_cor = '{pos}{dome}T_c'.format(pos=pos, dome=dome)
                _therm_label_raw = '{pos}{dome}T'.format(pos=pos, dome=dome)
                d[_therm_label_cor] = self.corr_thm(d[_therm_label_raw])

                _caldict = {
                    'P1': 'C',
                    'P2': 'R'
                }

                _calname = 'CALC{pos}{ins}F'.format(
                    pos=pos, ins=_caldict[dome]
                )

                # Apply pitch and roll offset corrections
                roll = d.ROLL_GIN_rmean + self.dataset[_calname][4]
                roll = roll / deg2rad
                pitch = d.PTCH_GIN_rmean + self.dataset[_calname][3]
                pitch = pitch / deg2rad

                # Correction for pitch and roll in direct radiation
                cos_beta = (
                    np.sin(roll) * np.sin(d.ZENRAD) * np.sin(d.SUNHDG)
                    + np.cos(roll) * np.cos(pitch) * np.cos(d.ZENRAD)
                    - np.cos(roll) * np.sin(pitch) * np.sin(d.ZENRAD) *
                    np.cos(d.SUNHDG)
                )

                _flux = '{}{}_flux'.format(pos, dome)

                # Thermisor corrections for linearity
                tsa, tsb, tsg = self.dataset[_calname][:3]
                th = d[_therm_label_cor]

                # Remove zero offset from the signal to obtain a flux
                _signal = '{}{}S'.format(pos, dome)
                _zero = '{}{}Z_rmean'.format(pos, dome)
                d[_flux] =  d[_signal] - d[_zero]

                # Perform temperature sensitivity correction
                _temp = d[_flux] / (1. + th * (tsa + th * (tsb + th * tsg)))
                d[_flux] = _temp

                # Make a copy of the critical value (diffuse vs direct)
                fcritval = d.FCRIT.copy(deep=True)
                if dome is 'P2':
                    fcritval /= 2

                # This is a horrible way to do this, essentially ripped
                # directly from the old FORTRAN code, but it does the job.
                index = np.round(d.SOL_ZEN / 10)

                d['_ceff'] = np.array([ceff[int(i)] if np.isfinite(i) else np.nan
                                      for i in index])

                d['_fdir'] = np.array([fdir[int(i)] if np.isfinite(i) else np.nan
                                      for i in index])

                # For the upper BBRs, apply the pitch and roll corrections when
                # in direct sunlight (flux <=> fcrit)
                if pos is 'U':
                    _above_crit = d[_flux] / (
                        1. - (d._fdir * (1. - d._ceff * (cos_beta / np.cos(d.ZENRAD))))
                    )

                    d[_flux].loc[d[_flux] > fcritval] = _above_crit.loc[d[_flux] > fcritval]

        # Create and add outputs
        sw_dn_c = DecadesVariable(d['UP1_flux'], name='SW_DN_C')
        sw_up_c = DecadesVariable(d['LP1_flux'], name='SW_UP_C')
        red_dn_c = DecadesVariable(d['UP2_flux'], name='RED_DN_C')
        red_up_c = DecadesVariable(d['LP2_flux'], name='RED_UP_C')

        for var in (sw_up_c, sw_dn_c, red_up_c, red_dn_c):
            self.add_output(var)

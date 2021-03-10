"""
This module provides the post processing module BBRFlux, which takes signals
from the Broadband Radiometers, calculated in upsteam modules, and calculates
radiative fluxes from these. See the class docstring for mor information.
"""
# pylint: disable=invalid-name
import numpy as np

from .base import PPBase
from .shortcuts import _l, _o, _z, _c
from ..decades import DecadesVariable, DecadesBitmaskFlag

# Limits for flagging
ROLL_LIMIT = 7.
SUN_ANGLE_MAX = 80.

FLUX_LIMITS = {
    'UP_SW_MAX': 1380.,
    'UP_RED_MAX': 700.,
    'DN_SW_MAX': 1380.,
    'DN_RED_MAX': 700.,

    'UP_SW_MIN': -20.,
    'UP_RED_MIN': -20.,
    'DN_SW_MIN': -20.,
    'DN_RED_MIN': -20.
}


class BBRFlux(PPBase):
    r"""
    Caluclates the corrected fluxes from the upper and lower clear dome and red
    dome pyranometers.

    The thermistors in the pyranometers have a characteristic non-linear
    temperature dependence due to the manufacturing process. If not corrected
    for, this can lead to errors in temperature of up to 1 :math:`^\circ\text{C}`.
    A quintic equation has been fitted to the manufacturer provided corrections
    for a range of temperatures, providing a correction between
    :math:`-50 ^\circ` C and :math:`40 ^\circ` C to within
    :math:`\pm0.07` :math:`^\circ` C.

    .. math::
        T_c = T + \left(\alpha_0 + T\left(\alpha_1 + T\left(\alpha_2 +
        T\left(\alpha_3 + T\left(\alpha_4 +
        T\alpha_5\right)\right)\right)\right)\right).

    The polynomial coefficents :math:`\alpha_0\ldots\alpha_5` are hard-coded, and
    take the values

    .. math::
        \left[-0.774, 6.08\times10^{-2}, 2.47\times10^{-3},
        -6.29\times10^{-5}, -8.78\times10^{-7}, 1.37\times10^{-8}\right].

    The flux for each dome, :math:`F_d`, is calculated by subtracting a 10 second
    running mean of the zero from the signal. The flux is then corrected for
    temperature sensitivity using

    .. math::
        F_{d_c} = \frac{F_d}{1 + T_c\left(\gamma_1 + T_c\left(\gamma_2 +
        T_c\gamma_3\right)\right)},

    where :math:`T_c` is the corrected dome thermistor temperature, and
    :math:`\gamma_n` are the first :math:`n` values in the dome constants array.

    A threshold value, :math:`F_{\text{crit}} = 920\cos(\zeta)^{1.28}`, is used
    to determine whether the dome is in direct or diffuse radiation, with fluxes
    above :math:`F_{\text{crit}}` (or :math:`F_{\text{crit}} / 2` for red domes)
    assumed to indicate direct radiation. This expression for
    :math:`F_{\text{crit}}` approximates the 'German' equation (ref?) but is
    simpler and remains positive at low sun elevations. If the flux is
    determined to be direct, then the upper radiometers are corrected for the
    pitch and roll of the aircraft (Ref: M/MRF/13/5):

    .. math::
        F = \frac{F_{d_c}}{1 -f_r(\zeta)\left(1-c(\zeta)\frac{\cos\beta}{\cos\zeta}
        \right)}.

    Here, :math:`f_r(\zeta)` is the ratio of of direct:direct+diffuse radiation,
    currently assumed to be ``0.95`` for all solar zenith angles,
    :math:`c(\zeta)` is a correction term for the cosine effect (Ref: Tech note
    8, table 4). The angle between the solar zenith and normal-to-instrument,
    :math:`\beta`, is given by

    .. math::
        \cos\beta &= \sin\phi\sin\zeta\sin\psi \\
                  &+ \cos\phi\cos\theta\cos\zeta \\
                  &- \cos\phi\sin\theta\sin\zeta\cos\psi,

    where :math:`\phi` is the aircraft roll, :math:`\zeta` is the solar zenith
    angle, :math:`\psi` is the 'sun heading', the difference between the solar
    azimuth angle and the aircraft heading, and :math:`\theta` is the aircraft
    pitch angle. Ref: Tech. note 7, page 10. Prior to this correction, platform
    relative pitch and roll offsets, determined through flying clear sky box
    patterns, are added to the instrument-derived pitch and roll. These are
    given as elements 4 and 5 in the flight constants for each dome.
    """

    inputs = [
        'CALCUCF', 'CALCURF', 'CALCUIF', 'CALCLCF', 'CALCLRF', 'CALCLIF',
        'UP1S', 'UP2S', 'UIRS', 'UP1Z', 'UP2Z', 'UIRZ', 'UP1T', 'UP2T', 'UIRT',
        'LP1S', 'LP2S', 'LIRS', 'LP1Z', 'LP2Z', 'LIRZ', 'LP1T', 'LP2T', 'LIRT',
        'SOL_AZIM', 'SOL_ZEN', 'ROLL_GIN', 'PTCH_GIN', 'HDG_GIN'
    ]

    @staticmethod
    def test():
        """
        Return some dummy input data for testing purposes.
        """
        return {
            'CALCUCF': ('const', [0, 0, 0, -3, 0, 1]),
            'CALCURF': ('const', [0, 0, 0, -3, 0, 1]),
            'CALCUIF': ('const', [0, 0, 0, -3, 0, 1]),
            'CALCLCF': ('const', [0, 0, 0, 0, 0, 1]),
            'CALCLRF': ('const', [0, 0, 0, 0, 0, 1]),
            'CALCLIF': ('const', [0, 0, 0, 0, 0, 1]),
            'UP1S': ('data', 600 * _o(100)),
            'UP2S': ('data', 300 * _o(100)),
            'UIRS': ('data', _z(100)),
            'UP1Z': ('data', _z(100)),
            'UP2Z': ('data', _z(100)),
            'UIRZ': ('data', _z(100)),
            'UP1T': ('data', 250 * _o(100)),
            'UP2T': ('data', 250 * _o(100)),
            'UIRT': ('data', 400 * _o(100)),
            'LP1S': ('data', 300 * _o(100)),
            'LP2S': ('data', 150 * _o(100)),
            'LIRS': ('data', _z(100)),
            'LP1Z': ('data', _z(100)),
            'LP2Z': ('data', _z(100)),
            'LIRZ': ('data', _z(100)),
            'LP1T': ('data', 250 * _o(100)),
            'LP2T': ('data', 250 * _o(100)),
            'LIRT': ('data', 400 * _o(100)),
            'SOL_AZIM': ('data', _l(200, 250, 100)),
            'SOL_ZEN': ('data', _l(25, 50, 100)),
            'ROLL_GIN': ('data', _c([_l(0, 5, 50), _l(5, 0, 50)])),
            'PTCH_GIN': ('data', 6 * _o(100)),
            'HDG_GIN': ('data', _z(100))
        }

    def declare_outputs(self):
        """
        Declare the output variables that this modules produces.
        """
        manufacturer = 'Kipp and Zonen'

        self.declare(
            'SW_DN_C',
            units='W m-2',
            frequency=1,
            standard_name='downwelling_shortwave_flux_in_air',
            long_name='Corrected downward short wave irradiance, clear dome',
            instrument_manufacturer=manufacturer,
            instrument_serial_number=self.dataset.lazy['BBRUP1_SN']
        )

        self.declare(
            'RED_DN_C',
            units='W m-2',
            frequency=1,
            long_name='Corrected downward short wave irradiance, red dome',
            instrument_manufacturer=manufacturer,
            instrument_serial_number=self.dataset.lazy['BBRUP2_SN']
        )

        self.declare(
            'SW_UP_C',
            units='W m-2',
            frequency=1,
            standard_name='upwelling_shortwave_flux_in_air',
            long_name='Corrected upward short wave irradiance, clear dome',
            instrument_manufacturer=manufacturer,
            instrument_serial_number=self.dataset.lazy['BBRLP1_SN']
        )

        self.declare(
            'RED_UP_C',
            units='W m-2',
            frequency=1,
            long_name='Corrected upward short wave irradiance, red dome',
            instrument_manufacturer=manufacturer,
            instrument_serial_number=self.dataset.lazy['BBRLP2_SN']
        )

    @staticmethod
    def corr_thm(therm):
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
        """
        Processing entry hook.
        """
        # pylint: disable=too-many-statements, too-many-locals

        # Collect the required inputs
        self.get_dataframe(method='onto', index=self.dataset['UP1S'].index,
                           circular='HDG_GIN')
        d = self.d

        deg2rad = 360. / (2 * np.pi)

        # Convert angular info to radians
        d['ZENRAD'] = d.SOL_ZEN / deg2rad
        d['AZMRAD'] = d.SOL_AZIM / deg2rad
        d['HDGRAD'] = d.HDG_GIN / deg2rad
        d['SUNHDG'] = d.AZMRAD - d.HDGRAD

        # The critical value to distinguish between direct and diffuse
        # radiation
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

                # Angle of the sun - for low sun flagging
                beta = np.arccos(cos_beta)

                # Key for the flux from the current dome
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

                # Red dome has half the critical value
                if dome == 'P2':
                    fcritval = fcritval / 2

                # This is a horrible way to do this, essentially ripped
                # directly from the old FORTRAN code, but it does the job.
                index = np.round(d.SOL_ZEN / 10)

                d['_ceff'] = np.array([ceff[min(int(i), 9)] if np.isfinite(i)
                                       else np.nan for i in index])

                d['_fdir'] = np.array([fdir[min(int(i), 9)] if np.isfinite(i)
                                       else np.nan for i in index])

                # For the upper BBRs, apply the pitch and roll corrections when
                # in direct sunlight (flux >= fcrit)
                # Note that the underscores here aren't really private
                # attributes, just badly named DataFrame columns
                # pylint: disable=protected-access
                if pos == 'U':
                    _above_crit = d[_flux] / (
                        1. - (d._fdir * (1. - d._ceff * (cos_beta / np.cos(d.ZENRAD))))
                    )
                    _direct = d[_flux] > fcritval
                    d.loc[_direct, _flux] = _above_crit[_direct]

                output_dome_dict = {
                    'P1': 'SW',
                    'P2': 'RED'
                }

                output_pos_dict = {
                    'U': 'DN',
                    'L': 'UP'
                }

                output_name = '{wavelength}_{pos}_C'.format(
                    wavelength=output_dome_dict[dome],
                    pos=output_pos_dict[pos]
                )

                # Create and add outputs
                output = DecadesVariable(d[_flux], name=output_name,
                                         flag=DecadesBitmaskFlag)

                # Flag when the aircraft is in a significant roll
                output.flag.add_mask(
                    np.abs(d.ROLL_GIN_rmean) >= ROLL_LIMIT,
                    'roll limit exceeded',
                    ('The aircraft is in a roll exceeding the specified max '
                     f'roll limit of {ROLL_LIMIT} degrees from horizontal.')
                )

                # Flag when sun angle is too low
                output.flag.add_mask(
                    beta > SUN_ANGLE_MAX / deg2rad,
                    'low sun angle',
                    ('The sun is low relative to the axis of the aircraft, '
                     'exceeding the maximum allowed limit of '
                     f'{SUN_ANGLE_MAX} degrees.')
                )

                # Flag when out of range
                flux_limit_lo = FLUX_LIMITS['{}_{}_MIN'.format(
                    output_pos_dict[pos],
                    output_dome_dict[dome]
                )]

                flux_limit_hi = FLUX_LIMITS['{}_{}_MAX'.format(
                    output_pos_dict[pos],
                    output_dome_dict[dome]
                )]

                output.flag.add_mask(
                    (d[_flux] < flux_limit_lo) | (d[_flux] > flux_limit_hi),
                    'flux out of range',
                    ('The calculated is outside of the specified allowable '
                     f'flux range [{flux_limit_lo}, {flux_limit_hi}] W/m2')
                )

                self.add_output(output)

"""
Provides a post processing module for data from the RVSM (air data computer)
system, recorded on the rear core console via the ARINC 429 data bus. See the
class docstring for more information.
"""
# pylint: disable=invalid-name
import datetime
import pandas as pd
import numpy as np

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..utils import pd_freq
from .base import PPBase, register_pp
from .shortcuts import _l, _o

PALT_MIN = -2000
PALT_MAX = 50000

IAS_MIN = -50
IAS_MAX = 500


@register_pp('core')
class Rvsm(PPBase):
    r"""
    Calculate derived parameters from the aircraft's RVSM system. Pressure
    altitude and indicated air speed are obtained from the aircraft's ARINC-429
    data bus, with a least significant bit resolution of 4 ft and :math:`1/32`
    kts respectively.

    Static pressure, :math:`P`, is obtained from the pressure altitude,
    :math:`Z_p`, using the ICAO standard atmosphere,

    .. math::
        P = P_0\left(\frac{T_0}{T_0 + L_0 \left(Z_p - h_0\right)}\right)^{\frac{g_0
        M}{R L_0}},

    where :math:`T_0=288.15`, :math:`L_0=-0.0065`, :math:`h_0=0`,
    :math:`g_0=9.80655`, :math:`M=0.0289644`, :math:`R=8.31432`,
    :math:`P_0=1013.25` below 11000 m, or

    .. math::
        P = P_1\exp\left(\frac{-g_0 M \left(Z_p - h_1\right)}{R T_1}\right),

    where :math:`T_1=216.65`, :math:`P_1=226.321`, :math:`h_1=11000`, above 11000 m.

    Pitot static pressure, :math:`q`, is given as

    .. math::
        q = P \left(\frac{M^2}{5} + 1\right)^{\frac{7}{2}} - 1,

    with the Mach number, :math:`M`, given by

    .. math::
        M = \frac{V_{IAS}}{V_0 \sqrt{\frac{P}{P_0}}},

    where :math:`V_0 = 340.294` and :math:`P_0=1013.25`, and :math:`V_{IAS}` is
    the indicated air speed.

    Data are flagged where either the pressure altitude or indicated air speed
    are considered out of range.
    """

    inputs = ['PRTAFT_pressure_alt', 'PRTAFT_ind_air_speed']

    @staticmethod
    def test():
        """
        Return dummy input data for testing.
        """
        return {
            'PRTAFT_pressure_alt': ('data', _l(0, 3000, 100), 32),
            'PRTAFT_ind_air_speed': ('data', 150 * _o(100), 32)
        }

    def declare_outputs(self):
        """
        Declare the outputs produced by this module.
        """
        self.declare(
            'PS_RVSM',
            units='hPa',
            frequency=32,
            standard_name='air_pressure',
            long_name=('Static pressure from the aircraft RVSM (air data) '
                       'system'),
            instrument_manufacturer='BAE Systems'
        )

        self.declare(
            'Q_RVSM',
            units='hPa',
            frequency=32,
            standard_name=None,
            long_name=('Pitot static pressure inverted from RVSM (air data) '
                       'system indicated airspeed'),
            instrument_manufacturer='BAE Systems'
        )

        self.declare(
            'PALT_RVS',
            units='m',
            frequency=32,
            standard_name='barometric_altitude',
            long_name=('Pressure altitude from the aircraft RVSM (air data) '
                       'system'),
            instrument_manufacturer='BAE Systems'
        )

    def calc_altitude(self):
        """
        Calculate the pressure altitude from the pressure altitude signal
        recorded on the rear core console from the ARINC 429 data bus.
        """
        d = self.d

        d['PALT_FEET'] = d['PRTAFT_pressure_alt'] * 4
        d['FLAG_ALT'] = 0

        d.loc[d['PALT_FEET'] < PALT_MIN, 'FLAG_ALT'] = 1
        d.loc[d['PALT_FEET'] > PALT_MAX, 'FLAG_ALT'] = 1

        d['PALT_METRES'] = d['PALT_FEET'] / 3.28084

    def calc_pressure(self):
        """
        Calculate pressure from pressure altitude, assuming a standard
        atmosphere.
        """
        d = self.d

        high = self.d['PALT_METRES'] > 11000

        T0 = 288.15
        L0 = -0.0065
        h0 = 0.0
        go = 9.80665
        M = 0.0289644
        R = 8.31432
        P0 = 1013.2500

        # Calulate pressure from standard atmosphere
        d['P'] = P0 * (
            T0 / (T0 + L0 * (d['PALT_METRES'] - h0))
        )**(go * M / (R * L0))

        T1 = 216.65
        P1 = 226.3210
        h1 = 11000.0

        d.loc[high, 'P'] = P1 * np.exp(
            (-go * M * (d['PALT_METRES'].loc[high] - h1)) / (R * T1)
        )

    def calc_ias(self):
        """
        Calculate indicated air speed from the indicated air speed signal
        recorded on the rear core console from the ARINC-429 data bus.
        """
        d = self.d
        d['IAS'] = d['PRTAFT_ind_air_speed'] * 0.514444 / 32.0
        d['IAS_FLAG'] = 0
        d.loc[d['IAS'] < IAS_MIN, 'IAS_FLAG'] = 1
        d.loc[d['IAS'] > IAS_MAX, 'IAS_FLAG'] = 1

    def calc_mach(self):
        """
        Calculate mach number from the indicated air speed and pressure,
        assuming a standard atmosphere.
        """
        d = self.d
        d['MACH'] = d['IAS'] / (340.294 * np.sqrt(d['P'] / 1013.25))

    def calc_pitot(self):
        """
        Calculate pitot-static pressure from static pressure and Mach number.
        """
        d = self.d
        d['PITOT'] = d['P'] * ((((d['MACH']**2) / 5 + 1)**3.5) - 1)

    def calc_ps_rvsm(self):
        """
        Produces the PS_RVSM output.

        Returns:
            ps_rvsm: a DecadesVariable containing PS_RVSM, the static air
                pressure from the RVSM air-data system.
        """
        d = self.d

        ps_rvsm = d['P']

        ps_rvsm = DecadesVariable(ps_rvsm, name='PS_RVSM',
                                  flag=DecadesBitmaskFlag)

        ps_rvsm.flag.add_mask(
            d.FLAG_ALT, 'altitude out of range',
            f'Pressure altitude outside acceptable range '
            f'[{PALT_MIN}, {PALT_MAX}]'
        )

        return ps_rvsm

    def calc_palt_rvs(self):
        """
        Produces the PALT_RVS output.

        Returns:
            palt_rvs: a DecadesVariable containing PALT_RVS, the pressure
                altitude from the RVSM air-data system.
        """
        d = self.d

        palt_rvs = d['PALT_METRES']

        palt_rvs = DecadesVariable(palt_rvs, name='PALT_RVS',
                                   flag=DecadesBitmaskFlag)

        palt_rvs.flag.add_mask(
            d.FLAG_ALT, 'altitude out of range',
            f'Pressure altitude outside acceptable range '
            f'[{PALT_MIN}, {PALT_MAX}]'
        )

        return palt_rvs

    def calc_q_rvsm(self):
        """
        Returns the Q_RVSM output.

        Returns:
            q_rvsm: a DecadesVariable containing Q_RVSM, the dynamic pressure
                derived from the RVSM air-data system.
        """
        d = self.d
        q_rvsm = d['PITOT']

        q_rvsm = DecadesVariable(q_rvsm, name='Q_RVSM',
                                 flag=DecadesBitmaskFlag)

        q_rvsm.flag.add_mask(
            d.FLAG_ALT, 'altitude out of range',
             f'Pressure altitude outside acceptable range '
             f'[{PALT_MIN}, {PALT_MAX}]'
        )
        q_rvsm.flag.add_mask(
            d.IAS_FLAG, 'ias out of range',
            f'Indicated air speed outside acceptable range '
            f'[{IAS_MIN}, {IAS_MAX}]'
        )

        return q_rvsm

    def process(self):
        """
        Processing entry hook.
        """

        _start, _end = self.dataset[self.inputs[0]].time_bounds()

        # This is a bit hacky, but it ensures we have a correct end point when
        # interpolating from 20 Hz to 32 Hz.
        _end += datetime.timedelta(
            seconds=(1/self.dataset[self.inputs[0]].frequency)*.99
        )

        # Generate the index we want, and build the dataframe
        _index = pd.date_range(start=_start, end=_end, freq=pd_freq[32])
        self.get_dataframe(method='onto', index=_index, limit=2)

        self.calc_altitude()
        self.calc_pressure()
        self.calc_ias()
        self.calc_mach()
        self.calc_pitot()

        ps_rvsm = self.calc_ps_rvsm()
        q_rvsm = self.calc_q_rvsm()
        palt_rvs = self.calc_palt_rvs()

        self.add_output(ps_rvsm)
        self.add_output(palt_rvs)
        self.add_output(q_rvsm)

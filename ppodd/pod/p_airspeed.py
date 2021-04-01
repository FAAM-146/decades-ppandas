"""
This module provides the postprocessing module AirSpeed, which calculates the
aircraft airspeeds from the RVSM system. See the class docstring for further
details.
"""
# pylint: disable=invalid-name

import numpy as np

from ppodd.decades import DecadesVariable, DecadesBitmaskFlag
from ppodd.utils.calcs import sp_mach
from ppodd.utils.constants import SPEED_OF_SOUND, ICAO_STD_TEMP, ICAO_STD_PRESS
from ppodd.pod.base import PPBase, register_pp
from ppodd.pod.shortcuts import _l, _o


@register_pp('core')
class AirSpeed(PPBase):
    r"""
    Calculates aircraft indicated and true air speeds. Mach number, :math:`M`,
    is calculated from static and dynamic pressures (here :math:`p` and
    :math:`q`, derived as ``PS_RVSM`` and ``Q_RVSM``, in processing
    module ``p_rvsm.py``) using the standard calculation ``sp_mach`` defined in
    ppodd.utils.calcs:

    .. math::
        M = \sqrt{5\left(1 + \frac{q}{p}\right)^{2/7} - 1}

    Indicated airspeed is then given as

    .. math::
        \text{IAS} = V_s M \sqrt{\frac{p}{P_\text{std}}},

    where :math:`V_s` is the speed of sound at standard temperature and
    pressure, and :math:`P_{std}` is the surface pressure in the ICAO standard
    atmosphere.

    True airspeed is given as

    .. math::
        \text{TAS} = T_c V_s M \sqrt{\frac{T_\text{di}}{T_\text{std}}},

    where :math:`T_c` is a TAS correction term, defined in the flight constants,
    :math:`T_\text{di}` is the temperature from the de-iced temperature sensor,
    and :math:`T_\text{std}` is the surface temperature in the ICAO standard
    atmosphere.
    """

    inputs = [
        'TASCORR',          #  Airspeed correction factor (const)
        'PS_RVSM',          #  Static Pressure (derived)
        'Q_RVSM',           #  Pitot-static pressure (derived)
        'TAT_DI_R'          #  Deiced true air temp (derived)
    ]

    @staticmethod
    def test():
        """
        Return some dummy input data for testing usage.
        """
        return {
            'TASCORR': ('const', 1.),
            'PS_RVSM': ('data', _l(1000, 300, 100), 32),
            'Q_RVSM': ('data', 250. * _o(100), 32),
            'TAT_DI_R': ('data', _l(290, 250, 100), 32)
        }

    def declare_outputs(self):
        """
        Declare all of the output variables produced by this module, through
        calls to self.declare.
        """

        self.declare(
            'IAS_RVSM',
            units='m s-1',
            frequency=32,
            long_name=('Indicated air speed from the aircraft RVSM '
                       '(air data) system')
        )

        self.declare(
            'TAS_RVSM',
            units='m s-1',
            frequency=32,
            long_name=('True air speed from the aircraft RVSM (air data) '
                       'system and deiced temperature'),
            standard_name='platform_speed_wrt_air'
        )

    def calc_ias(self):
        """
        Calculate indicated airspeed from the RVSM system. Store this in the
        instance dataframe.
        """
        d = self.d

        ias = (SPEED_OF_SOUND * d['MACHNO'] *
               np.sqrt(d['PS_RVSM'] / ICAO_STD_PRESS))

        d['IAS_RVSM'] = ias

    def calc_tas(self):
        """
        Calculate true airspeed from the RVSM system and the deiced
        temperature. Store this in the instance dataframe.
        """
        d = self.d

        tas = (
            self.dataset['TASCORR']
            * SPEED_OF_SOUND
            * d['MACHNO']
            * np.sqrt(d['TAT_DI_R'] / ICAO_STD_TEMP)
        )

        d['TAS_RVSM'] = tas

    def calc_mach(self):
        """
        Calculate the mach number from the RVSM air data computer. Store this
        in the instance dataframe as MACHNO.
        """
        d = self.d

        d['MACHNO'], d['MACHNO_FLAG'] = sp_mach(
            d['Q_RVSM'], d['PS_RVSM'], flag=True
        )

    def process(self):
        """
        Module entry hook.
        """

        # Get all of the required inputs.
        self.get_dataframe()

        # Run required calculations in turn. These are stored in instance
        # state.
        self.calc_mach()
        self.calc_ias()
        self.calc_tas()

        # Create output variables for the indicated and true airspeeds.
        ias = DecadesVariable(self.d['IAS_RVSM'], flag=DecadesBitmaskFlag)
        tas = DecadesVariable(self.d['TAS_RVSM'], flag=DecadesBitmaskFlag)

        # Flag the data wherever the mach number is out of range.
        for _var in (ias, tas):
            _var.flag.add_mask(
                self.d['MACHNO_FLAG'],
                'mach out of range',
                ('Either static or dynamic pressure out of acceptable limits '
                 'during calculation of mach number.')
            )
            self.add_output(_var)

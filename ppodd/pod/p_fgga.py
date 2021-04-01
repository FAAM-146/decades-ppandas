"""
Provides a postprocessing module for the S10 pressure transducer. See class
docstring for more info.
"""
import numpy as np

from ..decades import DecadesVariable, DecadesClassicFlag
from ..decades import flags
from .base import PPBase, register_pp
from .shortcuts import _o


@register_pp('fgga')
class FGGA(PPBase):
    r"""
    Calculate static pressure from the S10 fuselage ports. Static pressure is
    calulated by applying a polynomial transformation to the DLU signal. The
    coefficients, specified in the flight constants parameter ``CALS10SP``,
    combine both the DLU and pressure transducer calibrations.

    Data are flagged when they are considered out of range.
    """

    inputs = [
        'FGGA_CO2',
        'FGGA_CO2_FLAG',
        'FGGA_CH4',
        'FGGA_CH4_FLAG'
    ]

    @staticmethod
    def test():
        """
        Return dummy inputs for testing.
        """
        return {
            'CALS10SP': ('const', [-130, .035, 1.85e-9]),
            'S10SP_SN': ('const', 'xxxx'),
            'CORCON_s10_press': ('data', 850 * _o(100), 32),
        }

    def declare_outputs(self):
        """
        Declare the outputs that are going to be written by this module.
        """

        fgga_manufacturer = 'Los Gatos Research Inc.'
        fgga_model = self.dataset.lazy['FGGA_MODEL']
        fgga_serial = self.dataset.lazy['FGGA_SN']

        self.declare(
            'co2_mol_frac',
            units='1e-6',
            practical_units='ppm',
            frequency=1,
            long_name='Carbon Dioxide (CO2) dry mole fraction',
            standard_name='mole_fraction_of_carbon_dioxide_in_air',
            instrument_manufacturer=fgga_manufacturer,
            instrument_model=fgga_model,
            instrument_serial_number=fgga_serial
        )

        self.declare(
            'ch4_mol_frac',
            units='1e-9',
            practical_units='ppb',
            frequency=1,
            long_name='Methane (CH4) dry mole fraction',
            standard_name='mole_fraction_of_methane_in_air',
            instrument_manufacturer=fgga_manufacturer,
            instrument_model=fgga_model,
            instrument_serial_number=fgga_serial
        )

    def process(self):
        """
        Processing entry point.
        """
        self.get_dataframe()


        co2 = DecadesVariable(
            self.d['FGGA_CO2'], name='co2_mol_frac',
            flag=DecadesClassicFlag
        )

        ch4 = DecadesVariable(
            self.d['FGGA_CH4'], name='ch4_mol_frac',
            flag=DecadesClassicFlag
        )

        meanings = (
            (1, 'reduced accuracy or out of bounds'),
            (2, 'duty cycle not in measure mode')
        )

        for var in (co2, ch4):
            for meaning in meanings:
                var.flag.add_meaning(*meaning)

        co2.flag.add_flag(self.d['FGGA_CO2_FLAG'])
        ch4.flag.add_flag(self.d['FGGA_CH4_FLAG'])

        self.add_output(co2)
        self.add_output(ch4)

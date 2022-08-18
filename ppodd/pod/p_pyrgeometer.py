"""
This module provides a postprocessing module for the Kipp and Zonen CR4
pyrgeometers.
"""
# pylint: disable=invalid-name, too-many-locals
import numpy as np

from vocal.schema_types import DerivedString

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..decades.attributes import DocAttribute
from .base import PPBase, register_pp
from .shortcuts import _c, _o, _z
from ..utils.constants import STEF_BOLTZ


def thermistor(resistance):
    """
    The thermistor is a YSI-44031. Formual is taken from the spec sheet
    supplied by Kipp & Zonen.

    Args:
        resistance: measured resistance in Ohms.

    Returns:
        the temperaturem in Kelvin.
    """

    alpha = 1.0295 * (10**-3)
    beta = 2.391 * (10**-4)
    gamma = 1.568 * (10**-7)
    T = (alpha + (beta * np.log(resistance) + gamma * np.log(resistance)**3))**-1
    return T


def crg4(ampage, temperature):
    """
    Formula for the Kipp and Zonen CRG4 Pyranometer for calculating the
    longwave flux using the ampage and temperature output.

    No calibration coefficients are needed, because the Mapbox carries the
    sensor specific calibration.

    Args:
        ampage: the measured ampage, in mA
        temperature: the body temperature of the pyrgeometer, in Kelvin

    Returns:
        L_d: the radiation flux, in W m-2.
    """

    ioset = 4.0
    gain = 50.0
    eoset = 600.0
    L_d = (ampage - ioset) * gain + (STEF_BOLTZ * (temperature**4)) - eoset

    return L_d


@register_pp('core')
class KippZonenPyrgeometer(PPBase):
    r"""
    Calculation of longwave fluxes from the upward and downward facing
    Kipp and Zonen CR4 Pyrgeometers.

    The 0 - 32 mV output of the CR4 thermopile is mapped to a 4 - 20 mA signal
    in the amp box, which carries sensor specific calibrations, corresponding
    to a flux range of :math:`-600` - :math:`200` Wm\ :math:`^{-2}`. This is then
    converted to a voltage using a 350 :math:`\Omega` resistor, which is recorded
    in the DLU, with 16 bits covering a :math:`-10` - :math:`10` V range.
    Similarly, the thermistor is placed in parallel with a 100 k\ :math:`\Omega`
    linearising resistor, and 100 :math:`\mu`\ A is passed through the
    combination, with the resulting voltage measured at the DLU.

    This module first applies the inverse transformations to recover the amp
    box current and the thermistor resistance. The thermistor temperature is
    given by

    .. math::
        T = \left(\alpha + \left(\beta\log\left(R\right) +
        \gamma\log\left(R\right)^3\right)\right)^{-1},

    where :math:`R` is the thermistor resistance and :math:`\alpha`,
    :math:`\beta`, and :math:`\gamma` are calibration coefficients supplied by
    the manufacturer. The longwave flux, :math:`L_D`, is then given by

    .. math::
        L_D = \beta(A - \alpha) - \gamma + \sigma T^4,

    where :math:`\alpha = 4`, :math:`\beta=50`, and :math:`\gamma=600` map the
    current from the amp box, :math:`A`, onto the specified range of flux
    values, :math:`T` is the temperature recorded by the thermistor, and
    :math:`\sigma` is the Stefan-Boltzmann constant.
    """

    inputs = [
        'LOWBBR_radiometer_3_sig',
        'LOWBBR_radiometer_3_temp',
        'UPPBBR_radiometer_3_sig',
        'UPPBBR_radiometer_3_temp',
        'WOW_IND'
    ]

    @staticmethod
    def test():
        """
        Return dummy input data for testing.
        """
        return {
            'BBRLIR_SN': ('const', DocAttribute(value='1234', doc_value=DerivedString)),
            'BBRUIR_SN': ('const', DocAttribute(value='1234', doc_value=DerivedString)),
            'LOWBBR_radiometer_3_sig': ('data', 5e3 * _o(100), 1),
            'LOWBBR_radiometer_3_temp': ('data', 2e2 * _o(100), 1),
            'UPPBBR_radiometer_3_sig': ('data', 5e3 * _o(100), 1),
            'UPPBBR_radiometer_3_temp': ('data', 2e2 * _o(100), 1),
            'WOW_IND': ('data', _c([_o(30), _z(50), _o(20)]), 1)
        }

    def declare_outputs(self):
        """
        Declare module outputs.
        """

        self.declare(
            'IR_DN_C',
            units='W m-2',
            frequency=1,
            long_name='Corrected downward longwave irradiance',
            instrument_manufacturer='Kipp and Zonen',
            instrument_model='CR4',
            instrument_serial_number=self.dataset.lazy['BBRUIR_SN']
        )

        self.declare(
            'IR_UP_C',
            units='W m-2',
            frequency=1,
            long_name='Corrected upward longwave irradiance',
            instrument_manufacturer='Kipp and Zonen',
            instrument_model='CR4',
            instrument_serial_number=self.dataset.lazy['BBRLIR_SN']
        )

    def process(self):
        """
        Processing entry hook.
        """
        self.get_dataframe()
        d = self.d

        # CRIO DLU specific characteristics
        dlu_range = 20.     # +-10 Range Volt
        resolution = 2**16  # bit

        # Convert DLU raw counts to Voltage
        low_sig_v = d.LOWBBR_radiometer_3_sig * (dlu_range / resolution)
        upp_sig_v = d.UPPBBR_radiometer_3_sig * (dlu_range / resolution)

        # Convert DLU raw counts to Kelvin
        low_temp_v = d.LOWBBR_radiometer_3_temp * (dlu_range / resolution)
        upp_temp_v = d.UPPBBR_radiometer_3_temp * (dlu_range / resolution)

        # Temperature
        low_temp_tot_ohm = low_temp_v / 100e-6
        low_temp_ohm = 1. / ((1. / low_temp_tot_ohm) - 1e-5)

        upp_temp_tot_ohm = upp_temp_v / 100e-6
        upp_temp_ohm = 1. / ((1. / upp_temp_tot_ohm) - 1e-5)

        # Calculate instrument body temperature
        upp_cr4_temp = thermistor(upp_temp_ohm)
        low_cr4_temp = thermistor(low_temp_ohm)

        # Ampbox
        low_ampbox_output = low_sig_v * 1000. / 350.
        upp_ampbox_output = upp_sig_v * 1000. / 350.

        # Calculate longwave radiation
        low_l_d = crg4(low_ampbox_output, low_cr4_temp)
        upp_l_d = crg4(upp_ampbox_output, upp_cr4_temp)

        # Flagging
        d['WOW_FLAG'] = 0
        d.loc[d.WOW_IND == 1, 'WOW_FLAG'] = 1

        # Create output variables
        ir_up = DecadesVariable(
            low_l_d, name='IR_UP_C', flag=DecadesBitmaskFlag
        )

        ir_dn = DecadesVariable(
            upp_l_d, name='IR_DN_C', flag=DecadesBitmaskFlag
        )

        for var in (ir_up, ir_dn):
            var.flag.add_mask(
                d['WOW_FLAG'], flags.WOW, 'Aircraft is on the ground'
            )

        self.add_output(ir_up)
        self.add_output(ir_dn)

import numpy as np

from ..decades import DecadesVariable
from .base import PPBase
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


class KippZonenPyrgeometer(PPBase):
    """
    Calculating of the upward and downward long wave fluxes from the
    Kipp & Zonen CR4 Pyrgeometers.
    """

    inputs = [
        'LOWBBR_radiometer_3_sig',
        'LOWBBR_radiometer_3_temp',
        'UPPBBR_radiometer_3_sig',
        'UPPBBR_radiometer_3_temp',
        'WOW_IND'
    ]

    def declare_outputs(self):
        """
        Declare module outputs.
        """

        self.declare(
            'IR_DN_C',
            units='W m-2',
            frequency=1,
            number=1021,
            long_name='Corrected downward longwave irradiance',
            write=False
        )

        self.declare(
            'IR_UP_C',
            units='W m-2',
            frequency=1,
            number=1024,
            long_name='Corrected upward longwave irradiance',
            write=False
        )

    def process(self):
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
        ir_up = DecadesVariable(upp_l_d, name='IR_UP_C')
        ir_up.add_flag(d['WOW_FLAG'])

        ir_dn = DecadesVariable(low_l_d, name='IR_DN_C')
        ir_dn.add_flag(d['WOW_FLAG'])

        self.add_output(ir_up)
        self.add_output(ir_dn)

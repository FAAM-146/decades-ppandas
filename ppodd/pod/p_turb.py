import numpy as np
import pandas as pd

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from ..utils import get_range_flag
from ..utils.calcs import sp_mach
from .base import PPBase

MAX_ITERS = 5

class TurbProbe(PPBase):

    inputs = [
        'AOA_A0',    # (const) Coeffs of 2nd-o poly in Mach to calc AOA offset
        'AOA_A1',    # (const) Coeffs of 2nd-o poly in Mach to calc AOA sensitivity
        'AOSS_B0',   # (const) Coeffs of 2nd-o poly in Mach to calc AOSS offset
        'AOSS_B1',   # (const) Coeffs of 2nd-o poly in Mach to calc AOSS sensivity
        'TOLER',     # (const) Tolerance to AOA/AOSS iteration
        'TASCOR1',   # (const) True airspeed correction factor (fudge factor to remove
                     #         residual along-heading wind errors).
        'ALPH0',     # (const) Coeff 0 of linear correction to calculated AoA
        'ALPH1',     # (const) Coeff 1 of linear correction to calculated AoA
        'BET0',      # (const) Coeff 0 of linear correction to calculated AoSS
        'BET1',      # (const) Coeff 1 of linear correction to calculated AoSS
        'IAS_RVSM',  # (derived) Indicated Air speed
        'TAT_DI_R',  # (derived) Deiced True air temperature
        'TAT_ND_R',  # (derived) Nondeiced True air temperature
        'PS_RVSM',   # (derived) RVSM static pressure
        'Q_RVSM',    # (derived) RVSM pitot-static pressure
        'PALT_RVS',  # (derived) RVSM pressure altitude
        'P0_S10',    # (derived) P0 - S10 pressure differential
        'PA_TURB',   # (derived)
        'PB_TURB',   # (derived)
        'TBPC',      # (derived)
        'TBPD'       # (derived)
    ]

    def declare_outputs(self):

        self.declare(
            'AOA',
            units='degree',
            frequency=32,
            long_name=('Angle of attack from the turbulence probe (positive, '
                       'flow upwards wrt a/c axes)')
        )

        self.declare(
            'AOSS',
            units='degree',
            frequency=32,
            long_name=('Angle of sideslip from the turbulence probe '
                       '(positive, flow from left)')
        )

        self.declare(
            'TAS',
            units='m s-1',
            frequency=32,
            long_name='True airspeed (dry-air) from turbulence probe',
            standard_name='platform_speed_wrt_air'
        )

        self.declare(
            'PSP_TURB',
            units='hPa',
            frequency=32,
            long_name=('Pitot-static pressure from centre-port measurements '
                       'corrrected for AoA and AoSS')
        )

    def s10_position_err_correction(self, amach):
        """
        Calculate values of the S10 static pressure position error as a function
        of Mach number, derived from B001-B012 calibrations.

        Originally written by Phil Brown. Should these be in the constants file?

        Args:
            amach: Mach number.

        Returns:
            dcp_s10: the s10 pressure position error.
        """
        a0 = -0.011482
        a1 = -0.148295
        a2 = 0.407040

        dcp_s10 = a0 + amach * (a1 + amach * a2)

    def process(self):
        self.get_dataframe()
        d = self.d

        amach, mach_flag = sp_mach(self.d.Q_RVSM, self.d.PS_RVSM, flag=True)

        # Calculate a first guess at AOA and AOSS
        a0 = np.polyval(self.dataset['AOA_A0'][::-1], amach)
        a1 = np.polyval(self.dataset['AOA_A1'][::-1], amach)
        b0 = np.polyval(self.dataset['AOSS_B0'][::-1], amach)
        b1 = np.polyval(self.dataset['AOSS_B1'][::-1], amach)

        aoa = (d.PA_TURB / d.Q_RVSM - a0) / a1
        aoss = (d.PB_TURB / d.Q_RVSM - b0) / b1

        aoa[self.d.IAS_RVSM < 50] = np.nan
        aoss[self.d.IAS_RVSM < 50] = np.nan

        # Calculate position error in S10 static pressure
        dcp_s10 = self.s10_position_err_correction(amach)

        # Calculate and apply flow angle corrections to derive true pitot
        # pressure from centre-port measurement.
        dcpa = 0.0273 + aoa * (-0.0141 + aoa * (0.00193 - aoa * 5.2E-5))
        dcpb = aoss * (aoss * 7.6172E-4)

        # Apply corrections to measured differential pressure
        q = d.P0_S10 + (dcpa + dcpb) * d.Q_RVSM

        # Recalculate mach number
        amach = sp_mach(q, self.d.PS_RVSM)

        itern = 0
        while True:
            itern += 1

            a0 = np.polyval(self.dataset['AOA_A0'][::-1], amach)
            a1 = np.polyval(self.dataset['AOA_A1'][::-1], amach)
            b0 = np.polyval(self.dataset['AOSS_B0'][::-1], amach)
            b1 = np.polyval(self.dataset['AOSS_B1'][::-1], amach)

            aoa_new  = (d.PA_TURB / q - a0) / a1
            aoss_new = (d.PB_TURB / q - b0) / b1

            d_aoa = aoa_new - aoa
            d_aoss = aoss_new - aoss

            # Mask out anything with a low IAS, to prevent non-convergence
            # issues
            aoa_new[self.d.IAS_RVSM < 50] = np.nan
            aoss_new[self.d.IAS_RVSM < 50] = np.nan

            aoa = aoa_new
            aoss = aoss_new

            # Recalculate position error
            dcp_s10 = self.s10_position_err_correction(amach)

            # Recalculate flow-angle corrections to centre-port.
            dcpa = 0.0273 + aoa * (-0.0141 + aoa * (0.00193- aoa * 5.2E-5))
            dcpb = aoss * (aoss * 7.6172E-4)

            q = d.P0_S10 + (dcpa + dcpb) * q

            amach = sp_mach(q, d.PS_RVSM)

            tol = self.dataset['TOLER']
            if np.max(np.abs(d_aoa)) < tol and np.max(np.abs(d_aoss)) < tol:
                break

            if itern > MAX_ITERS:
                break

        tas = (
            self.dataset['TASCOR1'] * 340.294 * amach
            * np.sqrt(d.TAT_DI_R / 288.15)
        )

        # Apply linear corrections to calculated AoA and AoSS, derived from Al
        # Rodi analysis - minimization of residual vertical wind during yawing
        # orbits.
        aoa = aoa * self.dataset['ALPH1'] + self.dataset['ALPH0']
        aoss = aoss * self.dataset['BET1'] + self.dataset['BET0']

        aoa_out = DecadesVariable(
            pd.Series(aoa, index=d.index), name='AOA',
            flag=DecadesBitmaskFlag
        )

        aoss_out = DecadesVariable(
            pd.Series(aoss, index=d.index), name='AOSS',
            flag=DecadesBitmaskFlag
        )

        tas_out = DecadesVariable(
            pd.Series(tas, index=d.index), name='TAS',
            flag=DecadesBitmaskFlag
        )

        psp_turb_out = DecadesVariable(
            pd.Series(q, index=d.index), name='PSP_TURB',
            flag=DecadesBitmaskFlag
        )

        # Get out-of-range flags
        tas_flag = get_range_flag(tas, (50, 250))
        aoa_flag = get_range_flag(aoa, (0, 15))
        aoss_flag = get_range_flag(aoss, (-5, 5))

        # Add flags to variables
        aoa_out.flag.add_mask(aoa_flag, 'aoa out of range')
        aoa_out.flag.add_mask(tas_flag, 'tas out of range')
        aoa_out.flag.add_mask(mach_flag, 'mach out of range')
        aoss_out.flag.add_mask(aoss_flag, 'aoss out of range')
        aoss_out.flag.add_mask(tas_flag, 'tas out of range')
        aoss_out.flag.add_mask(mach_flag, 'mach out of range')
        tas_out.flag.add_mask(tas_flag, flags.OUT_RANGE)
        psp_turb_out.flag.add_mask(tas_flag, 'tas out of range')

        # Add outputs to the dataset
        self.add_output(aoa_out)
        self.add_output(aoss_out)
        self.add_output(tas_out)
        self.add_output(psp_turb_out)

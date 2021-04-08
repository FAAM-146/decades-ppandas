"""
Provides a postprocessing module for CCN data
"""
import numpy as np
import pandas as pd

from ..decades import DecadesVariable, DecadesClassicFlag
from ..decades import flags
from .base import PPBase, register_pp
from .shortcuts import _o, _z
#from .p_gin import gin_name, gin_model, gin_manufacturer

def col_bool(columns, name):
    return [i == name for i in columns]

@register_pp('ccn')
class CCN(PPBase):
    r"""
    This module postprocesses CCN data...
    """

    inputs = [
        'CCN_Number_Conc_A',
        'CCN_Number_Conc_B',
        'CCN_T_OPC_A',
        'CCN_T_OPC_B',
        'CCN_Sample_Pressure_A',
        'CCN_Sample_Pressure_B',
        'CCN_T1_Read_A',
        'CCN_T1_Read_B',
        'CCN_T2_Read_A',
        'CCN_T2_Read_B',
        'CCN_T3_Read_A',
        'CCN_T3_Read_B',
        'CCN_Sheath_Flow_A',
        'CCN_Sample_Flow_A',
        'CCN_Sheath_Flow_B',
        'CCN_Sample_Flow_B',
        'CCN_Current_SS_A',
        'CCN_Current_SS_B',
        'CCN_Temps_Stabilized_A',
        'CCN_Temps_Stabilized_B',
        'CCN_Current_SS_A',
        'CCN_Current_SS_B',
        'CCN_T1_Set_A',
        'CCN_T1_Set_B',
        'CCN_T_Sample_A',
        'CCN_T_Sample_B',
        'TAT_DI_R',
        'PS_RVSM',
        'LAT_GIN',
        'LON_GIN',
        'ALT_GIN'
    ]

    @staticmethod
    def test():
        """
        Return dummy inputs for testing.
        """
        n = 100
        return {
            'FGGA_CO2': ('data', 420 * _o(n), 1),
            'FGGA_CO2_FLAG': ('data', _z(n), 1),
            'FGGA_CH4': ('data', 1000 * _o(n), 1),
            'FGGA_CH4_FLAG': ('data',  _z(n), 1)
        }

    def declare_outputs(self):
        """
        Declare the outputs that are going to be written by this module.
        """

        ccn_manufacturer = 'Droplet Measurement Technologies Inc.'
        ccn_model = 'CFTG Dual Column CCNC'
        ccn_serial = self.dataset.lazy['CCN_SN']

        self.declare(
            'ccn_column_a_conc',
            units='cm-3',
            frequency=1,
            long_name='CCN column A number concentration at STP',
            standard_name='number_concentration_of_cloud_condensation_nuclei_at_stp_in_air',
            instrument_manufacturer=ccn_manufacturer,
            instrument_model=ccn_model,
            instrument_serial_number=ccn_serial
        )

        self.declare(
            'ccn_column_b_conc',
            units='cm-3',
            frequency=1,
            long_name='CCN column B number concentration at STP',
            standard_name='number_concentration_of_cloud_condensation_nuclei_at_stp_in_air',
            instrument_manufacturer=ccn_manufacturer,
            instrument_model=ccn_model,
            instrument_serial_number=ccn_serial
        )

        self.declare(
            'ccn_column_a_rh',
            units='percent',
            frequency=1,
            long_name='CCN column A relative humidity',
            standard_name='relative_humidity',
            instrument_manufacturer=ccn_manufacturer,
            instrument_model=ccn_model,
            instrument_serial_number=ccn_serial
        )

        self.declare(
            'ccn_column_b_rh',
            units='percent',
            frequency=1,
            long_name='CCN column B relative humidity',
            standard_name='relative_humidity',
            instrument_manufacturer=ccn_manufacturer,
            instrument_model=ccn_model,
            instrument_serial_number=ccn_serial
        )

    @staticmethod
    def correct_press_amb(t, p, conc, t_opc, s_pres):
        return conc * ((8.314472 * (t + 273.15)) / p / (
            ((8.314472 * (t_opc + 273.15)) / s_pres)
        ))

    def correct_press_stp(self, conc, t_opc, s_pres):
        return self.correct_press_amb(0, 1013.25, conc, t_opc, s_pres)

    def flag_ss(self):
        BUF_LEN = 180
        self.d['SSA_FLAG'] = 0
        self.d['SSB_FLAG'] = 0
        SSchangeA = self.d['CCN_Current_SS_A'].diff()
        SSchangeB = self.d['CCN_Current_SS_B'].diff()
        groups_a = self.d.groupby(
            ((SSchangeA != 0) & (SSchangeA.notnull())).cumsum()
        )
        groups_b = self.d.groupby(
            ((SSchangeB != 0) & (SSchangeB.notnull())).cumsum()
        )

        for col, groups in zip(('SSA_FLAG', 'SSB_FLAG'), (groups_a, groups_b)):
            for group in groups:
                df = group[1]
                _start = df.index[0]
                _end = df.index[min(BUF_LEN, len(df.index)-1)]
                mask = (self.d.index >= _start) & (self.d.index <= _end)
                self.d.loc[mask, col] = 1

    def flag_t(self):
        BUF_LEN = 180
        self.d['TA_FLAG'] = 0
        self.d['TB_FLAG'] = 0
        SSchangeA = self.d['CCN_Current_SS_A'].diff()
        SSchangeB = self.d['CCN_Current_SS_B'].diff()
        TchangeA = self.d['CCN_T1_Set_A'].diff()
        TchangeB = self.d['CCN_T1_Set_B'].diff()
        groups_a = self.d.groupby(
            ((TchangeA != 0) & (SSchangeA.notnull())).cumsum()
        )
        groups_b = self.d.groupby(
            ((TchangeB != 0) & (SSchangeB.notnull())).cumsum()
        )

        for col, groups in zip(('TA_FLAG', 'TB_FLAG'), (groups_a, groups_b)):
            for group in groups:
                df = group[1]
                _start = df.index[0]
                _end = df.index[min(BUF_LEN, len(df.index)-1)]
                mask = (self.d.index >= _start) & (self.d.index <= _end)
                self.d.loc[mask, col] = 1

    def flag(self):
        _mask = lambda arr, lo, hi: ((arr <= lo) | (arr >= hi))

        d = self.d
        self.flag_ss()
        self.flag_t()

        a_col = 'ErrorBitsA'
        b_col = 'ErrorBitsB'
        d[a_col] = 0
        d[b_col] = 0
        d['CCNA_FLAG'] = 3
        d['CCNB_FLAG'] = 3

        d.loc[_mask(d.FloRatioA, 6, 13), a_col] += 1
        d.loc[_mask(d.CCN_Sample_Flow_A, 40, 60), a_col] += 10
        d.loc[(d.SSA_cor <= 0.065), a_col] += 100
        d.loc[_mask(d.CCN_Sample_Pressure_A - d.CCN_Sample_Pressure_A.shift(30), -2, 2),
         a_col] += 10**3
        d.loc[_mask(d.CCN_Sample_Pressure_A - d.CCN_Sample_Pressure_A.shift(1), -1, 1),
         a_col] += 10**4
        d.loc[_mask(d.PS_RVSM - d.CCN_Sample_Pressure_A, -9e99, 650)
          | (d.CCN_Sample_Pressure_A >= d.PS_RVSM), a_col] += 10**5
        d.loc[(d.SSA_FLAG == 1) | (d.TA_FLAG == 1), a_col] += 10*6
        d.loc[(d.CCN_T_OPC_A <= d.CCN_T3_Read_A) |
              (d.CCN_T3_Read_A <= d.CCN_T2_Read_A) |
              (d.CCN_T2_Read_A <= d.CCN_T1_Read_A) |
              (d.CCN_T1_Read_A <= d.CCN_T_Sample_A),
          a_col] += 10**7

        d.loc[_mask(d.FloRatioB, 6, 13), b_col] += 1
        d.loc[_mask(d.CCN_Sample_Flow_B, 40, 60), b_col] += 10
        d.loc[(d.SSB_cor <= 0.065), b_col] += 100
        d.loc[_mask(d.CCN_Sample_Pressure_B - d.CCN_Sample_Pressure_B.shift(30), -2, 2),
              b_col] += 10**3
        d.loc[_mask(d.CCN_Sample_Pressure_B - d.CCN_Sample_Pressure_B.shift(1), -1, 1),
              b_col] += 10**4
        d.loc[_mask(d.PS_RVSM - d.CCN_Sample_Pressure_B, -9e99, 650)
              | (d.CCN_Sample_Pressure_B >= d.PS_RVSM), b_col] += 10**5
        d.loc[(d.SSB_FLAG == 1) | (d.TB_FLAG == 1), b_col] += 10*6
        d.loc[(d.CCN_T_OPC_B <= d.CCN_T3_Read_B) |
              (d.CCN_T3_Read_B <= d.CCN_T2_Read_B) |
              (d.CCN_T2_Read_B <= d.CCN_T1_Read_B) |
              (d.CCN_T1_Read_B <= d.CCN_T_Sample_B), b_col] += 10**7

        code = [0] + [2**i for i in range(9)]
        meaning = ['no error', 'flow ratio error', 'sample flow error',
                   'ss min error', 'slow p leak', 'fast p leak',
                   'other p failure', 'temperature instability',
                   'temperature grabient instability']
        bitwise = [str(bin(i))[2:] for i in code]

        d.loc[(d.ErrorBitsA == 0), 'CCNA_FLAG'] = 0
        d.loc[(d.ErrorBitsA >= 1) & (d.ErrorBitsA <= 11), 'CCNA_FLAG'] = 1
        d.loc[(d.ErrorBitsA >= 12) & (d.ErrorBitsA <= 1111), 'CCNA_FLAG'] = 2
        d.loc[(d.ErrorBitsA >= 1112), 'CCNA_FLAG'] = 3

        d.loc[(d.ErrorBitsB == 0), 'CCNB_FLAG'] = 0
        d.loc[(d.ErrorBitsB >= 1) & (d.ErrorBitsB <= 11), 'CCNB_FLAG'] = 1
        d.loc[(d.ErrorBitsB >= 12) & (d.ErrorBitsB <= 1111), 'CCNB_FLAG'] = 2
        d.loc[(d.ErrorBitsB >= 1112), 'CCNB_FLAG'] = 3


    def process(self):
        """
        Processing entry point.
        """
        index = self.dataset['CCN_T_OPC_A'].index
        self.get_dataframe(method='onto', index=index)
        d = self.d

        ccna_cor = self.correct_press_amb(
            d.TAT_DI_R, d.PS_RVSM, d.CCN_Sample_Pressure_A,
            d.CCN_T_OPC_A, d.CCN_Sample_Pressure_A
        )

        ccnb_cor = self.correct_press_amb(
            d.TAT_DI_R, d.PS_RVSM, d.CCN_Sample_Pressure_B,
            d.CCN_T_OPC_B, d.CCN_Sample_Pressure_B
        )

        ccna_stp = self.correct_press_stp(
            d.CCN_Sample_Pressure_A, d.CCN_T_OPC_A, d.CCN_Sample_Pressure_A
        )

        ccnb_stp = self.correct_press_stp(
            d.CCN_Sample_Pressure_B, d.CCN_T_OPC_B, d.CCN_Sample_Pressure_B
        )

        # Correct SS for pressure based on work in Roberts 2010 - resulting in
        # error of +- 10% in SS used since June 2013
        # Work in Airborne CCN Measurements show specific instrument to be +=- 7% so
        Mp = 5.087e-5 * d.CCN_Sample_Pressure_A + 1.833e-2
        Yp = 1.429e-4 * d.CCN_Sample_Pressure_A + -0.1747

        SSA_cor = Mp * d.CCN_T3_Read_A - d.CCN_T1_Read_A + Yp
        SSB_cor = Mp * d.CCN_T3_Read_B - d.CCN_T1_Read_B + Yp
        d['SSA_cor'] = SSA_cor
        d['SSB_cor'] = SSB_cor

        # SS expressed as a relative humidity to match CF naming convention
        RHA_col = SSA_cor + 100
        RHB_col = SSB_cor + 100

        d['FloRatioA'] = d.CCN_Sheath_Flow_A / d.CCN_Sample_Flow_A
        d['FloRatioB'] = d.CCN_Sheath_Flow_B / d.CCN_Sample_Flow_B

        self.flag()

        ccnkwargs = {
            'flag': DecadesClassicFlag,
            'flag_postfix': 'qcflag'
        }

        ccn_a = DecadesVariable({'ccn_column_a_conc': ccna_stp}, **ccnkwargs)
        ccn_b = DecadesVariable({'ccn_column_b_conc': ccnb_stp}, **ccnkwargs)
        ccn_a_rh = DecadesVariable({'ccn_column_a_rh': RHA_col}, **ccnkwargs)
        ccn_b_rh = DecadesVariable({'ccn_column_b_rh': RHB_col}, **ccnkwargs)

        for var in (ccn_a, ccn_b, ccn_a_rh, ccn_b_rh):
            var.flag.add_meaning(
                0, 'data good', 'Data are considered good.'
            )
            var.flag.add_meaning(
                1, 'minor data issue',
                ('Minor flow issues, be aware error on RH will be larger than '
                 ' 10%')
            )
            var.flag.add_meaning(
                2, 'major data issue',
                ('Data should not be used routinely please contact FAAM '
                 'before using.')
            )
            var.flag.add_meaning(
                3, 'bad data','Do not use data under any circumstances'
            )

        self.add_output(ccn_a)
        self.add_output(ccn_b)
        self.add_output(ccn_a_rh)
        self.add_output(ccn_b_rh)

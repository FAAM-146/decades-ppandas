import numpy as np

from ..decades import DecadesVariable
from .base import PPBase


class TPress(PPBase):

    inputs = [
        'CALTP1',
        'CALTP2',
        'CALTP3',
        'CALTP4',
        'CALTP5',
        'CORCON_tp_p0_s10',
        'CORCON_tp_up_down',
        'CORCON_tp_left_right',
        'CORCON_tp_top_s10',
        'CORCON_tp_right_s10'
    ]

    def declare_outputs(self):

        self.declare(
            'P0_S10',
            units='hPa',
            frequency=32,
            number=773,
            long_name=('Calibrated differential pressure between centre (P0) '
                       'port and S10 static'),
        )

        self.declare(
            'PA_TURB',
            units='hPa',
            frequency=32,
            number=774,
            long_name=('Calibrated differential pressure between turbulence ',
                       'probe vertical ports')
        )

        self.declare(
            'PB_TURB',
            units='hPa',
            frequency=32,
            number=775,
            long_name=('Calibrated differential pressure between turbulence ',
                       'probe horizontal ports')
        )

        self.declare(
            'TBPC',
            units='hPa',
            frequency=32,
            number=776,
            long_name='TURB PROBE Ca'
        )

        self.declare(
            'TBPD',
            units='hPa',
            frequency=32,
            number=777,
            long_name='TURB PROBE Cb'
        )

    def get_range_flag(self, var, limits):
        flag = np.zeros_like(self.d.index)
        flag[self.d[var] < limits[0]] = 2
        flag[self.d[var] > limits[1]] = 2
        flag[self.d[var] == 0] = 3
        print(flag)
        return flag

    def process(self):
        self.get_dataframe()
        d = self.d

        d['P0_S10'] = np.polyval(
            self.dataset['CALTP1'][::-1], d.CORCON_tp_p0_s10
        )

        d['PA_TURB'] = np.polyval(
            self.dataset['CALTP2'][::-1], d.CORCON_tp_up_down
        )

        d['PB_TURB'] = np.polyval(
            self.dataset['CALTP3'][::-1], d.CORCON_tp_left_right
        )

        d['TBPC'] = np.polyval(
            self.dataset['CALTP4'][::-1], d.CORCON_tp_top_s10
        )

        d['TBPD'] = np.polyval(
            self.dataset['CALTP5'][::-1], d.CORCON_tp_right_s10
        )

        p0_s10_flag = self.get_range_flag('P0_S10', (30, 180))
        pa_turb_flag = self.get_range_flag('PA_TURB', (-30, 30))
        pb_turb_flag = self.get_range_flag('PB_TURB', (-20, 20))
        tbpc_flag = self.get_range_flag('TBPC', (50, 200))
        tbpd_flag = self.get_range_flag('TBPD', (50, 200))

        p0_s10_out = DecadesVariable(d.P0_S10)
#        p0_s10_out.add_flag(p0_s10_flag)
        self.add_output(p0_s10_out)

        pa_turb_out = DecadesVariable(d.PA_TURB)
#        pa_turb_out.add_flag(pa_turb_flag)
        self.add_output(pa_turb_out)

        pb_turb_out = DecadesVariable(d.PB_TURB)
#        pb_turb_out.add_flag(pb_turb_flag)
        self.add_output(pb_turb_out)

        tbpc_out = DecadesVariable(d.TBPC)
#        tbpc_out.add_flag(tbpc_flag)
        self.add_output(tbpc_out)

        tbpd_out = DecadesVariable(d.TBPD)
#        tbpd_out.add_flag(tbpd_flag)
        self.add_output(tbpd_out)

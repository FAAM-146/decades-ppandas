import numpy as np

from ..decades import DecadesVariable
from ..decades import flags
from .base import PPBase
from .shortcuts import _o, _z


class TPress(PPBase):

    inputs = [
        'CALTP1',
        'CALTP2',
        'CALTP3',
        'CORCON_tp_p0_s10',
        'CORCON_tp_up_down',
        'CORCON_tp_left_right',
    ]

    @staticmethod
    def test():
        return {
            'CALTP1': ('const', [.88, 5.5e-3, 2e-9, -3.6e-14]),
            'CALTP2': ('const', [-.14, 2.1e-5, 2e-15, -5.64e-23]),
            'CALTP3': ('const', [-.73, 2.1e-5, 1.5e-15, 5.57e-22]),
            'CALTP4': ('const', [1.5, 1.0e-2]),
            'CALTP5': ('const', [-1.2, 1.0e-2]),
            'CORCON_tp_p0_s10': ('data', 15e3 * _o(100)),
            'CORCON_tp_up_down': ('data', 15e4 * _o(100)),
            'CORCON_tp_left_right': ('data', _z(100)),
        }

    def declare_outputs(self):

        self.declare(
            'P0_S10',
            units='hPa',
            frequency=32,
            long_name=('Calibrated differential pressure between centre (P0) '
                       'port and S10 static'),
        )

        self.declare(
            'PA_TURB',
            units='hPa',
            frequency=32,
            long_name=('Calibrated differential pressure between turbulence '
                       'probe vertical ports')
        )

        self.declare(
            'PB_TURB',
            units='hPa',
            frequency=32,
            long_name=('Calibrated differential pressure between turbulence '
                       'probe horizontal ports')
        )

    def get_range_flag(self, var, limits):
        flag = np.zeros_like(self.d[var])
        flag[self.d[var] < limits[0]] = 1
        flag[self.d[var] > limits[1]] = 1
        flag[self.d[var] == 0] = 1
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

        p0_s10_flag = self.get_range_flag('P0_S10', (30, 180))
        pa_turb_flag = self.get_range_flag('PA_TURB', (-30, 30))
        pb_turb_flag = self.get_range_flag('PB_TURB', (-20, 20))

        p0_s10_out = DecadesVariable(d.P0_S10)
        p0_s10_out.flag.add_meaning(0, flags.DATA_GOOD)
        p0_s10_out.flag.add_meaning(1, flags.OUT_RANGE)
        p0_s10_out.flag.add_flag(p0_s10_flag)
        self.add_output(p0_s10_out)

        pa_turb_out = DecadesVariable(d.PA_TURB)
        pa_turb_out.flag.add_meaning(0, flags.DATA_GOOD)
        pa_turb_out.flag.add_meaning(1, flags.OUT_RANGE)
        pa_turb_out.flag.add_flag(pa_turb_flag)
        self.add_output(pa_turb_out)

        pb_turb_out = DecadesVariable(d.PB_TURB)
        pb_turb_out.flag.add_meaning(0, flags.DATA_GOOD)
        pb_turb_out.flag.add_meaning(1, flags.OUT_RANGE)
        pb_turb_out.flag.add_flag(pb_turb_flag)
        self.add_output(pb_turb_out)

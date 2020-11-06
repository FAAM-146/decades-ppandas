import numpy as np

from .base import QAMod, QAFigure

class TurbProbe(QAMod):
    inputs = [
        'PSP_TURB',
        'P0_S10',
        'Q_RVSM',
        'PA_TURB',
        'PB_TURB',
        'AOA', 'AOSS',
        'U_C', 'V_C', 'W_C'
    ]

    def make_psp_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .75, .8, .15], labelx=False)
        psp = fig.filter_in_flight(self.dataset['PSP_TURB'].data)
        p0s10 = fig.filter_in_flight(self.dataset['P0_S10'].data)
        q = fig.filter_in_flight(self.dataset['Q_RVSM'].data)

        ax.plot(psp, label='PSP Turb')
        ax.plot(p0s10, label='P0-S10')
        ax.plot(q, label='Q (RVSM)')
        ax.set_ylabel('Pressure (hPa)')
        ax.legend(fontsize=6, loc='lower right')

    def make_papb_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .58, .8, .15], labelx=False)
        pa = fig.filter_in_flight(self.dataset['PA_TURB'].data)
        pb = fig.filter_in_flight(self.dataset['PB_TURB'].data)

        ax.plot(pa, label='PA', alpha=.6)
        ax.plot(pb, label='PB', alpha=.6)
        ax.set_ylabel('Pressure (hPa)')
        ax.legend(fontsize=6, loc='lower right')

    def make_aox_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .41, .8, .15], labelx=False)

        aoa = fig.filter_in_flight(self.dataset['AOA'].data)
        aoss = fig.filter_in_flight(self.dataset['AOSS'].data)
        ax.plot(aoa, label='AOA', alpha=.6)
        ax.plot(aoss, label='AOSS', alpha=.6)
        ax.legend(fontsize=6, loc='lower right')
        ax.set_ylabel('Flow Angle ($^\circ$)')

    def make_w_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .24, .8, .15], labelx=False)

        w = fig.filter_in_flight(self.dataset['W_C'].data)
        ax.plot(w, label='W_C')
        ax.add_zero_line()
        ax.legend(fontsize=6, loc='lower right')
        ax.set_ylabel('Velocity (m/s)')

    def make_uv_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .07, .8, .15])

        u = fig.filter_in_flight(self.dataset['U_C'].data)
        v = fig.filter_in_flight(self.dataset['V_C'].data)
        ax.plot(u, label='U_C', alpha=.6)
        ax.plot(v, label='V_C', alpha=.6)
        ax.legend(fontsize=6, loc='lower right')
        ax.set_ylabel('Velocity (m/s)')


    def run(self):
        with QAFigure(self.dataset, 'Turbulence', landscape=True) as fig:
            self.make_psp_timeseries(fig)
            self.make_papb_timeseries(fig)
            self.make_aox_timeseries(fig)
            self.make_w_timeseries(fig)
            self.make_uv_timeseries(fig)

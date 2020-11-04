import numpy as np

from .base import QAMod, QAFigure

class TecoO3(QAMod):
    inputs = [
        'PALT_RVS',
        'O3_TECO'
    ]

    def make_palt_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .7, .8, .2], labelx=False)
        ts = self.dataset['PALT_RVS'].data
        ax.plot(ts, label='RVSM Alt.')
        ax.set_ylabel('Altitude (m)')
        ax.legend(fontsize=6)

    def make_o3_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .1, .8, .58])
        ts = fig.filter_in_flight(self.dataset['O3_TECO'].data)
        ax.plot(ts, label='TECO Ozone')
        ax.set_ylabel('O3 conc.')
        ax.legend(fontsize=6)


    def run(self):
        with QAFigure(self.dataset, 'TECO O3', landscape=True) as fig:
            self.make_palt_timeseries(fig)
            self.make_o3_timeseries(fig)

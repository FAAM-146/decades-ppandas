import numpy as np

from .base import QAMod, QAFigure

class PSAP(QAMod):
    inputs = [
        'PSAP_TRA',
        'PSAP_LOG',
        'PSAP_LIN',
        'PSAP_FLO'
    ]

    def make_tra_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .72, .8, .2], labelx=False)

        ts = fig.filter_in_flight(
            self.dataset['PSAP_TRA'].data
        )
        ax.plot(ts, label='Transmittance')
        ax.set_ylabel('Trans.')
        ax.legend(fontsize=6)

    def make_loglin_timeseries(self, fig):
        ax1, ax2 = fig.timeseries_axes([.1, .5, .8, .2], labelx=False,
                                       twinx=True)

        log = fig.filter_in_flight(self.dataset['PSAP_LOG'].data)
        lin = fig.filter_in_flight(self.dataset['PSAP_LIN'].data)

        ax1.plot(lin*1e5, label='PSAP Lin')
        ax2.plot(log*1e5, color='tab:orange', label='PSAP Log')
        ax1.legend(loc='upper left', fontsize=6)
        ax2.legend(loc='upper right', fontsize=6)
        ax1.set_ylabel('PSAP lin * 1e5')
        ax2.set_ylabel('PSAP log * 1e5')

    def make_flo_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .28, .8, .2], labelx=False)

        ts = fig.filter_in_flight(
            self.dataset['PSAP_FLO'].data
        )
        ax.plot(ts, label='PSAP Flow')
        ax.set_ylabel('Flow')
        ax.legend(fontsize=6)

    def make_flag_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .06, .8, .2])

        ts = fig.filter_in_flight(
            self.dataset['PSAP_FLO'].flag()
        )
        ax.plot(ts, 'tab:red', linewidth=3, label='PSAP Flag')
        ax.set_ylabel('Flag')
        ax.legend(fontsize=6)

    def run(self):
        with QAFigure(self.dataset, 'PSAP', landscape=False) as fig:
            self.make_tra_timeseries(fig)
            self.make_loglin_timeseries(fig)
            self.make_flo_timeseries(fig)
            self.make_flag_timeseries(fig)

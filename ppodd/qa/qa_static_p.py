import numpy as np
import matplotlib.pyplot as plt

from .base import QAMod, QAFigure
from ..utils.calcs import sp_mach

class StaticPressure(QAMod):
    inputs = [
        'PS_RVSM',
        'Q_RVSM'
    ]

    def make_press_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .7, .8, .2])
        rvsm = fig.filter_in_flight(self.dataset['PS_RVSM'].data)
        ax.plot(rvsm, label='RVSM Press.')

        try:
            p9 = fig.filter_in_flight(self.dataset['P9_STAT'])
            ax.plot(p9, label='P9 Press.')
        except KeyError:
            pass

        try:
            p10 = fig.filter_in_flight(self.dataset['P10_STAT'])
            ax.plot(p10, label='P10 Press.')
        except KeyError:
            pass

        ax.set_ylabel('Pressure (hPa)')
        ax.legend(fontsize=6)

    def make_histograms(self, fig):
        def _hist(ax, param):
            p = fig.filter_in_flight(self.dataset[param].data)
            m = (np.isfinite(p)) & (np.isfinite(rvsm))
            ax.hist(rvsm[m]-p[m], bins=np.arange(-4,4.2,.2))
            ax.set_title(f'PS_RVSM - {param}')

        ax1 = fig.axes([.1, .4, .35, .25])
        ax2 = fig.axes([.55, .4, .35, .25])

        p9 = None
        p10 = None
        rvsm = fig.filter_in_flight(self.dataset['PS_RVSM'].data)

        try:
            _hist(ax1, 'P9_STAT')
        except KeyError:
            self._blank_axis(ax1, 'P9_STAT')

        try:
            _hist(ax2, 'P10_STAT')
        except KeyError:
            self._blank_axis(ax2, 'P9_STAT')

    def _blank_axis(self, ax, param):
        ax.text(.25, .5, f'{param} not available')
        ax.set_xticks([])
        ax.set_yticks([])

    def make_scatters(self, fig):
        ax1 = fig.axes([.1, .1, .35, .25])
        ax2 = fig.axes([.55, .1, .35, .25])

        def _scatter(ax, param):
            p = fig.filter_in_flight(self.dataset[param].data.asfreq('1s'))
            m = (np.isfinite(p)) & (np.isfinite(rvsm))
            ax.scatter(rvsm[m], p[m], 5, c=mach.reindex(p.index)[m])
            ax.add_121(linewidth=.5)
            fit = np.polyfit(rvsm[m], p[m], 1)
            ax.set_title(f'PS_RVSM v {param} (m={fit[0]:0.3f}, c={fit[1]:0.2f})')
            ax.set_xlabel('PS_RVSM (hPa)')
            ax.set_ylabel(f'{param} (hPa)')

        p9 = None
        p10 = None
        rvsm = fig.filter_in_flight(self.dataset['PS_RVSM'].data.asfreq('1s'))
        q = fig.filter_in_flight(self.dataset['Q_RVSM'].data.asfreq('1s'))
        mach = sp_mach(q, rvsm)

        try:
            _scatter(ax1, 'P9_STAT')
        except KeyError:
            self._blank_axis(ax1, 'P9_STAT')

        try:
            _scatter(ax2, 'P10_STAT')
        except KeyError:
            self._blank_axis(ax2, 'P10_STAT')

    def run(self):
        with QAFigure(self.dataset, 'Static Pressure') as fig:
            self.make_press_timeseries(fig)
            self.make_histograms(fig)
            self.make_scatters(fig)

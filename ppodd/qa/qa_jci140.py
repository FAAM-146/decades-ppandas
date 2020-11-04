import numpy as np

from .base import QAMod, QAFigure

class Jci140QA(QAMod):
    inputs = [
        'PRTAFT_jci140_signal',
    ]

    def make_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .1, .8, .8])

        ts = fig.filter_in_flight(
            self.dataset['PRTAFT_jci140_signal'].data
        )
        ax.plot(ts, label='EXX JCI')

    def run(self):
        with QAFigure(self.dataset, 'EXX JCI', landscape=True) as fig:
            self.make_timeseries(fig)

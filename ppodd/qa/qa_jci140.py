import numpy as np

from .base import QAMod, QAFigure


class Jci140QA(QAMod):
    inputs = [
        "PRTAFT_jci140_signal",
    ]

    def make_timeseries(self, fig: QAFigure) -> None:
        ax = fig.timeseries_axes([0.1, 0.1, 0.8, 0.8])

        ts = fig.filter_in_flight(self.dataset["PRTAFT_jci140_signal"].data)
        ax.plot(ts, label="EXX JCI")

    def run(self) -> None:
        with QAFigure(self.dataset, "EXX JCI", landscape=True) as fig:
            self.make_timeseries(fig)

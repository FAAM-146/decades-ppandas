import numpy as np

from .base import QAMod, QAFigure


class Cabin(QAMod):
    inputs = ["CAB_TEMP", "CAB_PRES", "PS_RVSM"]

    def make_temp_timeseries(self, fig: QAFigure) -> None:
        ax = fig.timeseries_axes([0.1, 0.55, 0.8, 0.35], labelx=False)

        ts = fig.filter_in_flight(self.dataset["CAB_TEMP"].data)
        ax.plot(ts, label="Cabin Temperature")
        ax.set_ylabel(r"Temperature ($^\circ$C)")
        ax.legend(fontsize=6)

    def make_pressure_timeseries(self, fig: QAFigure) -> None:
        ax1, ax2 = fig.timeseries_axes([0.1, 0.15, 0.8, 0.35], twinx=True)
        cab_ts = fig.filter_in_flight(self.dataset["CAB_PRES"].data)
        ps_ts = fig.filter_in_flight(self.dataset["PS_RVSM"].data)
        ax1.plot(cab_ts, label="Cabin pressure")
        ax2.plot(ps_ts, "tab:orange", label="Ambient pressure")
        ax1.set_ylabel("Cabin pressure (hPa)")
        ax2.set_ylabel("Ambient pressure (hPa)")
        ax1.legend(loc="lower left", fontsize=6)
        ax2.legend(loc="lower right", fontsize=6)

    def run(self) -> None:
        with QAFigure(self.dataset, "Cabin", landscape=True) as fig:
            self.make_temp_timeseries(fig)
            self.make_pressure_timeseries(fig)

import numpy as np

from .base import QAMod, QAFigure


class NephQA(QAMod):

    inputs = [
        "TSC_BLUU",
        "TSC_GRNU",
        "TSC_REDU",
        "BSC_BLUU",
        "BSC_GRNU",
        "BSC_REDU",
        "NEPH_PR",
        "NEPH_T",
        "NEPH_RH",
        "PS_RVSM",
    ]

    def make_backscatter_timeseries(self, fig: QAFigure) -> None:
        _ax = fig.timeseries_axes([0.1, 0.7, 0.8, 0.2], labelx=False)

        _ax.plot(self.dataset["BSC_REDU"](), color="tab:red", alpha=0.5)
        _ax.plot(self.dataset["BSC_GRNU"](), color="tab:green", alpha=0.5)
        _ax.plot(self.dataset["BSC_BLUU"](), color="tab:blue", alpha=0.5)

        r = fig.filter_in_flight(self.dataset["BSC_REDU"]())
        g = fig.filter_in_flight(self.dataset["BSC_GRNU"]())
        b = fig.filter_in_flight(self.dataset["BSC_BLUU"]())

        _ax.set_ylim(
            np.min([r.min(), g.min(), b.min()]),
            np.max([r.max(), g.max(), b.max()]),
        )

        _ax.set_ylabel("Backscatter (m$^{-1}$)")

    def make_total_timeseries(self, fig: QAFigure) -> None:
        _ax = fig.timeseries_axes([0.1, 0.5, 0.8, 0.2], labelx=False)

        _ax.plot(self.dataset["TSC_REDU"](), color="tab:red", alpha=0.5)
        _ax.plot(self.dataset["TSC_GRNU"](), color="tab:green", alpha=0.5)
        _ax.plot(self.dataset["TSC_BLUU"](), color="tab:blue", alpha=0.5)

        r = fig.filter_in_flight(self.dataset["TSC_REDU"]())
        g = fig.filter_in_flight(self.dataset["TSC_GRNU"]())
        b = fig.filter_in_flight(self.dataset["TSC_BLUU"]())

        _ax.set_ylim(
            np.min([r.min(), g.min(), b.min()]),
            np.max([r.max(), g.max(), b.max()]),
        )

        _ax.set_ylabel("Total scatter (m$^{-1}$)")

    def make_temp_timeseries(self, fig: QAFigure) -> None:
        _ax, _ax2 = fig.timeseries_axes([0.1, 0.3, 0.8, 0.2], labelx=False, twinx=True)

        _ax.plot(self.dataset["NEPH_T"](), color="tab:red")

        t = fig.filter_in_flight(self.dataset["NEPH_T"]())
        _ax.set_ylim([t.min() - 1, t.max() + 1])

        _ax2.plot(self.dataset["NEPH_RH"]())

        _ax.set_ylabel("Temperature (K)", color="tab:red")
        _ax2.set_ylabel("RH (%)", color="tab:blue")

    def make_press_timeseries(self, fig: QAFigure) -> None:
        _ax = fig.timeseries_axes([0.1, 0.1, 0.8, 0.2])
        _ax.plot(self.dataset["NEPH_PR"](), label="Neph")
        _ax.plot(self.dataset["PS_RVSM"](), label="PS_RVSM", color="tab:red")

        t = fig.filter_in_flight(self.dataset["PS_RVSM"]())
        _ax.set_ylim([t.min() - 10, t.max() + 10])

        _ax.set_ylabel("Pressure (hPa)")
        _ax.legend(fontsize=6, loc="lower left")

    def run(self) -> None:
        with QAFigure(self.dataset, "TSI 3563 Nephelometer", landscape=True) as fig:
            self.make_backscatter_timeseries(fig)
            self.make_total_timeseries(fig)
            self.make_temp_timeseries(fig)
            self.make_press_timeseries(fig)

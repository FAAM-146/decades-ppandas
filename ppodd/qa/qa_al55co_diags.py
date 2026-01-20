import numpy as np

from .base import QAMod, QAFigure


class AL55CODiagsQA(QAMod):
    """
    Produces a QC plot for the AL55CO instrument during a flight. This gives an
    idea of the performance of the instrument.
    """

    inputs = ["AL55CO_counts", "CO_AERO", "WOW_IND"]

    def make_altitude_plot(self, fig: QAFigure) -> None:
        ax = fig.timeseries_axes([0.1, 0.75, 0.8, 0.15], labelx=False)

    def make_conc_plot(self, fig: QAFigure) -> None:
        ax = fig.timeseries_axes([0.1, 0.6, 0.8, 0.14], labelx=False)

        co = fig.filter_in_flight(self.dataset["CO_AERO"]())
        co_unc = fig.filter_in_flight(self.dataset["CO_AERO_CU"]())
        cal_mask = fig.filter_in_flight(self.dataset["CO_AERO"].flag())
        co_unc.loc[cal_mask > 0] = np.nan
        ax.fill_between(
            co.index,
            co - co_unc,
            co + co_unc,
            color="gray",
            alpha=0.5,
            label="1σ uncertainty",
        )
        co.loc[cal_mask > 0] = np.nan
        ax.plot(co, label="CO_AERO", linewidth=0.5)
        ax.legend(loc="upper left", fontsize=6)
        ax.set_ylabel("CO conc. (ppb)")

    def make_flow_plot(self, fig: QAFigure) -> None:
        ax, ax2 = fig.timeseries_axes([0.1, 0.45, 0.8, 0.14], labelx=False, twinx=True)
        ax.plot(self.dataset["AL55CO_flowlamp"](), color="tab:green", label="Flow Lamp")
        ax.plot(
            self.dataset["AL55CO_flowmono"](), color="tab:orange", label="Flow Mono"
        )
        ax2.plot(
            self.dataset["AL55CO_MFC1_mass_flow"](), color="tab:blue", label="MFC1 Flow"
        )
        ax.legend(loc="upper left", fontsize=6)
        ax2.legend(loc="upper right", fontsize=6)
        ax.set_ylabel("Flow Lamp/Mono (sccm)")
        ax2.set_ylabel("MFC1 Flow (slpm)")

        ax.set_ylim(30, 40)

    def make_vaccum_plot(self, fig: QAFigure) -> None:
        ax = fig.timeseries_axes([0.1, 0.3, 0.8, 0.14], labelx=False)
        ax.plot(self.dataset["AL55CO_pcell"](), color="tab:purple", label="Pcell")
        ax.set_ylabel("Cell Pressure (Torr)")
        ax.legend(loc="upper left", fontsize=6)
        ax.set_ylim(7, 8)

    def make_temperature_plot(self, fig: QAFigure) -> None:
        ax = fig.timeseries_axes([0.1, 0.15, 0.8, 0.14], labelx=True)
        ax.plot(self.dataset["AL55CO_templamp"](), color="tab:blue", label="Temp Lamp")
        ax.plot(self.dataset["AL55CO_tempcell"](), color="tab:green", label="Temp Cell")
        ax.set_ylabel("Temperature (°C)")
        ax.legend(loc="upper left", fontsize=6)
        ax.set_ylim(26, 38)

    def run(self) -> None:
        with QAFigure(self.dataset, "AL55CO Diagnostics") as fig:
            self.make_altitude_plot(fig)
            self.make_conc_plot(fig)
            self.make_flow_plot(fig)
            self.make_vaccum_plot(fig)
            self.make_temperature_plot(fig)

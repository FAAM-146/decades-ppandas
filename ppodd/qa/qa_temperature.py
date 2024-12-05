import numpy as np

from .base import QAMod, QAFigure
from matplotlib.colors import ListedColormap
from ppodd.utils.calcs import sp_mach


class TemperatureQA(QAMod):
    inputs = [
        "TAT_DI_R",
        "TAT_ND_R",
        "IAT_DI_R",
        "IAT_ND_R",
        "WOW_IND",
        "PRTAFT_deiced_temp_flag",
    ]

    def make_cloud_plot(self, fig: QAFigure) -> None:

        lwc_axis = fig.timeseries_axes([0.1, 0.80, 0.8, 0.05], labelx=False)

        try:
            clear_air = self.dataset["NV_CLEAR_AIR_MASK"].data.asfreq("1s")
        except KeyError:
            return
        cloud = 1 - clear_air

        wow = self.dataset["WOW_IND"].data.asfreq("1s")
        wow = wow.reindex(cloud.index).bfill().ffill()

        cloud.loc[wow == 1] = np.nan

        _x = np.abs(np.vstack((cloud, cloud)))

        cmap = ListedColormap(
            np.array(
                [[214 / 256, 243 / 256, 255 / 256], [180 / 256, 180 / 256, 180 / 256]]
            )
        )

        lwc_axis.pcolormesh(cloud.index, [0, 1], _x, cmap=cmap)

        lwc_axis.set_ylabel("Cloud", rotation=0, labelpad=20)
        lwc_axis.set_xticks([])
        lwc_axis.set_yticks([])

    def make_heater_plot(self, fig: QAFigure) -> None:
        heater_axis = fig.timeseries_axes([0.1, 0.85, 0.8, 0.02], labelx=False)
        heater = self.dataset["PRTAFT_deiced_temp_flag"].data.asfreq("1s")
        heater.loc[heater == 0] = np.nan
        _x = np.vstack([heater, heater])

        heater_axis.pcolormesh(heater.index, [0, 1], _x, cmap="RdYlBu")

        heater_axis.set_xticks([])
        heater_axis.set_yticks([])
        heater_axis.set_ylabel("DI heater", rotation=0, labelpad=20)

    def make_temperature_plot(self, fig: QAFigure) -> None:
        temp_axis, temp2_axis = fig.timeseries_axes([0.1, 0.55, 0.8, 0.25], twinx=True)
        temp_axis.patch.set_alpha(0.0)

        tat_di = self.dataset["TAT_DI_R"].data.asfreq("1s")
        tat_nd = self.dataset["TAT_ND_R"].data.asfreq("1s")

        temp2_axis.plot(
            tat_di - tat_nd, color="k", linewidth=0.5, alpha=0.3, label="DI - ND"
        )

        tmin = fig.filter_in_flight(tat_di).dropna().min()
        tmax = fig.filter_in_flight(tat_di).dropna().max()

        try:
            u_tat_di = self.dataset["TAT_DI_R_CU"].data.asfreq("1s")
            temp_axis.fill_between(
                u_tat_di.index,
                tat_di - u_tat_di,
                tat_di + u_tat_di,
                alpha=0.5,
                color="tab:orange",
            )
        except Exception as e:
            print(str(e))

        try:
            u_tat_nd = self.dataset["TAT_ND_R_CU"].data.asfreq("1s")
            temp_axis.fill_between(
                u_tat_nd.index,
                tat_nd - u_tat_nd,
                tat_nd + u_tat_nd,
                alpha=0.5,
                color="tab:blue",
            )
        except Exception as e:
            print(str(e))

        temp_axis.plot(tat_nd, label="ND", color="tab:blue")
        temp_axis.plot(tat_di, label="DI", color="tab:orange")

        temp2_axis.add_zero_line()
        try:
            temp_axis.set_ylim([tmin, tmax])
        except Exception as e:
            pass
        temp_axis.set_xticklabels([])
        temp_axis.set_ylabel("Temp. (K)")
        temp2_axis.set_ylabel(r"$\Delta$ Temp. (K)")
        temp_axis.legend(fontsize=6)
        temp2_axis.legend(fontsize=6)
        temp2_axis.set_ylim([-2, 2])

    def make_mach_alt_plot(self, fig: QAFigure) -> None:
        sp = self.dataset["PS_RVSM"].data.asfreq("1s")
        psp = self.dataset["Q_RVSM"].data.asfreq("1s")

        ma_axis, pa_axis = fig.timeseries_axes([0.1, 0.37, 0.8, 0.17], twinx=True)

        ma_axis.plot(sp_mach(psp, sp), label="Mach", color="purple")
        ma_axis.legend(fontsize=6, loc="upper right")
        ma_axis.set_ylim([0, 0.65])

        pa_axis.plot(
            self.dataset["PALT_RVS"].data.asfreq("1s"),
            color="green",
            label="Press. Alt.",
        )
        pa_axis.legend(fontsize=6, loc="upper left")

        ma_axis.set_xlabel("Time (UTC)")
        ma_axis.set_ylabel("Mach #")
        pa_axis.set_ylabel("Pressure Alt (m)")

    def make_scatter_plot(self, fig: QAFigure) -> None:
        scat_axis = fig.axes([0.1, 0.12, 0.38, 0.2])

        tat_di = self.dataset["TAT_DI_R"].data.asfreq("1s")
        tat_nd = self.dataset["TAT_ND_R"].data.asfreq("1s")
        wow = self.dataset["WOW_IND"].data.asfreq("1s")

        tat_di.loc[wow == 1] = np.nan
        tat_nd.loc[wow == 1] = np.nan

        scat_axis.scatter(tat_di, tat_nd, 1, color="#03dbfc")

        scat_axis.add_121()
        scat_axis.set_xlabel("TAT DI (K)")
        scat_axis.set_ylabel("TAT ND (K)")

    def make_spectra_plot(self, fig: QAFigure) -> None:
        def spectra() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            _index = self.dataset["IAT_DI_R"].data.index
            _mask = (_index > fig.to_time) & (_index < fig.land_time)

            tat_di = self.dataset["IAT_DI_R"].data.loc[_mask].dropna()
            tat_nd = self.dataset["IAT_ND_R"].data.loc[_mask].dropna()

            index = tat_di.index.intersection(tat_nd.index)
            tat_di = tat_di.reindex(index)
            tat_nd = tat_nd.reindex(index)

            freqs = np.fft.fftfreq(tat_di.size, 1 / 32)

            ps_nd = np.abs(np.fft.fft(tat_nd)) ** 2
            ps_di = np.abs(np.fft.fft(tat_di)) ** 2
            idx = np.argsort(freqs)

            return freqs[idx], ps_nd[idx], ps_di[idx]

        def running_mean(x: np.ndarray, N: int) -> np.ndarray:
            return np.convolve(x, np.ones((N,)) / N)[(N - 1) :]

        freqs, ps_nd, ps_di = spectra()

        spec_axis = fig.axes([0.55, 0.12, 0.38, 0.2])
        spec_axis.loglog(
            freqs[freqs < 15.5], running_mean(ps_nd, 200)[freqs < 15.5], label="ND"
        )

        spec_axis.loglog(
            freqs[freqs < 15.5], running_mean(ps_di, 200)[freqs < 15.5], label="DI"
        )

        spec_axis.set_ylim(0.5, 10**4)
        spec_axis.set_xlim(1, 16)

        spec_axis.set_xlabel("Frequency")
        spec_axis.set_ylabel("Power")
        spec_axis.legend(fontsize=6)

    def make_info_text(self, fig: QAFigure) -> None:

        GOOD_THRESH = 0.5
        GOOD_FRAC = 0.95

        _index = self.dataset["TAT_DI_R"].data.index
        _mask = (_index > fig.to_time) & (_index < fig.land_time)

        tat_di = self.dataset["TAT_DI_R"].data.loc[_mask]
        tat_nd = self.dataset["TAT_ND_R"].data.loc[_mask]

        num_good = sum(np.abs(tat_di - tat_nd) <= GOOD_THRESH)
        num_bad = sum(np.abs(tat_di - tat_nd) > GOOD_THRESH)

        good = num_good / (num_good + num_bad) >= GOOD_FRAC

        if good:
            _txt = r"|$\Delta TAT$| < 0.5 K for at least 95% of flight"
            _col = "green"
        else:
            _txt = r"|$\Delta TAT$| > 0.5 K for at least 5% of flight"
            _col = "red"

        fig.text(
            0.5, 0.92, _txt, horizontalalignment="center", color=_col, size="small"
        )

        di_higher = (tat_di - tat_nd).mean() > 0
        if di_higher:
            _txt = "On average, TAT_DI reads higher"
        else:
            _txt = "On average, TAT_ND reads higher"
        fig.text(0.5, 0.90, _txt, horizontalalignment="center", size="small")

        _txt = "DI: {0} ({1}), NDI: {2} ({3})".format(
            *(self.dataset["DITSENS"] + self.dataset["NDTSENS"])
        )

        fig.text(0.5, 0.88, _txt, horizontalalignment="center", size="small")

    def run(self) -> None:
        with QAFigure(self.dataset, "Temperature Probes") as fig:
            self.make_heater_plot(fig)
            self.make_cloud_plot(fig)
            self.make_temperature_plot(fig)
            self.make_mach_alt_plot(fig)
            self.make_scatter_plot(fig)
            self.make_spectra_plot(fig)
            self.make_info_text(fig)

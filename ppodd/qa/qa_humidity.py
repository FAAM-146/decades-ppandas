import numpy as np

from .base import QAMod, QAFigure


class HumidityQA(QAMod):
    inputs = [
        "TDEW_GE",
        "TDEWCR2C",
        "VMR_CR2",
        "TAT_DI_R",
    ]

    def make_cloud_plot(self, fig: QAFigure) -> None:
        """
        Create a plot showing when we're in or out of cloud.
        """

        from matplotlib.colors import ListedColormap

        lwc_axis = fig.timeseries_axes([0.1, 0.83, 0.8, 0.05], labelx=False)

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

    def make_tdew_timeseries(self, fig: QAFigure) -> None:
        """
        Make a timeseries of dewpoint temperature for the CR2, GE and wvss2.
        Note the the wvss2 only proveds VMR, so we need to calculate a
        dewpoint. This is done as per the realtime calcs, using the equations
        from the CR2 manual.
        """

        def vp2dp(vp: np.ndarray, p: np.ndarray, temp: np.ndarray) -> np.ndarray:
            """
            Convert a vapour pressure, pressure & temp to a dew point.
            Equations given in the Buck CR2 manual.
            """
            a, b, c, d = (6.1121, 18.678, 257.14, 234.5)
            ef = 1 + 10**-4 * (
                2.2 + p / 10.0 * (0.0383 + 6.4 * 10**-5 * (temp - 273.15) * 2)
            )
            s = np.log(vp / ef) - np.log(a)
            result = d / 2 * (b - s - ((b - s) ** 2 - 4 * c * s / d) ** 0.5)
            return result + 273.15

        def vp2fp(vp: np.ndarray, p: np.ndarray, temp: np.ndarray) -> np.ndarray:
            """
            Convert vapour pressure, pressure and temp to a frost point.
            Equations given in the Buck CR2 manual.
            """
            a, b, c, d = (6.1115, 23.036, 279.82, 333.7)
            ef = 1 + 10**-4 * (
                2.2 + p / 10 * (0.0383 + 6.4 * 10**-5 * (temp - 273.15) * 2)
            )
            s = np.log(vp / ef) - np.log(a)
            result = d / 2 * (b - s - ((b - s) ** 2 - 4 * c * s / d) ** 0.5)
            return result + 273.15

        # Calculate a dew point from VMR for the WVSS2
        try:
            _wvss2_p = self.dataset[f"WVSS2F_PRESS_U"].data
            _wvss2_vmr = self.dataset[f"WVSS2F_VMR_C"].data / 1.6077
            wvss2_available = True
        except KeyError:
            wvss2_available = False

        if wvss2_available:
            _temp = self.dataset["TAT_DI_R"].data.reindex(_wvss2_vmr.index)
        wow = self.dataset["WOW_IND"].data

        if wvss2_available:
            _vp = _wvss2_vmr * _wvss2_p / (622 * 10**3 + _wvss2_vmr)

            # Dew point and frost point
            _dp = vp2dp(_vp, _wvss2_p, _temp)
            _fp = vp2fp(_vp, _wvss2_p, _temp)

            # Use frost point when below freezing
            _dp[_temp < 273.15] = _fp[_temp < 273.15]

        _axis = fig.timeseries_axes([0.1, 0.58, 0.8, 0.25], labelx=False)

        cr2 = self.dataset["TDEWCR2C"].data
        ge = self.dataset["TDEW_GE"].data

        dew_min = fig.filter_in_flight(cr2).dropna().min()
        dew_max = fig.filter_in_flight(cr2).dropna().max()

        _axis.plot(cr2, label="CR2")
        _axis.plot(ge, label="GE")
        if wvss2_available:
            _axis.plot(_dp, label=f"WVSS2-F")

        _axis.set_ylabel("Dew point (K)")
        _axis.set_ylim([dew_min, dew_max])
        _axis.legend(fontsize=6)

    def make_tdew_scatter(self, fig: QAFigure) -> None:
        """
        Create a scatter plot of dewpoint temperature, comparing the GE and
        CR2.
        """
        _axis = fig.axes([0.1, 0.1, 0.35, 0.2])

        index = self.dataset["TDEWCR2C"].index.intersection(
            self.dataset["TDEW_GE"].index
        )

        _cr2 = self.dataset["TDEWCR2C"].data.loc[index]
        _ge = self.dataset["TDEW_GE"].data.loc[index]
        _wow = self.dataset["WOW_IND"].data.loc[index]

        _ge.loc[_wow == 1] = np.nan
        _cr2.loc[_wow == 1] = np.nan

        _axis.scatter(_cr2, _ge, 1, color="#03dbfc")

        _axis.add_121()

        _axis.set_xlabel("$T_{D}$ GE")
        _axis.set_ylabel("$T_{D}$ CR2")

    def make_vmr_scatter(self, fig: QAFigure) -> None:
        """
        Create a scatter plot of VMR, comparing the CR2 and WVSS2-R.
        """

        _index = self.dataset["VMR_CR2"].index.intersection(
            self.dataset["WVSS2F_VMR_C"].index
        )

        _axis = fig.axes([0.55, 0.1, 0.35, 0.2])

        _cr2 = self.dataset["VMR_CR2"].data.loc[_index]
        _wvss2 = self.dataset["WVSS2F_VMR_C"].data.loc[_index]
        _wow = self.dataset["WOW_IND"].data.loc[_index]

        _cr2.loc[_wow == 1] = np.nan
        _wvss2.loc[_wow == 1] = np.nan

        _axis.scatter(_cr2, _wvss2, 1, color="#03dbfc")

        _axis.set_xlabel("VMR CR2")
        _axis.set_ylabel("VMR WVSS2-F")
        _axis.add_121()

    def make_vmr_timeseries(self, fig: QAFigure) -> None:
        """
        Make a timeseries of Volume Mixing Ratio for the Buck, GE, and WVSS2-R.
        """

        _temp = self.dataset["TDEW_GE"].data
        _press = self.dataset["PS_RVSM"].data * 100
        _wow = self.dataset["WOW_IND"].data

        index = _temp.index.intersection(_press.index).intersection(_wow.index)

        # Calculate vapour pressure from the GE using Sonntag 1990
        ew = np.exp(
            -6096.9385 * (_temp**-1)
            + 21.2409642
            - 2.711193e-2 * _temp
            + 1.673952e-5 * _temp**2
            + 2.433502 * np.log(_temp)
        )

        ge = 1e6 * ew.loc[index] / (_press.loc[index] - ew.loc[index])

        _axis = fig.timeseries_axes([0.1, 0.33, 0.8, 0.25])

        cr2 = self.dataset["VMR_CR2"].data

        try:
            wvss = self.dataset[f"WVSS2F_VMR_C"].data
            wvss2_available = True
        except KeyError:
            wvss2_available = False

        # Create a common index, so we can filter by WOW
        index = (
            cr2.index.intersection(ge.index)
            .intersection(wvss.index)
            .intersection(_wow.index)
        )

        # Get the max in-flight vmr, to fix y limits
        max_cr2 = cr2.loc[index].loc[_wow == 0].max()
        max_ge = ge.loc[index].loc[_wow == 0].max()

        if wvss2_available:
            max_wvss = wvss.loc[index].loc[_wow == 0].max()
        else:
            max_wvss = 0

        y_max = np.max([max_cr2, max_ge, max_wvss]) + 2000

        # Plot data
        _axis.plot(cr2, label="CR2")
        _axis.plot(ge, label="GE")

        if wvss2_available:
            _axis.plot(wvss, label=f"WVSS2-F")

        _axis.set_ylabel("VMR")
        _axis.set_ylim([0, y_max])
        _axis.legend(fontsize=6)

    def make_text(self, fig: QAFigure) -> None:
        """
        Make some text to describe the agreement between the GE and the CR2.
        'GOOD' if in flight, T > -20, out of cloud, mean delta < 1 K, 'BAD'
        otherwise'.
        """

        # Required variables
        _ge = self.dataset["TDEW_GE"].data
        _cr2 = self.dataset["TDEWCR2C"].data
        _wow = self.dataset["WOW_IND"].data
        _temp = self.dataset["TAT_DI_R"].data

        try:
            _clearair = self.dataset["NV_CLEAR_AIR_MASK"].data
        except KeyError:
            _text = "No cloud mask available; GE/CR2 not compared"
            _col = "gray"
            fig.text(0.5, 0.91, _text, color=_col, horizontalalignment="center")

        # Create a common index
        _index = (
            _ge.index.intersection(_cr2.index)
            .intersection(_wow.index)
            .intersection(_temp.index)
            .intersection(_clearair.index)
        )

        # Comparison where wow == 0, temp > -20
        _temp = _temp.loc[_index].loc[_wow == 0]
        _ge = _ge.loc[_index].loc[_wow == 0].loc[_temp > 253].loc[_clearair == 1]
        _cr2 = _cr2.loc[_index].loc[_wow == 0].loc[_temp > 253].loc[_clearair == 1]

        # Calculate whether agreement is 'good' or not.
        _diff = np.abs(_ge - _cr2).mean()
        if _diff > 1:
            _text = "Some disagreement out of cloud ($\\Delta T_D > 1 K$)"
            _col = "red"
        else:
            _text = "Good agreement out of cloud ($\\Delta T_D < 1 K$)"
            _col = "green"

        # Add text to figure
        fig.text(0.5, 0.91, _text, color=_col, horizontalalignment="center")

    def run(self) -> None:
        """
        Entry point for QA figure.
        """
        with QAFigure(self.dataset, "Humidity") as fig:
            self.make_cloud_plot(fig)
            self.make_tdew_timeseries(fig)
            self.make_vmr_timeseries(fig)
            self.make_tdew_scatter(fig)
            self.make_vmr_scatter(fig)
            self.make_text(fig)

import numpy as np

from mpl_toolkits.basemap import Basemap  # type: ignore

from .base import QAMod, QAFigure
from ppodd.utils.conversions import uv_to_spddir


class GINQA(QAMod):
    inputs = [
        "LAT_GIN",
        "LON_GIN",
        "ALT_GIN",
        "ROLL_GIN",
        "HDG_GIN",
        "PALT_RVS",
        "U_NOTURB",
        "V_NOTURB",
    ]

    def make_map(self, fig: QAFigure) -> None:
        """
        Make a map showing the flight track.
        """
        _ax = fig.axes([0.1, 0.6, 0.8, 0.31])

        m = Basemap(
            projection="tmerc",
            urcrnrlat=self.lat.max() + 1.5,
            urcrnrlon=self.lon.max() + 3,
            llcrnrlat=self.lat.min() - 1.5,
            llcrnrlon=self.lon.min() - 3,
            resolution="i",
            lat_0=self.lat.mean(),
            lon_0=self.lon.mean(),
        )

        m.plot(
            self.lon.values, self.lat.values, latlon=True, linewidth=3, color="#cc0000"
        )
        m.drawcoastlines()
        m.fillcontinents()
        m.drawcountries()
        m.drawmapboundary(fill_color="#b0e9ff")

    def make_alt_plot(self, fig: QAFigure) -> None:
        """
        Make a timeseries plot showing GIN and RVSM altitudes.
        """
        _ax = fig.timeseries_axes([0.1, 0.42, 0.8, 0.15], labelx=False)
        _ax.plot(self.alt, label="GIN alt.")
        _ax.plot(self.alt_rvs, label="RVSM alt.")
        _ax.legend(fontsize=6)
        _ax.set_ylabel("Altitude (m)")

    def make_roll_hdg_plot(self, fig: QAFigure) -> None:
        """
        Create a timeseries plot showing GIN roll and heading.
        """
        _ax, _ax2 = fig.timeseries_axes(
            [0.1, 0.27, 0.8, 0.15], labelx=False, twinx=True
        )

        _ax.plot(self.hdg, ".", markersize=3, color="green", label="Heading")
        _ax.set_ylabel("Heading")
        _ax.legend(fontsize=6, loc="upper right")
        _ax2.plot(self.roll, color="purple", label="Roll")
        _ax2.set_ylabel("Roll")
        _ax2.legend(fontsize=6, loc="upper left")

    def make_gin_wind_plot(self, fig: QAFigure) -> None:
        """
        Create a timeseries plot showing the noturb wind speed and direction.
        """
        _ax, _ax2 = fig.timeseries_axes([0.1, 0.12, 0.8, 0.15], twinx=True)
        _spd, _dir = uv_to_spddir(self.u, self.v)
        _ax2.plot(_dir, ".", color="red", markersize=3, label="Wind Dir.")
        _ax.plot(_spd, label="Wind Speed")

        _ax.set_ylabel("Wind spd (m/s)")
        _ax2.set_ylabel("Wind dir.")

    def run(self) -> None:
        """
        QA plotting entry point.
        """
        with QAFigure(self.dataset, "GIN") as fig:

            # Get required variables
            _wow = self.dataset["WOW_IND"].data
            _lat = self.dataset["LAT_GIN"].data
            _lon = self.dataset["LON_GIN"].data
            _alt = self.dataset["ALT_GIN"].data
            _roll = self.dataset["ROLL_GIN"].data
            _hdg = self.dataset["HDG_GIN"].data
            _alt_rvs = self.dataset["PALT_RVS"].data
            _u = self.dataset["U_NOTURB"].data
            _v = self.dataset["V_NOTURB"].data

            # Make a common index
            _index = _wow.index.intersection(_lat.index).intersection(_alt_rvs.index)

            # Save variables to instance
            self.lat = _lat.loc[_index].loc[_wow == 0]
            self.lon = _lon.loc[_index].loc[_wow == 0]
            self.alt = _alt.loc[_index].loc[_wow == 0]
            self.roll = _roll.loc[_index].loc[_wow == 0]
            self.hdg = _hdg.loc[_index].loc[_wow == 0]
            self.alt_rvs = _alt_rvs.loc[_index].loc[_wow == 0]
            self.u = _u.loc[_index].loc[_wow == 0]
            self.v = _v.loc[_index].loc[_wow == 0]

            # Create plots
            self.make_map(fig)
            self.make_alt_plot(fig)
            self.make_roll_hdg_plot(fig)
            self.make_gin_wind_plot(fig)

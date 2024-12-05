import numpy as np
from statistics import NormalDist

import pandas as pd

from .base import QAMod, QAFigure


class TEiSO2V2ZeroQA(QAMod):
    """
    Produces a QC plot for the TEi SO2 zeroes run during a flight. This gives an
    idea of the current detection limit of the instrument.
    """

    inputs = ["CHTSOO_concentration", "CHTSOO_V6", "CHTSOO_V7"]

    def plot_timeseries(self, in_zero: pd.Series, fig: QAFigure) -> None:
        """
        Polt a timeseries of the zero valve states and the reported SO2 concentration
        when the valve states are indicating that the instrument is in zero mode.
        """
        ax, ax2 = fig.timeseries_axes([0.1, 0.8, 0.8, 0.12], twinx=True)

        ax.plot(in_zero, alpha=0.5)
        conc = self.dataset["CHTSOO_concentration"]().copy()
        conc2 = conc.copy()
        conc[in_zero == 0] = np.nan
        conc2[in_zero != 0] = np.nan
        ax2.plot(conc2, color="gray")
        ax2.plot(conc, color="tab:orange")

        ax.set_ylabel("Zero state")
        ax2.set_ylabel("[SO2]")

    def run(self) -> None:
        """
        Entry hook
        """
        in_zero = (self.dataset["CHTSOO_V6"]() == 1) | (
            self.dataset["CHTSOO_V7"]() == 1
        )
        conc = self.dataset["CHTSOO_concentration"]().copy()

        with QAFigure(self.dataset, "TEi 43i SO2 Zero") as fig:
            self.plot_timeseries(in_zero, fig)

            # hax is the distribution axis
            hax = fig.axes([0.1, 0.1, 0.8, 0.48])

            # ax is the grouped zero axis
            ax = fig.timeseries_axes([0.1, 0.65, 0.8, 0.12])

            # Generate pandas groups for each zero.
            groups = (in_zero != in_zero.shift()).cumsum()
            groups[in_zero != 1] = np.nan
            groups.dropna(inplace=True)
            g = conc.groupby(groups)

            # Create a plot which shows each mean-corrected zero group
            a = np.array([])
            b = []
            bt = []
            for _, n in g:
                ax.plot(n - n.mean(), ".", alpha=0.3)
                b.append(n.std())
                bt.append(n.index[0])
                a = np.concatenate([a, n - n.mean()])

            ax.fill_between(bt, b, -np.array(b), alpha=0.2)
            ax.axhline(0, color="k", linewidth=0.5)
            ax.set_ylabel("[SO2] zero")

            # Get a normal distribution
            norm = NormalDist.from_samples(a)
            mean = norm.mean
            std = norm.stdev

            # PLot the fit
            x = np.arange(-3, 3, 0.01)
            # (implement normal ourselves, because idiocy)
            y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
                -0.5 * ((x - mean) / std) ** 2
            )

            # Plot the histogram
            n, bins, _ = hax.hist(
                a, 30, density=True, facecolor="tab:green", edgecolor="gray", alpha=0.7
            )
            hax.plot(x, y)
            hax.axvline(x=0, ymin=0, ymax=1, color="k", linewidth=0.5)

            n_lim = np.max(np.abs(np.max(bins))) + 0.2

            the_max = np.max(n) + 0.1
            hax.fill_between([-std, std], y1=0, y2=the_max, alpha=0.5)
            hax.set_ylim([0, the_max])
            hax.set_xlim([-n_lim, n_lim])
            hax.set_ylabel("Density")
            hax.set_xlabel("Zero value (mean corrected)")

            hax.set_title(f"Mean = {mean:0.3f}, Std. dev. = {std:0.3f}")

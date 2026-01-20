import numpy as np
import scipy

from ppodd.utils import get_constant_groups

from .base import QAMod, QAFigure


class AL55COCalQA(QAMod):
    """
    Produces a QA plot for the AL55CO calibration data during a flight. This gives an
    idea of the performance of the calibration.

    -------------------------------------------
    |           AL55CO Calibrations           |
    |                                         |
    | ======================================  |
    | |   Calibration Events Timeseries    |  |
    | ======================================  |
    |                                         |
    | ==================  ==================  |
    | | Calibration Fits| | Sens & Zero    |  |
    | ==================  ==================  |
    |                                         |
    | ======================================  |
    | |      Target Comparisons            |  |
    | ======================================  |
    |                                         |
    -------------------------------------------
    """

    inputs = [
        "CO_AERO",  # The calibrated CO concentration
        "AL55CO_V1",  # V1 valve state (raw)
        "AL55CO_V2",  # V2 valve state (raw)
        "AL55CO_V3",  # V3 valve state (raw)
        "AL55CO_V4",  # V4 valve state (raw)
        "AL55CO_CALCULATED_CALS",  # The calculated calibration data
        "AL55CO_interpolated_sens",  # The interpolated sensitivity
        "AL55CO_interpolated_zero",  # The interpolated zero
        "AL55CO_TAR_MR",  # Target cylinder concentration
        "AL55CO_TAR_EU",  # Target cylinder uncertainty
        "AL55CO_TAR_SN",  # Target cylinder serial number
    ]

    def make_calibration_timeseries(self, fig: QAFigure) -> None:
        """
        Makes a timeseries plot showing the calibration events during the flight.

        Args:
            fig (QAFigure): The figure to plot on.
        """

        # 20 minute padding either side of flight so we can see pre/post flight cals
        ax, ax2 = fig.timeseries_axes(
            [0.1, 0.75, 0.8, 0.15], twinx=True, flight_padding_mins=20
        )

        # Plot the AL55CO counts
        ax.plot(
            (self.dataset["AL55CO_counts"]() / 1000),
            color="black",
            linewidth=0.5,
        )

        # Get valve states, and use to identify cal types
        v1 = self.dataset["AL55CO_V1"]()
        v2 = self.dataset["AL55CO_V2"]()
        v3 = self.dataset["AL55CO_V3"]()
        v4 = self.dataset["AL55CO_V4"]()

        hi_cal_mask = v1 & v2 & v3 & v4
        lo_cal_mask = v1 & v2 & (~v3) & v4
        target_mask = v1 & (~v2) & (~v3) & v4

        # Plot valve states
        ax2.plot(hi_cal_mask, color="red", label="High Cal", alpha=0.7)
        ax2.plot(lo_cal_mask, color="blue", label="Low Cal", alpha=0.7)
        ax2.plot(target_mask, color="green", label="Target", alpha=0.7)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(["Off", "On"])
        ax2.legend(loc="upper left", fontsize=6)

        # Axes labels and titles
        ax.set_ylabel("AL55CO Counts (kcps)")
        ax.set_title("AL55CO Calibration Events")
        ax2.set_ylabel("Valve State")

    def make_calibration_fits(self, fig: QAFigure) -> None:
        """
        Makes a plot showing the calibration fits for each calibration event.

        Args:
            fig (QAFigure): The figure to plot on.
        """
        ax = fig.axes([0.1, 0.45, 0.35, 0.25])

        # The calibration data should have been put into the AL55CO_CALCULATED_CALS variable
        # during processing.
        cals = self.dataset["AL55CO_CALCULATED_CALS"]

        # Plot each calibration
        for cal in cals:
            points = cal.points
            if points is None:
                continue

            x_vals = np.array([points[0].x, points[1].x])
            y_vals = np.array([points[0].y, points[1].y])

            # Plot the fit line
            x_fit = np.array([min(x_vals) - 10, max(x_vals) + 10])
            y_fit = cal.sensitivity * x_fit + cal.zero

            ax.plot(x_fit, y_fit, linewidth=0.5)

            # Plot the calibration points with error bars
            ax.errorbar(
                x_vals,
                y_vals,
                xerr=[points[0].x_unc, points[1].x_unc],
                yerr=[points[0].y_unc, points[1].y_unc],
                fmt="",
                color="black",
                linestyle="none",
                linewidth=0.5,
            )

        # Axes labels and titles
        ax.set_title("Calibration Fits")
        ax.set_xlabel("CO Concentration (ppb)")
        ax.set_ylabel("AL55CO Counts (Hz)")

    def make_sens_and_zero_plots(self, fig: QAFigure) -> None:
        """
        Makes a plot showing the interpolated sensitivity and zero over the flight.

        Args:
            fig (QAFigure): The figure to plot on.
        """
        ax, ax2 = fig.timeseries_axes(
            [0.55, 0.45, 0.35, 0.25], flight_padding_mins=20, twinx=True
        )

        # Plot interpolated sensitivity and zero
        ax.plot(self.dataset["AL55CO_interpolated_sens"](), label="Sensitivity")
        ax2.plot(
            self.dataset["AL55CO_interpolated_zero"](), label="Zero", color="tab:orange"
        )

        # Overplot the calibration points, with error bars
        cals = self.dataset["AL55CO_CALCULATED_CALS"]
        for cal in cals:
            ax.errorbar(
                cal.time,
                cal.sensitivity,
                yerr=cal.sensitivity_unc,
                fmt="o",
                color="tab:blue",
                alpha=0.5,
                capsize=2,
            )

            ax2.errorbar(
                cal.time,
                cal.zero,
                yerr=cal.zero_unc,
                fmt="o",
                color="tab:orange",
                alpha=0.5,
                capsize=2,
            )

        # Axes labels and titles
        ax.set_title("Interpolated Sensitivity and Zero")
        ax.set_ylabel("Sensitivity (Hz ppb⁻¹)")
        ax2.set_ylabel("Zero (Hz)")

        # Add legends
        ax.legend(loc="upper left", fontsize=6)
        ax2.legend(loc="upper right", fontsize=6)

    def make_target_plots(self, fig: QAFigure) -> None:
        """
        Makes a plot showing the target comparison during the flight,
        comparing the measured CO_AERO concentration to the target cylinder
        concentration.

        Args:
            fig (QAFigure): The figure to plot on.
        """

        # ===================
        # 1. Timeseries plot
        # ===================

        ax = fig.axes([0.1, 0.1, 0.6, 0.28])

        # Get the target comparison data and metadata
        target_comparison = self.dataset["AL55CO_TARGET_COMPARISON"]
        conc = self.dataset["CO_AERO"]()
        cylinder_conc = self.dataset["AL55CO_TAR_MR"]
        cylinder_conc_cu = self.dataset["AL55CO_TAR_EU"] / 2
        cylinder_sn = self.dataset["AL55CO_TAR_SN"]

        # Plot each target segment, ignoring gaps when not in target mode
        offset = 0
        groups = get_constant_groups(target_comparison.mask)
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)

        # Plot each target segment
        for i, group in enumerate(groups):
            ax.plot(
                np.arange(len(group[1])) + offset,
                conc.loc[group[1].index] - cylinder_conc,
                linewidth=0.5,
                color="tab:red",
            )
            offset += len(group[1]) + 5
            if i < len(groups) - 1:
                ax.axvline(offset - 2.5, color="black", linestyle="-", linewidth=0.5)

        # Save x limits for later
        xlims = ax.get_xlim()

        # Add the uncertainty band of the target cylinder
        ax.fill_between(
            ax.get_xlim(),
            [cylinder_conc_cu, cylinder_conc_cu],
            [-cylinder_conc_cu, -cylinder_conc_cu],
            color="lightgray",
            alpha=0.5,
            label="±1σ uncertainty",
        )

        # Restore x limits
        ax.set_xlim(xlims)

        ax.set_title(
            f"CO_AERO offset during targets (vs cylinder {cylinder_sn} @ {cylinder_conc:.2f} ±{cylinder_conc_cu:.2f} ppb)"
        )

        ax.set_ylabel("$\Delta$CO_AERO (ppb)")
        ax.set_xticks([])
        ylims = ax.get_ylim()

        # ==================
        # 2. Histogram plot
        # ==================

        ax = fig.axes([0.7, 0.1, 0.2, 0.28])
        target_comparison = self.dataset["AL55CO_TARGET_COMPARISON"]
        conc = self.dataset["CO_AERO"]()

        # Create histogram of target differences where in target mode
        targets = conc[target_comparison.mask == 1]
        ax.hist(
            targets - cylinder_conc,
            bins=np.arange(-10, 11, 0.5),
            color="tab:red",
            alpha=0.7,
            edgecolor="tab:red",
            orientation="horizontal",
            density=True,
        )

        # Fit a normal distribution to the data
        mu, std = scipy.stats.norm.fit(targets - cylinder_conc)

        # Plot the fitted distribution
        xmin, xmax = ax.get_ylim()
        x = np.linspace(xmin, xmax, 100)
        p = scipy.stats.norm.pdf(x, mu, std)
        ax.plot(p, x, "k", linewidth=2, alpha=0.5)

        # Add text box with fit results
        ax.text(
            0.63,
            0.92,
            f"μ = {mu:.2f} ppb\nσ = {std:.2f} ppb",
            transform=ax.transAxes,
            fontsize=6,
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.5)

        # Axes labels and titles
        ax.set_yticks([])
        ax.set_ylim(ylims)
        ax.set_xlabel("Density")

    def run(self) -> None:
        """
        Entry point to create the QA figure.
        """
        with QAFigure(self.dataset, "AL55CO Calibrations") as fig:
            self.make_calibration_timeseries(fig)
            self.make_calibration_fits(fig)
            self.make_sens_and_zero_plots(fig)
            self.make_target_plots(fig)

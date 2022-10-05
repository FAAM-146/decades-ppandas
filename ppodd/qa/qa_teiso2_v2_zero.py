import numpy as np
from statistics import NormalDist

from .base import QAMod, QAFigure


class TEiSO2V2ZeroQA(QAMod):
    """
    Produces a QC plot for the TEi SO2 zeroes run during a flight. This gives an
    idea of the current detection limit of the instrument.
    """

    inputs = [
        'CHTSOO_concentration',
        'CHTSOO_V6',
        'CHTSOO_V7'
    ]

    def plot_timeseries(self, in_zero, fig):
        """
        Polt a timeseries of the zero valve states and the reported SO2 concentration
        when the valve states are indicating that the instrument is in zero mode.
        """
        ax, ax2 = fig.timeseries_axes([.1, .8, .8, .12], twinx=True)
        
        ax.plot(in_zero, alpha=.5)
        conc = self.dataset['CHTSOO_concentration']()
        conc[in_zero == 0] = np.nan
        ax2.plot(conc, color='tab:orange')

    def run(self):
        """
        Entry hook
        """
        in_zero = (self.dataset['CHTSOO_V6']() == 1) | (self.dataset['CHTSOO_V7']() == 1)
        conc = self.dataset['CHTSOO_concentration']()

        with QAFigure(self.dataset, 'TEi 43i SO2 Zero') as fig:
            self.plot_timeseries(in_zero, fig)

            # hax is the distribution axis
            hax = fig.axes([.1, .1, .8, .48])

            # ax is the grouped zero axis
            ax = fig.timeseries_axes([.1, .65, .8, .12])

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
                ax.plot(n - n.mean(), '.', alpha=.3)
                b.append(n.std())
                bt.append(n.index[0])
                a = np.concatenate([a, n - n.mean()])
            
            ax.fill_between(bt,b,-np.array(b), alpha=.2)
            ax.axhline(0, color='k', linewidth=.5)

            # Get a normal distribution
            norm = NormalDist.from_samples(a)
            mean = norm.mean
            std = norm.stdev

            # PLot the fit
            x = np.arange(-3, 3, .01)
            # (implement normal ourselves, because idiocy)
            y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-.5 * ((x-mean)/std)**2)

            # Plot the histogram
            n, bins, _ = hax.hist(a, 30, density=True, facecolor='tab:green', edgecolor='gray', alpha=.7)
            hax.plot(x, y)
            hax.axvline(x=0, ymin=0, ymax=1, color='k' , linewidth=.5)

            n_lim = np.max(np.abs(np.max(bins))) + .2

            the_max = np.max(n) + .1
            hax.fill_between([-std, std], y1=0, y2=the_max, alpha=.5)
            hax.set_ylim([0, the_max])
            hax.set_xlim([-n_lim, n_lim])
            hax.set_title(f'Mean = {mean:0.3f}, Std. dev. = {std:0.3f}')
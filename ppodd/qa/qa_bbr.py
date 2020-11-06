import numpy as np

from .base import QAMod, QAFigure

class BBR(QAMod):
    inputs = [
        'SW_DN_C',
        'SW_UP_C',
        'RED_UP_C',
        'RED_DN_C',
        'SOL_AZIM',
        'SOL_ZEN',
        'HDG_GIN'
    ]

    def make_solar_timeseries(self, fig):
        ax, ax2 = fig.timeseries_axes(
            [.1, .75, .8, .15], labelx=False, twinx=True
        )

        az = fig.filter_in_flight(self.dataset['SOL_AZIM'].data)
        zen = fig.filter_in_flight(self.dataset['SOL_ZEN'].data)
        hdg = fig.filter_in_flight(self.dataset['HDG_GIN'].data)

        ax2.set_ylabel('Solar angle ($^\circ$)')
        ax.set_ylabel('Relative sun angle ($^\circ$)')
        rel_ang = (az-hdg.reindex(az.index)) % 360
        ax.plot(rel_ang, '.', color='tab:green', markersize=3,
                label='Relative sun angle')
        ax2.plot(az, label='Solar Azimuth')
        ax2.plot(zen, label='Solar Zenith')
        ax.set_ylim([0, 360])
        ax2.legend(fontsize=6, loc='lower right')
        ax.legend(fontsize=6, loc='lower left')

    def make_upper_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .43, .8, .3], labelx=False)
        clear = fig.filter_in_flight(self.dataset['SW_DN_C'])
        red = fig.filter_in_flight(self.dataset['RED_DN_C'])
        ax.plot(clear, 'k', label='Upper clear dome flux')
        ax.plot(red, 'tab:red', label='Upper red dome flux')
        ax.legend(fontsize=6, loc='upper left')

    def make_lower_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .11, .8, .3])
        clear = fig.filter_in_flight(self.dataset['SW_UP_C'])
        red = fig.filter_in_flight(self.dataset['RED_UP_C'])
        ax.plot(clear, 'k', label='Lower clear dome flux')
        ax.plot(red, 'tab:red', label='Lower red dome flux')
        ax.legend(fontsize=6, loc='upper left')


    def run(self):
        with QAFigure(self.dataset, 'Broadband Radiometers', landscape=True) as fig:
            self.make_solar_timeseries(fig)
            self.make_upper_timeseries(fig)
            self.make_lower_timeseries(fig)

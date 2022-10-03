from turtle import position
import numpy as np

from .base import QAMod, QAFigure
from ppodd.utils.calcs import sp_mach


class TEiSO2V2QA(QAMod):
    inputs = [
        'CHTSOO_concentration',
        'SO2_TECO',
        'SO2_TECO_ZERO',
        'WOW_IND',
    ]

    def make_figure_1(self, fig):
        ax1, ax2 = fig.timeseries_axes([.1, .80, .8, .12], labelx=False, twinx=True)
        ax1.plot(self.dataset['CHTSOO_concentration'], label='Raw conc.')
        ax1.plot(self.dataset['SO2_TECO_ZERO'], linewidth=1.5, color='tab:red', label='Interp. zero')
        ax1.set_ylim([-10, 60])
        ax1.set_ylabel('Unscaled SO2 (ppb)')
        ax1.axhline([0], color='gray', linewidth=.5, alpha=.5)

        ax2.plot(self.dataset['CHTSOO_V6'], linewidth=.5, label='eV6 (ground only)')
        ax2.plot(self.dataset['CHTSOO_V7'], linewidth=.5, label='eV7 (ground and inflight)')
        ax2.plot(self.dataset['CHTSOO_V8'], linewidth=.5, label='eV8 (calibration span)')
        ax2.set_ylim([0, 8])
        ax2.set_ylabel('Valve state')
        ax2.set_yticks([])

        ax1.legend(loc='upper left', fontsize=6)
        ax2.legend(loc='upper right', fontsize=6)

    def make_figure_2(self, fig):
        ax = fig.timeseries_axes([.1, .68, .8, .12], labelx=False)
        so2 = self.dataset['SO2_TECO']()
        flags = self.dataset['SO2_TECO'].flag()
        so2.loc[flags>0] = np.nan
        ax.plot(so2, label='SO2 conc. (scaled)')
        ax.legend(loc='upper right', fontsize=6)
        ax.axhline([0], linewidth=1, color='tab:red')
        ax.set_ylabel('Scaled SO2 (ppb)')

    def make_figure_3(self, fig):
        ax1, ax2 = fig.timeseries_axes([.1, .56, .8, .12], labelx=False, twinx=True)
        ax1.plot(self.dataset['CHTSOO_cRIO_temp_c'], label='CRIO temp')
        ax1.plot(self.dataset['CHTSOO_internal_temper'], label='Internal temp')
        ax1.set_ylim([15, 35])
        ax1.set_ylabel('Temp (C)')

        ax2.plot(self.dataset['CHTSOO_react_temper'], color='tab:red', label='React. temp')
        ax2.set_ylim([40, 50])
        ax1.set_ylabel('React temp (C)')

        ax1.legend(loc='upper left', fontsize=6)
        ax2.legend(loc='upper right', fontsize=6)

    def make_figure_4(self, fig):
        ax1, ax2 = fig.timeseries_axes([.1, .44, .8, .12], labelx=False, twinx=True)
        ax1.plot(self.dataset['CHTSOO_react_press'], label='React. press.')
        ax1.set_ylim([500, 850])
        ax1.set_ylabel('React. press. (hPa)')

        ax2.plot(self.dataset['CHTSOO_MFC3_absolute_pressure'], label='MFC3 abs. press', color='tab:orange')
        ax2.set_ylabel('MFC abs press. (hPa)')

        ax1.legend(loc='lower left', fontsize=6)
        ax2.legend(loc='lower right', fontsize=6)


    def make_figure_5(self, fig):
        ax1, ax2 = fig.timeseries_axes([.1, .32, .8, .12], labelx=False, twinx=True)
        ax1.plot(self.dataset['CHTSOO_MFM'], label='MFM')

        ax2.plot(self.dataset['CHTSOO_MFC2_mass_flow'], label='MFC2 mass flow', color='tab:orange')
        ax2.plot(self.dataset['CHTSOO_MFC3_mass_flow'], label='MFC3 mass flow', color='tab:green')

        ax1.set_ylabel('MFM')
        ax2.set_ylabel('MFC')

        ax1.legend(loc='lower left', fontsize=6)
        ax2.legend(loc='lower right', fontsize=6)

    def make_figure_6(self, fig):
        ax1, ax2 = fig.timeseries_axes([.1, .20, .8, .12], labelx=False, twinx=True)

        ax1.plot(self.dataset['CHTSOO_pmt_volt']() * -1, label='PMT Volt')
        ax1.plot(self.dataset['CHTSOO_lamp_volt'], label='Lamp Volt')
        ax1.set_ylim([500, 1200])

        ax2.plot(self.dataset['CHTSOO_lamp_intens'], label='Lamp Intens.', color='tab:green')
        ax2.set_ylim([80, 100])

        ax1.legend(loc='lower left', fontsize=6)
        ax2.legend(loc='lower right', fontsize=6)

    def make_figure_7(self, fig):
        ax1 = fig.timeseries_axes([.1, .08, .8, .12])
        ax1.plot(self.dataset['SO2_TECO'].flag(), label='Flag')
        ax1.legend(loc='lower left', fontsize=6)
        ax1.set_ylabel('Flag Value')

    def run(self):
        with QAFigure(self.dataset, 'TEi 43i SO2') as fig:
            self.make_figure_1(fig)
            self.make_figure_2(fig)
            self.make_figure_3(fig)
            self.make_figure_4(fig)
            self.make_figure_5(fig)
            self.make_figure_6(fig)
            self.make_figure_7(fig)
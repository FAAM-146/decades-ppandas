import numpy as np

from .base import QAMod, QAFigure


class TwoBOzone(QAMod):
    inputs = [
        'O3_2BTECH',
        'TWBOZO_MFM',
        'TWBOZO_flow',
        'TWBOZO_press',
        'TWBOZO_temp',
        'TWBOZO_cRIO_temp_c'
    ]

    def make_conc_plots(self, fig):
        have_teco = True

        ax1 = fig.timeseries_axes([.1, .73, .5, .2])
        ax2 = fig.axes([.65, .73, .25, .2])

        ax1.plot(
            fig.filter_in_flight(self.dataset['O3_2BTECH']()),
            label='2BTech'
        )

        try:
            teco = self.dataset['O3_TECO']()
            if teco.max() == 0:
                raise ValueError

            ax1.plot(
                fig.filter_in_flight(teco),
                label='TECO'
            )
        except (KeyError, ValueError):
            have_teco = False

        ax1.legend(loc='lower left', fontsize=6)
        ax1.set_ylabel('Ozone (ppb)')

        if not have_teco:
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.text(.1, .5, 'O3_TECO not available')
            return

        ax2.scatter(
            fig.filter_in_flight(self.dataset['O3_2BTECH']()),
            fig.filter_in_flight(self.dataset['O3_TECO']()),
            color='tab:orange',
            alpha=.5
        )

        ax2.set_ylabel('TECO O3')
        ax2.set_xlabel('2BTech O3')

        ax2.add_121()

    def make_flow(self, fig):
        ax1, ax2 = fig.timeseries_axes([.1, .55, .8, .15], twinx=True, labelx=False)
        ax1.plot(self.dataset['TWBOZO_MFM'], color='tab:blue', label='2B MFM')
        ax1.legend(loc='lower left', fontsize=6)
        ax1.axhline(1.2, linestyle='--', color='tab:blue')
        ax1.set_ylabel('MFM (lpm)')
        
        ax2.plot(self.dataset['TWBOZO_flow'], color='tab:orange', label='2B flow')
        ax2.legend(loc='lower right', fontsize=6)
        ax2.axhline(1500, linestyle='--', color='tab:orange')
        ax2.set_ylabel('flow (sccm)')

    def make_press(self, fig):
        ax1, ax2 = fig.timeseries_axes([.1, .4, .8, .15], twinx=True, labelx=False)
        ax1.plot(fig.filter_in_flight(
            self.dataset['TWBOZO_press']
        ), color='tab:green', label='2B Pressure')
        ax1.legend(loc='lower left', fontsize=6)
        ax1.set_ylabel('2B press (Torr)')

        try:
            ax2.plot(fig.filter_in_flight(
                self.dataset['CAB_PRES']
            ) * 0.750062, color='tab:purple', label='Cabin Pressure')

            ax2.legend(loc='lower right', fontsize=6)
            ax2.set_ylabel('Cabin press (Torr)')
        except KeyError:
            ax2.set_yticks([])

    def make_temp(self, fig):
        ax1 = fig.timeseries_axes([.1, .25, .8, .15])
        ax1.plot(fig.filter_in_flight(self.dataset['TWBOZO_temp']), label='2B temp.')
        ax1.plot(fig.filter_in_flight(self.dataset['TWBOZO_cRIO_temp_c']), label='2B crio temp.')
        ax1.set_ylabel('Temp (C)')
        ax1.legend(loc='lower left', fontsize=6)
        
    def make_zero(self, fig):
        ax = fig.axes([.1, .07, .8, .15])

        try:
            v6 = self.dataset['TWBOZO_V6']()
        except KeyError:
            ax.text(.35, .5, '2B Valve state not available')
            ax.set_xticks([])
            ax.set_yticks([])
            return

        conc = self.dataset['TWBOZO_conc']()
        in_zero = (v6 == 1)
       
        if in_zero.max() == 0:
            ax.text(.37, .5, 'No zeroes were performed')
            ax.set_xticks([])
            ax.set_yticks([])
            return

        groups = (in_zero != in_zero.shift()).cumsum()
        groups[in_zero != 1] = np.nan
        groups.dropna(inplace=True)
        g = conc.groupby(groups)

        m = 0
        for _, df in g:
            ax.plot(df.values)
            if len(df) > m:
                m = len(df)

        ax.fill_between([0, m], -3, 3, color='tab:green', alpha=.3)
        ax.set_xlim([0, m])
        ax.set_ylim([-5, 5])

    def run(self):
        """
        QA plotting entry point.
        """
        with QAFigure(self.dataset, '2B TECH Ozone') as fig:
            self.make_conc_plots(fig)
            self.make_flow(fig)
            self.make_press(fig)
            self.make_temp(fig)
            self.make_zero(fig)
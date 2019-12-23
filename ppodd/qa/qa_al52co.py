import datetime
import numpy as np
import matplotlib.pyplot as plt

from .base import QAMod, QAFigure
from ppodd.pod.p_al52co import CAL_FLUSH_TIME


class AL52QA(QAMod):
    inputs = [
        'CO_AERO',
        'AL52CO_lamptemp',
        'AL52CO_lampflow',
        'AL52CO_monoflow',
        'AL52CO_lamptemp',
        'AL52CO_cal_status',
        'AL52CO_cellpress',
        'AL52CO_conc'
    ]

    def make_lamptemp(self, fig):
        lamptemp = self.dataset['AL52CO_lamptemp'].data
        ax = fig.timeseries_axes([.1, .8, .8, .1], labelx=False)
        ax.plot(lamptemp.loc[lamptemp < 100], label='Lamp temp.')
        ax.legend(fontsize=6)
        ax.set_ylabel('Temp. (degC)')

    def make_flows(self, fig):
        lampflow = self.dataset['AL52CO_lampflow'].data
        monoflow = self.dataset['AL52CO_monoflow'].data
        ax = fig.timeseries_axes([.1, .69, .8, .1], labelx=False)
        ax.plot(lampflow, label='Lamp flow')
        ax.plot(monoflow.loc[monoflow != 0], label='Mono flow')
        ax.legend(fontsize=6)
        ax.set_ylabel('Flow (sccm)')

    def make_cellpress(self, fig):
        cellpress = self.dataset['AL52CO_cellpress'].data
        ax = fig.timeseries_axes([.1, .58, .8, .1])
        ax.plot(cellpress, label='Cell press.')
        ax.legend(fontsize=6)
        ax.set_ylabel('Torr')

    def make_co(self, fig):
        co_u = self.dataset['AL52CO_conc'].data
        co_c = self.dataset['CO_AERO'].data
        ax, ax2 = fig.timeseries_axes([.1, .15, .8, .4], twinx=True)

        co_u_d = co_u.copy()
        co_u_d.loc[co_u_d < -10] = np.nan
        co_c_d = co_c.copy()
        co_c_d.loc[co_c_d < -10] = np.nan

        delta = (co_u_d - co_c_d).rolling(60).mean().reindex(self.cal.index)
        delta.loc[self.cal == 1] = np.nan
        delta.loc[co_u < -10] = np.nan
        ax2.plot(delta, color='g', label='dCO', linewidth=.5)
        ax2.set_ylim([-5, 5])
        ax2.set_ylabel('delta CO')

        ax.plot(co_u, label='CO raw', linewidth=1)
        ax.plot(co_c, label='CO corr.', linewidth=1)
        ax.legend(fontsize=6)
        ax.set_ylabel('CO conc. (ppb)')
        ax.set_ylim([-50, (co_c.mean() + co_c.std()*2)])

        ax2.plot(self.cal.loc[self.cal == 1], 'ko')

    def run(self):
        """
        QA plotting entry point.
        """
        with QAFigure(self.dataset, 'AL52 Carbon Monoxide') as fig:

            cal = self.dataset['AL52CO_cal_status'].data.copy()
            _groups = (cal != cal.shift()).cumsum()
            _groups[cal < 1] = np.nan
            _groups.dropna(inplace=True)
            groups = cal.groupby(_groups)

            in_cal = self.dataset['AL52CO_cal_status'].data.copy() * 0

            for group in groups:
                start = group[1].index[0]
                end = group[1].index[-1] + datetime.timedelta(seconds=CAL_FLUSH_TIME)
                in_cal.loc[start:end] = 1

            self.cal = in_cal

            self.make_lamptemp(fig)
            self.make_flows(fig)
            self.make_cellpress(fig)
            self.make_co(fig)

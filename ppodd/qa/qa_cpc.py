import numpy as np

from .base import QAMod, QAFigure

class CPCQA(QAMod):
    inputs = [
        'CPC378_counts',
        'CPC378_sample_flow',
        'CPC378_total_flow',
        'CPC378_sheath_flow',
        'CPC378_saturator_temp',
        'CPC378_growth_tube_temp',
        'CPC378_optics_temp'
    ]

    def make_flow_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .7, .8, .2])
        ax.plot(self.dataset['CPC378_sample_flow'], label='Sample flow')
        ax.plot(self.dataset['CPC378_sheath_flow'], label='Sheath flow')
        ax.legend(loc='lower left', fontsize=8)
        ax.set_ylim([250, 350])
        ax.set_ylabel('Flows')
        ax2 = ax.twinx()
        ax2.plot(self.dataset['CPC378_total_flow'], 'r', label='Total flow')
        for label in ax2.get_yticklabels():
            label.set_fontsize(6)
        ax2.yaxis.label.set_fontsize(6)
        ax2.legend(loc='lower right', fontsize=8)
        ax2.set_ylim([400, 700])
        ax2.set_ylabel('Total Flows')

    def make_temp_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .5, .8, .2])
        ax.plot(
            self.dataset['CPC378_saturator_temp'], 'r',
            label='Saturator temp'
        )
        ax.legend(loc='lower left', fontsize=8)
        ax.set_ylabel('Saturator temp.')
        ax2 = ax.twinx()
        for label in ax2.get_yticklabels():
            label.set_fontsize(6)
        ax2.yaxis.label.set_fontsize(6)
        ax2.plot(self.dataset['CPC378_growth_tube_temp'], label='Growth tube temp')
        ax2.plot(self.dataset['CPC378_optics_temp'], label='Optics temp')
        ax2.set_ylabel('Optics/GT temp.')
        ax2.legend(loc='lower right', fontsize=8)

    def make_counts_timeseries(self, fig):
        ax = fig.timeseries_axes([.1, .15, .8, .3])
        ax2 = ax.twinx()
        ax2.yaxis.label.set_fontsize(6)
        for label in ax2.get_yticklabels():
            label.set_fontsize(6)

        ts = self.dataset['CPC378_counts'].data.loc[
            (self.dataset['CPC378_counts'].data.index > fig.to_time) &
            (self.dataset['CPC378_counts'].data.index < fig.land_time)
        ]

        ax.plot(ts, color='gray', label='CPC #')
        ax.set_ylabel('CPC # (log)')
        ax2.plot(ts, label='CPC #')

        ax.set_yscale('symlog')
        ax2.set_ylabel('CPC # (lin)')



    def run(self):
        with QAFigure(self.dataset, 'CPC3781') as fig:
            self.make_flow_timeseries(fig)
            self.make_temp_timeseries(fig)
            self.make_counts_timeseries(fig)

import numpy as np

from .base import QAMod, QAFigure


class NevzorovQA(QAMod):

    inputs = [
        'NV_TWC_COL_P',
        'NV_LWC1_COL_P',
        'NV_LWC2_COL_P',
        'NV_REF_P',
        'NV_LWC1_U',
        'NV_LWC2_U',
        'NV_TWC_U',
        'NV_LWC1_C',
        'NV_LWC2_C',
        'NV_TWC_C',
        'NV_CLEAR_AIR_MASK'
    ]

    def make_power_timeseries(self, fig):
        _ax = fig.timeseries_axes([.1, .7, .8, .2], labelx=False)

        _index = self.dataset['NV_LWC1_COL_P'].index
        _mask = (_index > fig.to_time) & (_index < fig.land_time)
        max_power = np.max([
            self.dataset['NV_REF_P'].data.loc[_mask],
            self.dataset['NV_LWC1_COL_P'].data.loc[_mask],
            self.dataset['NV_LWC2_COL_P'].data.loc[_mask],
            self.dataset['NV_TWC_COL_P'].data.loc[_mask]
        ])

        _ax.plot(self.dataset['NV_REF_P'].data, label='Ref. power')
        _ax.plot(self.dataset['NV_LWC1_COL_P'].data, label='LWC1 power')
        _ax.plot(self.dataset['NV_LWC2_COL_P'].data, label='LWC2 power')
        _ax.plot(self.dataset['NV_TWC_COL_P'].data, label='TWC power')
        _ax.legend(fontsize=6, loc='upper right')
        _ax.set_ylim([0, max_power + 2])


    def make_uncorrected_timeseries(self, fig):
        _ax = fig.timeseries_axes([.1, .5, .8, .2], labelx=False)

        _index = self.dataset['NV_TWC_U'].index
        _mask = (_index > fig.to_time) & (_index < fig.land_time)

        twc = self.dataset['NV_TWC_U'].data
        lwc1 = self.dataset['NV_LWC1_U'].data
        lwc2 = self.dataset['NV_LWC2_U'].data

        max_u = np.max([twc.loc[_mask], lwc1.loc[_mask], lwc2.loc[_mask]])
        min_u = np.min([twc.loc[_mask], lwc1.loc[_mask], lwc2.loc[_mask]])

        _ax.plot(twc, label='TWC (U)')
        _ax.plot(lwc1, label='LWC1 (U)')
        _ax.plot(lwc2, label='LWC2 (U)')
        _ax.legend(fontsize=6, loc='upper right')
        _ax.set_ylim([min_u, max_u + .1])

    def make_corrected_timeseries(self, fig):
        _ax = fig.timeseries_axes([.1, .3, .8, .2], labelx=False)

        _index = self.dataset['NV_TWC_U'].index
        _mask = (_index > fig.to_time) & (_index < fig.land_time)

        twc = self.dataset['NV_TWC_C'].data
        lwc1 = self.dataset['NV_LWC1_C'].data
        lwc2 = self.dataset['NV_LWC2_C'].data

        max_c = np.max([twc.loc[_mask], lwc1.loc[_mask], lwc2.loc[_mask]])
        min_c = np.min([twc.loc[_mask], lwc1.loc[_mask], lwc2.loc[_mask]])

        _ax.plot(twc, label='TWC (C)')
        _ax.plot(lwc1, label='LWC1 (C)')
        _ax.plot(lwc2, label='LWC2 (C)')
        _ax.legend(fontsize=6, loc='upper right')
        _ax.set_ylim([min_c, max_c + .1])


    def make_mask_plot(self, fig):
        lwc_axis = fig.timeseries_axes([.1, .2, .8, .1])
        clear_air = self.dataset['NV_CLEAR_AIR_MASK'].data.asfreq('1S')
        cloud = 1 - clear_air
        wow = self.dataset['WOW_IND'].data.asfreq('1S')
        wow = wow.reindex(cloud.index).bfill().ffill()
        cloud.loc[wow == 1] = np.nan
        _x = np.abs(np.vstack((cloud, cloud)))
        lwc_axis.pcolormesh(cloud.index, [0, 1], _x, cmap='Reds')
        lwc_axis.set_ylabel('Mask', rotation=0, labelpad=20)
        lwc_axis.set_yticks([])

    def run(self):
        with QAFigure(self.dataset, 'Nevzorov', landscape=True) as fig:
            self.make_power_timeseries(fig)
            self.make_uncorrected_timeseries(fig)
            self.make_corrected_timeseries(fig)
            self.make_mask_plot(fig)

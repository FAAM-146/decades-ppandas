import numpy as np

from .base import QAMod, QAFigure


class NevzorovQA(QAMod):

    inputs = [
        'NV_TWC_COL_P',
        'NV_LWC1_COL_P',
        'NV_LWC2_COL_P',
        'NV_REF_P',
        'NV_LWC1_C',
        'NV_LWC2_C',
        'NV_TWC_C',
        'NV_CLEAR_AIR_MASK'
    ]

    def make_power_timeseries(self, fig: QAFigure) -> None:
        """
        Create a timeseries plot of the power of the Nevzorov elements.
        """

        _ax = fig.timeseries_axes([.1, .75, .8, .15], labelx=False)

        _index = self.dataset['NV_TWC_COL_P'].index

        try:
            _mask = (_index > fig.to_time) & (_index < fig.land_time)
        except Exception:
            _mask = np.ones_like(_index, dtype=bool)

        twc = self.dataset['NV_TWC_COL_P'].data
        lwc1 = self.dataset['NV_LWC1_COL_P'].data
        lwc2 = self.dataset['NV_LWC2_COL_P'].data
        ref = self.dataset['NV_REF_P'].data

        max_u = np.max([twc.loc[_mask], lwc1.loc[_mask], lwc2.loc[_mask]])
        min_u = np.min([twc.loc[_mask], lwc1.loc[_mask], lwc2.loc[_mask]])

        _ax.plot(twc, label='TWC P', alpha=.6)
        _ax.plot(lwc1, label='LWC1 P', alpha=.6)
        _ax.plot(lwc2, label='LWC2 P', alpha=.6)
        _ax.plot(ref, label='Ref. P', alpha=.6)
        _ax.legend(fontsize=6, loc='upper left')
        _ax.set_ylabel('Element powers (W)')
        _ax.axhline(0, color='k', linewidth=.5)
        try:
            _ax.set_ylim([min_u - 10, max_u + 10])
        except Exception:
            pass

    def make_corrected_timeseries(self, fig: QAFigure) -> None:
        """
        Create a timeseries plot of the corrected water content of the Nevzorov
        elements.
        """

        _ax = fig.timeseries_axes([.1, .6, .8, .15], labelx=False)

        _index = self.dataset['NV_REF_P'].index
        try:
            _mask = (_index > fig.to_time) & (_index < fig.land_time)
        except Exception:
            _mask = np.ones_like(_index, dtype=bool)

        twc = self.dataset['NV_TWC_C'].data
        lwc1 = self.dataset['NV_LWC1_C'].data
        lwc2 = self.dataset['NV_LWC2_C'].data

        max_c = np.max([twc.loc[_mask], lwc1.loc[_mask], lwc2.loc[_mask]])
        min_c = np.min([twc.loc[_mask], lwc1.loc[_mask], lwc2.loc[_mask]])

        _ax.plot(twc, label='TWC (C)', alpha=.6)
        _ax.plot(lwc1, label='LWC1 (C)', alpha=.6)
        _ax.plot(lwc2, label='LWC2 (C)', alpha=.6)
        _ax.legend(fontsize=6, loc='upper left')
        _ax.set_ylabel('Corrected water ($g/m^3$)')
        _ax.axhline(0, color='k', linewidth=.5)

        try:
            _ax.set_ylim([min_c, max(max_c + .1, 1)])
        except Exception:
            pass

    def make_mask_plot(self, fig: QAFigure) -> None:
        """
        Create a plot of the clear air mask.
        """

        lwc_axis = fig.timeseries_axes([.1, .55, .8, .05], labelx=False)
        clear_air = self.dataset['NV_CLEAR_AIR_MASK'].data.asfreq('1S')
        cloud = 1 - clear_air
        wow = self.dataset['WOW_IND'].data.asfreq('1S')
        wow = wow.reindex(cloud.index).bfill().ffill()
        cloud.loc[wow == 1] = np.nan
        _x = np.abs(np.vstack((cloud, cloud)))
        lwc_axis.pcolormesh(cloud.index, [0, 1], _x, cmap='Reds')
        lwc_axis.set_ylabel('Mask', rotation=0, labelpad=20)
        lwc_axis.set_yticks([])

    def make_k_plot(self, fig: QAFigure) -> None:
        """
        Make a timeseries plot of the measured and parameterised k values
        """
        k_axis = fig.timeseries_axes([.1, .35, .8, .2])
        colors = ['tab:blue', 'tab:orange', 'tab:green']

        for ele, color in zip(['TWC', 'LWC1', 'LWC2'], colors):
            power = self.dataset[f'NV_{ele}_COL_P'.upper()]()
            ref = self.dataset['NV_REF_P']()
            ratio = power / ref
            ratio.loc[self.dataset['NV_CLEAR_AIR_MASK']()==0] = np.nan

            k_axis.plot(ratio, label=f'{ele} power ratio', linewidth=4, alpha=.5, color=color)
            k_axis.plot(self.dataset[f'NV_{ele}_K'], label=f'{ele} K', color=color, linewidth=1.5)
        
        k_axis.set_ylabel('Dry air col/ref power ratio')

    def make_lwc_comparison(self, fig: QAFigure) -> None:
        """
        Make a scatter plot of LWC1 vs LWC2
        """
        lwc_axis = fig.axes([.1, .1, .3, .2])

        lwc1 = self.dataset['NV_LWC1_C']().loc[
            self.dataset['NV_CLEAR_AIR_MASK']()==0
        ].rolling(64).mean().asfreq('s')
        lwc2 = self.dataset['NV_LWC2_C']().loc[
            self.dataset['NV_CLEAR_AIR_MASK']()==0
        ].rolling(64).mean().asfreq('s')

        lwc_axis.scatter(lwc1, lwc2, s=2, label='LWC1', alpha=.8, color='tab:red')
        lwc_axis.set_xlim(min(lwc1.min(), lwc2.min())-.05, max(lwc1.max(), lwc2.max())+.05)
        lwc_axis.set_ylim(min(lwc1.min(), lwc2.min())-.05, max(lwc1.max(), lwc2.max())+.05)
        lwc_axis.set_xlabel('LWC1 ($g/m^3$)')
        lwc_axis.set_ylabel('LWC2 ($g/m^3$)')
        lwc_axis.add_121()
        lwc_axis.set_title('LWC1 vs LWC2')

    def make_k_comparison(self, fig: QAFigure) -> None:
        """
        Make a histogram of the measured vs parameterised k values for
        TWC, LWC1 and LWC2
        """

        k_axis1 = fig.axes([.45, .1, .15, .2])
        k_axis2 = fig.axes([.6, .1, .15, .2])
        k_axis3 = fig.axes([.75, .1, .15, .2])

        def _make_plot(ax, col, title=None, y_label=None):
            """
            Create a single histogram plot of the measured vs parameterised k

            Args:
                ax: The axis to plot on
                col: The column to plot

            Kwargs:
                title: The title of the plot
                y_label: The y label of the plot
            """

            power = self.dataset[f'NV_{col}_COL_P'.upper()]()
            ref = self.dataset['NV_REF_P']()
            ratio = power / ref
            ratio.loc[self.dataset['NV_CLEAR_AIR_MASK']()==0] = np.nan

            ax.hist2d(
                ratio.dropna(),
                self.dataset[f'NV_{col}_K'.upper()].loc[ratio.dropna().index],
                bins=30, cmap='inferno_r', cmin=1
            )

            ax.text(.1, .9, f'{col}', transform=ax.transAxes, fontsize=8)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.add_121()
            if y_label:
                ax.set_ylabel(y_label)
            if title:
                ax.set_title(title)

        _make_plot(k_axis1, 'TWC', y_label='Parameterised $k$')
        _make_plot(k_axis2, 'LWC1', title='Measured vs parameterised $k$')
        _make_plot(k_axis3, 'LWC2')


    def run(self):
        with QAFigure(self.dataset, 'Nevzorov') as fig:
            self.make_power_timeseries(fig)
            self.make_corrected_timeseries(fig)
            self.make_k_plot(fig)
            self.make_mask_plot(fig)
            self.make_lwc_comparison(fig)
            self.make_k_comparison(fig)

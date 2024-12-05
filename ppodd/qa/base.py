import abc
import datetime
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from ppodd.decades.dataset import DecadesDataset


class QAMod(abc.ABC):
    inputs: list[str] = []

    def __init__(self, dataset: DecadesDataset):
        self.dataset = dataset

    def ready(self):
        for _input in self.inputs:
            if _input not in self.dataset.variables:
                return False
        return True

    @abc.abstractmethod
    def run(self):
        """Create a QA Figure"""


class QAAxis(object):
    def __init__(self, ax):
        self._ax = ax

    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except KeyError:
            pass

        return getattr(self._ax, attr)

    def add_121(self, linewidth=1):
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        _start = np.min([xlim[0], ylim[0]])
        _end = np.max([xlim[1], ylim[1]])
        self._ax.plot(
            [_start, _end], [_start, _end], '--k', linewidth=linewidth
        )
        self._ax.set_xlim([_start, _end])
        self._ax.set_ylim([_start, _end])

    def add_zero_line(self):
        xlim = self._ax.get_xlim()
        self._ax.plot(
            xlim, (0, 0), color='k', zorder=10, linewidth=.5, linestyle='--'
        )
        self._ax.set_xlim(xlim)


class QAFigure(object):

    def __init__(self, dataset, title, subtitle=None, landscape=False):

        _figsize = (8, 8*1.414)
        if landscape:
            _figsize = _figsize[::-1]

        self._fig = plt.figure(figsize=_figsize)
        self._ts_axes = []
        self._axes = []
        self.title = title
        self.subtitle = subtitle
        self.dataset = dataset
        self.landscape = landscape

        try:
            self.flightnum = self.dataset.globals['flight_number']
        except KeyError:
            self.flightnum = 'nXXX'

        try:
            self.flight_date = self.dataset.date
        except KeyError:
            raise

        self._fig.text(
            .5, .97, 'QA: {}'.format(title),
            horizontalalignment='center',
            size='x-large'
        )

        if subtitle:
            self._fig.text(
                .5, .95, subtitle,
                horizontalalignment='center',
                size='x-small'
            )

        self._fig.text(
            .5, .02, 'Produced on {}'.format(
                datetime.datetime.utcnow().strftime('%Y-%m-%d at %H:%Mz')
            ), horizontalalignment='center', size='x-small'
        )

    def _set_sizes(self, axes=None, size=6):
        if axes is None:
            axes = self._ts_axes + self._axes

        for ax in axes:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                         + ax.get_xticklabels() + ax.get_yticklabels()):

                item.set_fontsize(6)

    def _finalize(self):
        self._set_sizes()
        for ax in self._ts_axes:
            if self.to_time is not None and self.land_time is not None:
                ax.set_xlim([
                    self.to_time - datetime.timedelta(minutes=5),
                    self.land_time + datetime.timedelta(minutes=5)
                ])
                ax.axvline([self.to_time], color='gray', linewidth=2,
                            alpha=.5)
                ax.axvline([self.land_time], color='gray', linewidth=2,
                            alpha=.5)

    def _savefig(self):
        if self.dataset.qa_dir:
            _save_path = self.dataset.qa_dir
        else:
            _save_path = '.'

        _save_file = '{}_{}_{}.pdf'.format(
            self.flightnum, self.title.replace(' ', ''),
            'l' if self.landscape else 'p'
        )

        _save_file = os.path.join(_save_path, _save_file)

        self._fig.savefig(_save_file)

    def __enter__(self):
        to_time = self.dataset.takeoff_time
        land_time = self.dataset.landing_time

        try:
            self.to_time = pd.to_datetime(to_time)
        except Exception:
            self.to_time = self.dataset.time_bounds[0]

        try:
            self.land_time = pd.to_datetime(land_time)
        except Exception:
            self.land_time = self.dataset.time_bounds[1]

        self.set_subtitle(
            'Report for {flight}, on {date}'.format(
                flight=self.flightnum.upper(),
                date=self.flight_date.strftime('%Y-%m-%d')
            )
        )

        return self

    def __exit__(self, *args):
        self._finalize()
        self._savefig()

    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except KeyError:
            pass

        return getattr(self._fig, attr)

    def filter_in_flight(self, var):
        try:
            return var.loc[
                (var.index > self.to_time) & (var.index < self.land_time)
            ]
        except Exception:
            return var

    def set_subtitle(self, subtitle):
        self.text(.5, .95, subtitle, horizontalalignment='center',
                  size='x-small')

    def timeseries_axes(self, location, twinx=False, labelx=True):
        def _set_xticks(_ax):
            hours = mdates.HourLocator()
            minutes = mdates.MinuteLocator(byminute=range(10, 60, 10))
            hours_formatter = mdates.DateFormatter('%Hz')
            _ax.xaxis.set_major_locator(hours)
            _ax.xaxis.set_major_formatter(hours_formatter)
            _ax.xaxis.set_minor_locator(minutes)

        ax = QAAxis(self._fig.add_axes(location))
        self._ts_axes.append(ax)
        if labelx:
            _set_xticks(ax)

        if twinx:
            ax2 = QAAxis(ax.twinx())
            if labelx:
                _set_xticks(ax2)
            self._ts_axes.append(ax2)
            self._set_sizes([ax, ax2])

            if not labelx:
                ax.set_xticks([])
                ax2.set_xticks([])

            return ax, ax2

        self._set_sizes([ax])
        if not labelx:
            ax.set_xticks([])

        return ax

    def axes(self, location, twinx=False):
        ax = QAAxis(self._fig.add_axes(location))
        self._axes.append(ax)

        if twinx:
            ax2 = QAAxis(ax.twinx())
            self._axes.append(ax2)
            self._set_sizes([ax, ax2])

            return ax, ax2

        self._set_sizes([ax])
        return ax

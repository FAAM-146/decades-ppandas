import abc
import datetime

import numpy as np
import pandas as pd

from netCDF4 import Dataset

from ..utils import pd_freq


class DecadesWriter(abc.ABC):

    def __init__(self, dataset):
        self.dataset = dataset

    @abc.abstractmethod
    def write(self, filename):
        """Write some output"""


class NetCDFWriter(DecadesWriter):

    _FillValue = -9999

    def __init__(self, *args, **kwargs):
        self.output_freqs = []
        self.start_time = None
        self.end_time = None
        self.nc = None
        self.sps_vars = {}

        super().__init__(*args, **kwargs)

        self._get_time_bounds()
        self._get_output_freqs()

    def _get_output_freqs(self):
        """Get all of the required output frequencies"""
        output_freqs = []
        for var in self.dataset.outputs:
            if var.frequency not in output_freqs:
                output_freqs.append(var.frequency)
        self.output_freqs = output_freqs


    def _get_time_bounds(self):
        """Get the earliest and latest times of all output variables"""
        start_time = datetime.datetime(2999, 1, 1)
        end_time = datetime.datetime(1900, 1, 1)

        for var in self.dataset.outputs:
            if var.data.index[0] < start_time:
                start_time = var.data.index[0]
            if var.data.index[-1] > end_time:
                end_time = var.data.index[-1]

        start_time = start_time.replace(microsecond=0)

        self.start_time = start_time
        self.end_time = end_time

    def _write_var(self, nc, var):
        _var_name = var.name
        _flag_name = '{}_FLAG'.format(var.name)

        _freq = pd_freq[var.frequency]

        if 60 % var.frequency == 0:
            _end = self.end_time
        else:
            _end = self.end_time + datetime.timedelta(seconds=1/var.frequency)

        _index = pd.DatetimeIndex(
            start=self.start_time,
            end=_end,
            freq=_freq
        )

        _data = var.data.reindex(_index).fillna(self._FillValue)
        _flag = var.flag.reindex(_index)
        _flag.loc[~np.isfinite(_flag)] = 3

        print('Writing: {}'.format(var.long_name))

        _shape = (int(len(_data) / var.frequency), var.frequency)

        # If the frequency of a varable is not 1, then we need to reshape itt
        if var.frequency != 1:
            nc[_var_name][:] = _data.values.reshape(_shape)
            nc[_flag_name][:] = _flag.values.reshape(_shape)
        else:
            nc[_var_name][:] = _data.values
            nc[_flag_name][:] = _flag.values

        for _name in (_var_name, _flag_name):
            nc[_name].units = var.units
            nc[_name].frequency = np.int32(var.frequency)
            if _name == _var_name:
                nc[_name].long_name = var.long_name
            else:
                nc[_name].long_name = 'Flag for {}'.format(var.long_name)
        if var.standard_name is not None:
            nc[_var_name].standard_name = var.standard_name

    def _init_nc_file(self, nc):
        nc.createDimension('Time', None)
        for freq in self.output_freqs:
            if freq == 1:
                continue
            sps_name = 'sps{0:02d}'.format(freq)
            nc.createDimension(sps_name, freq)

        nc.createVariable('Time', int, ('Time',))
        nc['Time'][:] = list(pd.DatetimeIndex(
            start=self.start_time, end=self.end_time, freq='1s'
        ).astype(int) / 1e9)
        nc['Time'].units = 'seconds since 1970-01-01'

        for var in self.dataset.outputs:
            if var.frequency == 1:
                nc.createVariable(
                    var.name, float, ('Time',),
                    fill_value=self._FillValue
                )
                nc.createVariable(
                    '{}_FLAG'.format(var.name), np.int8, ('Time',)
                )
            else:
                nc.createVariable(
                    var.name,
                    float,
                    ('Time', 'sps{0:02d}'.format(var.frequency)),
                    fill_value=self._FillValue
                )
                nc.createVariable(
                    '{}_FLAG'.format(var.name),
                    np.int8,
                    ('Time', 'sps{0:02d}'.format(var.frequency))
                )

    def _write_global_attrs(self, nc):
        for attr, value in self.dataset.constants.items():
            try:
                setattr(nc, attr, value)
            except TypeError:
                setattr(nc, attr, str(value))

    def write(self, filename, freq=None):
        with Dataset(filename, 'w') as nc:
            self._init_nc_file(nc)
            for var in self.dataset.outputs:
                self._write_var(nc, var)

            self._write_global_attrs(nc)

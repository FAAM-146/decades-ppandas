import abc
import datetime

import numpy as np
import pandas as pd

from netCDF4 import Dataset

from ..utils import pd_freq


class DecadesWriter(abc.ABC):
    """
    Define an interface for classes writing out DecadesDataset data.
    """

    def __init__(self, dataset, *args, **kwargs):
        self.dataset = dataset

    @abc.abstractmethod
    def write(self, filename):
        """Write some output"""


class NetCDFWriter(DecadesWriter):
    """
    Write a DecadesDataset to NetCDF.

    A NetCDFWriter takes a DecadesDataset instance in its constructor and will
    write this dataset to file when its write method is called. This can
    include an optional freq keyword to force the output to a specified
    frequency.
    """

    def __init__(self, *args, **kwargs):
        self.output_freqs = []
        self.start_time = None
        self.end_time = None
        self.nc = None
        self.sps_vars = {}

        super().__init__(*args, **kwargs)

        self._get_time_bounds()
        self.output_freqs = self._get_output_freqs()

    def _get_output_freqs(self):
        """Get all of the required output frequencies"""
        output_freqs = []

        for var in self.dataset.outputs:

            if not var.write:
                continue

            if var.frequency not in output_freqs:
                output_freqs.append(var.frequency)

        return output_freqs

    def _get_time_bounds(self):
        """Get the earliest and latest times of all output variables"""

        start_time = datetime.datetime.max
        end_time = datetime.datetime.min

        for var in self.dataset.outputs:

            if not var.write:
                continue

            if var.data.index[0] < start_time:
                start_time = var.data.index[0]

            if var.data.index[-1] > end_time:
                end_time = var.data.index[-1]

        start_time = start_time.replace(microsecond=0)

        self.start_time = start_time
        self.end_time = end_time

    def _write_var(self, nc, var):
        """
        Write a given DecadesVariable to a netCDF file, including flag.

        Args:
            nc: a netCDF4.Dataset, opened to write
            var: DecadesVariable
        """

        _freq = self.write_freq or var.frequency

        # If the variable is at 1 Hz, it only needs a time dimension,
        # otherwise, it also requires an spsNN dimension.
        if _freq == 1:
            ncvar = nc.createVariable(
                var.name, float, ('Time',),
                fill_value=var.attrs['_FillValue']
            )

            ncflag = nc.createVariable(
                '{}_FLAG'.format(var.name),
                np.int8, ('Time',), fill_value=-1
            )
        else:
            ncvar = nc.createVariable(
                var.name, float, ('Time', 'sps{0:02d}'.format(_freq)),
                fill_value=var.attrs['_FillValue']
            )

            ncflag = nc.createVariable(
                '{}_FLAG'.format(var.name), np.int8,
                ('Time', 'sps{0:02d}'.format(_freq)), fill_value=-1
            )

        # Write variable attributes, excluding _FillValue, which must be given
        # when the variable is created.
        for attr_key, attr_val in var.attrs.items():
            if attr_key is '_FillValue' or attr_val is None:
                continue
            setattr(ncvar, attr_key, attr_val)

        # Add a few required attributes to the flag variable.
        ncflag.standard_name = 'status_flag'

        # Create a new DatetimeIndex to interpolate to, given frequency
        _end = self.end_time - datetime.timedelta(seconds=1/_freq)

        _index = pd.date_range(
            self.start_time,
            _end,
            freq=pd_freq[_freq]
        )

        if _freq == var.frequency:
            # variable is alreay at the correct frequency, all we require is a
            # reindex, filling any missing data with _FillValue, and flagging
            # as a 3
            _data = var.data.reindex(_index).fillna(var.attrs['_FillValue'])
            _flag = var.flag.reindex(_index)
            _flag.loc[~np.isfinite(_flag)] = 3
        else:
            # Variable and flag must be resampled to bring onto the correct
            # frequency and then reindexed. Apply a mean to the data and a pad
            # to the flag.
            _data = var.data.resample(
                pd_freq[_freq], limit=var.frequency-1
            ).apply('mean').reindex(_index)

            _flag = var.flag.resample(
                pd_freq[_freq], limit=var.frequency-1
            ).pad().reindex(_index)

        # Reshape the data if it is not at 1 Hz
        if _freq != 1:
            _data = _data.values.reshape(int(len(_index) / _freq), _freq)
            _flag = _flag.values.reshape(int(len(_index) / _freq), _freq)
        else:
            _data = _data.values
            _flag = _flag.values

        # Finally write the data
        print('Writing {}...'.format(var.attrs['long_name']))
        ncvar[:] = _data
        ncflag[:] = _flag

    def _init_netcdf(self, nc):
        """
        Initialise an output netCDF file, creating dimensions as required.

        Args:
            nc: handle to a netCDF4.Dataset opened for write
        """

        # Create time dimension, variable, and set attributes
        nc.createDimension('Time', None)
        self.time = nc.createVariable('Time', int, ('Time',), fill_value=-1)
        self.time.long_name = 'Time of measurement'
        self.time.standard_name = 'time'
        self.time.calendar = 'gregorian'
        self.time.units = 'seconds since 1970-01-01'

        # If not forcing to a frequency, create a dimension for each frequency
        # that we're going to output.
        if self.write_freq is None:
            for _freq in sorted(self.output_freqs):
                nc.createDimension('sps{0:02d}'.format(_freq), _freq)
        else:
            if self.write_freq != 1:
                # If forcing to non-1hz freq, create the required dimension
                nc.createDimension(
                    'sps{0:02d}'.format(self.write_freq), self.write_freq
                )

    def _write_constant(self, nc, key, value):
        """
        Write a (key, value) pair to a given netCDF handle as global
        attributes. If value is a datetime, coerce to date string. If value is
        a dict, recursively add these as attributes, prepended by {key}_

        Args:
            nc: a netCDF4.Dataset, opened for writing
            key: the global attribute key to write to nc
            value: the global attribute value to write to nc
        """

        # Coerce datetimes to a date string
        try:
            value = value.strftime('%Y-%m-%d')
        except AttributeError:
            pass

        # If given a dict, recursively build keys
        if type(value) is dict:
            for _key, _val in value.items():
                _key = '{}_{}'.format(key, _key)
                self._write_constant(nc, _key, _val)
            return

        setattr(nc, key, value)

    def write(self, filename, freq=None):
        """
        Write a DecadesDataset to a netCDF file, optionally forcing the output
        to a specified frequency. Note that forcing to anything onther than 1
        hz is currently untested.

        Args:
            filename: the path of the netCDF file to create.

        Kwargs:
            freq (None): frequency to force the output to.
        """

        # The frequency to force the output to, if any
        self.write_freq = freq

        with Dataset(filename, 'w') as nc:

            # Init the netCDF file
            self._init_netcdf(nc)

            # Write each variable in turn
            for var in self.dataset.outputs:
                self._write_var(nc, var)

            # Create an index for the Time variable
            dates = pd.date_range(self.start_time, self.end_time,
                                  freq='S')

            # Time will natively be nanoseconds from 1970-01-01, so just
            # convert this to seconds.
            self.time[:] = dates.values.astype(np.int64) / 1e9

            # Write flight constants as global attributes
            for const in self.dataset.constants.items():
                self._write_constant(nc, *const)

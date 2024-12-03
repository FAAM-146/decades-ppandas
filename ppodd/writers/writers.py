import abc
import datetime
import json
import logging
import os

import numpy as np
import pandas as pd

from netCDF4 import Dataset
import sqlite3

from ppodd.utils import pd_freq, try_to_call, unwrap_array, stringify_if_datetime
from ppodd.decades.attributes import ATTRIBUTE_NOT_SET

__all__ = ['SQLiteWriter', 'NetCDFWriter']

logger = logging.getLogger(__name__)


class DecadesWriter(abc.ABC):
    """
    Define an interface for classes writing out DecadesDataset data.
    """

    def __init__(self, dataset, *args, **kwargs):
        self.dataset = dataset
        self.output_freqs = []
        self.start_time = None
        self.end_time = None
        self._get_time_bounds()
        self.output_freqs = self._get_output_freqs()

    @abc.abstractmethod
    def write(self, filename):
        """Write some output"""

    def _get_time_bounds(self):
        """Get the earliest and latest times of all output variables"""

        self.start_time, self.end_time = self.dataset.time_bounds()

        self.start_time = self.start_time.replace(microsecond=0)
        self.end_time = self.end_time.replace(microsecond=0)

    def _get_output_freqs(self):
        """Get all of the required output frequencies"""
        output_freqs = []

        for _var in self.dataset.variables:
            var = self.dataset[_var]

            if not var.write:
                continue

            if var.frequency not in output_freqs:
                output_freqs.append(var.frequency)

        return output_freqs


class SQLiteWriter(DecadesWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _initdb(self, filename):
        self.conn = sqlite3.connect(filename)
        c = self.conn.cursor()

        c.execute('CREATE TABLE var_meta (name text, meta text)')

        self.conn.commit()

    def _write_var(self, var):

        print('Writing {}'.format(var.name))

        _sql = 'CREATE TABLE {} (time integer, data real, flag integer)'
        c = self.conn.cursor()
        c.execute(_sql.format(var.name))

        _times = var.data.index.astype(np.int64) / 1e9
        _data = var.data.values
        _flags = var.flag()

        _pairs = [
            (_time, _datum, _flag)
            for _time, _datum, _flag in zip(_times, _data, _flags)
        ]

        _sql = 'INSERT INTO {} values (?, ?, ?)'.format(var.name)
        c.executemany(_sql, _pairs)

        _meta = {
            'data': {
                'long_name': var.long_name,
                'name': var.name,
                'frequency': var.frequency,
            },
            'flag': var.flag.cfattrs
        }

        for n, i in var.flag.cfattrs.items():
            print(n, i, type(i))

        temp = json.dumps(_meta)
        print(temp)

        _sql = 'INSERT INTO var_meta VALUES (?, ?)'
        c.execute(_sql, (var.name, json.dumps(_meta)))

        self.conn.commit()

    def write(self, filename):
        if os.path.exists(filename):
            print('Warning: overwriting SQLite3 DB file')
            os.remove(filename)

        self._initdb(filename)

        for var in self.dataset.outputs:
            if var.write:
                self._write_var(var)

        self.conn.close()


class NetCDFWriter(DecadesWriter):
    """
    Write a DecadesDataset to NetCDF.

    A NetCDFWriter takes a DecadesDataset instance in its constructor and will
    write this dataset to file when its write method is called. This can
    include an optional freq keyword to force the output to a specified
    frequency.
    """

    def __init__(self, *args, **kwargs):
        self.nc = None
        self._format = kwargs.pop('nc_format', 'NETCDF4_CLASSIC')
        self.sps_vars = {}

        super().__init__(*args, **kwargs)

    def _write_var(self, nc, var):
        """
        Write a given DecadesVariable to a netCDF file, including flag.

        Args:
            nc: a netCDF4.Dataset, opened to write
            var: DecadesVariable
        """

        _freq = self.write_freq or var.frequency
        ncflag = None

        try:
            fill_value = var.attrs['_FillValue']
        except KeyError:
            try:
                fill_value = var.attrs['FillValue']
            except KeyError:
                fill_value = None

        if var.flag is not None:
            try:
                flag_fill_value = var.flag.cfattrs['_FillValue']
            except KeyError:
                try:
                    flag_fill_value = var.flag.cfattrs['FillValue']
                except KeyError:
                    flag_fill_value = None

        # If the variable is at 1 Hz, it only needs a time dimension,
        # otherwise, it also requires an spsNN dimension.
        if _freq == 1:
            ncvar = nc.createVariable(
                var.name, 'f4', ('Time',),
                fill_value=fill_value
            )

            if var.flag is not None:
                ncflag = nc.createVariable(
                    '{}_{}'.format(var.name, var.flag.postfix),
                    np.int8, ('Time',), fill_value=flag_fill_value
                )
        else:
            ncvar = nc.createVariable(
                var.name, 'f4', ('Time', 'sps{0:02d}'.format(_freq)),
                fill_value=fill_value
            )

            if var.flag is not None:
                ncflag = nc.createVariable(
                    '{}_{}'.format(var.name, var.flag.postfix), np.int8,
                    ('Time', 'sps{0:02d}'.format(_freq)),
                    fill_value=flag_fill_value
                )

        # Write variable attributes, excluding _FillValue, which must be given
        # when the variable is created, and frequency, which is set as the
        # write frequency, if given
        for attr_key, attr_val in var.attrs().items():
            if 'FillValue' in attr_key or attr_val is None:
                continue
            if attr_key == 'frequency' and self.write_freq is not None:
                setattr(ncvar, attr_key, self.write_freq)
                continue
            if attr_val == ATTRIBUTE_NOT_SET:
                logger.warning('Unset variable attribute: {}.{}'.format(
                    var.name, attr_key
                ))
                continue

            try:
                setattr(ncvar, attr_key, stringify_if_datetime(attr_val))
            except TypeError:
                logger.error(
                    f'Error setting variable attr {attr_key} = {attr_val} '
                    f'on {var.name}'
                )

        # Add a few required attributes to the flag variable.
        # ncflag.standard_name = 'status_flag'
        if var.flag is not None:
            for attr_key, attr_val in var.flag.cfattrs.items():
                if 'FillValue' in attr_key or attr_val is None:
                    continue
                if attr_key == 'frequency' and self.write_freq is not None:
                    setattr(ncflag, f'{attr_key}', self.write_freq)
                    continue
                setattr(ncflag, attr_key, attr_val)

        # Create a new DatetimeIndex to interpolate to, given frequency
        if self.end_time.microsecond == 0 and _freq != 1:
            _end = self.end_time - datetime.timedelta(seconds=1/_freq)
        else:
            _end = self.end_time

        _index = pd.date_range(
            self.start_time,
            _end,
            freq=pd_freq[_freq]
        )

        if _freq == var.frequency:
            # variable is alreay at the correct frequency, all we require is a
            # reindex, filling any missing data with _FillValue
            _data = var().reindex(_index).fillna(fill_value)
            if var.flag is not None:
                _flag = var.flag().reindex(_index).fillna(
                    flag_fill_value
                )
        else:
            # Variable and flag must be resampled to bring onto the correct
            # frequency and then reindexed. Apply a mean to the data and a pad
            # to the flag.
            if getattr(var, 'circular', False):
                _data = unwrap_array(var())
            else:
                _data = var()

            _data = _data.resample(
                pd_freq[_freq]
            ).apply('mean').reindex(_index).fillna(fill_value)

            if getattr(var, 'circular', False):
                _data[_data != fill_value] %= 360

            if var.flag is not None:
                _flag = var.flag().resample(
                    pd_freq[_freq]
                ).mean().pad().reindex(_index).fillna(flag_fill_value)

        # Reshape the data if it is not at 1 Hz
        if _freq != 1:
            _data = _data.values.reshape(int(len(_index) / _freq), _freq)
            if var.flag is not None:
                _flag = _flag.values.reshape(int(len(_index) / _freq), _freq)
        else:
            _data = _data.values
            if var.flag is not None:
                _flag = _flag.values

        # Finally write the data
        logger.info('Writing {}...'.format(var.attrs['long_name']))
        ncvar[:] = _data
        if var.flag is not None:
            ncflag[:] = _flag

    def _init_netcdf(self, nc):
        """
        Initialise an output netCDF file, creating dimensions as required.

        Args:
            nc: handle to a netCDF4.Dataset opened for write
        """

        # Create time dimension, variable, and set attributes
        nc.createDimension('Time', None)
        self.time = nc.createVariable('Time', 'i4', ('Time',), fill_value=-9999)
        self.time.long_name = 'Time of measurement'
        self.time.standard_name = 'time'
        self.time.calendar = 'gregorian'
        self.time.coverage_content_type = 'coordinate'
        self.time.frequency = 1
        self.time.units = 'seconds since {} 00:00:00 +0000'.format(
            self.dataset.date.strftime('%Y-%m-%d')
        )

        # If not forcing to a frequency, create a dimension for each frequency
        # that we're going to output.
        if self.write_freq is None:
            for _freq in sorted(self.output_freqs):
                if _freq == 1:
                    continue
                nc.createDimension('sps{0:02d}'.format(_freq), _freq)
        else:
            if self.write_freq != 1:
                # If forcing to non-1hz freq, create the required dimension
                nc.createDimension(
                    'sps{0:02d}'.format(self.write_freq), self.write_freq
                )

    def _write_global(self, nc, key, value):
        """
        Write a (key, value) pair to a given netCDF handle as global
        attributes. If value is a datetime, coerce to date string. If value is
        a dict, recursively add these as attributes, prepended by {key}_

        Args:
            nc: a netCDF4.Dataset, opened for writing
            key: the global attribute key to write to nc
            value: the global attribute value to write to nc
        """

        logger.debug(f'Writing global: {key} = {value}')

        if value == ATTRIBUTE_NOT_SET:
            logger.warning(f'Expected global attribute not set: {key}')
            return

        # Coerce datetimes to a date string
        value = stringify_if_datetime(value)

        # If given a dict, recursively build keys
        if type(value) is dict:
            for _key, _val in value.items():
                _key = '{}_{}'.format(key, _key)
                self._write_global(nc, _key, _val)
            return

        try:
            setattr(nc, key, value)
        except Exception:
            logger.error(f'Error setting global attribute {key} = {value}')

    def write(self, filename=None, freq=None):
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

        # If no filename is given, build a standard pattern
        if filename is None:
            _freq = f'_{freq}hz'
            if freq is None:
                _freq = ''
            filename = (
                'core_faam_{date}_v005_r{revision}_{flight}{freq}.nc'.format(
                    date=self.dataset.date.strftime('%Y%m%d'),
                    revision=self.dataset.globals()['revision_number'],
                    flight=self.dataset.globals()['flight_number'],
                    freq=_freq
                )
            )

        with Dataset(filename, 'w', format=self._format) as nc:

            # Init the netCDF file
            self._init_netcdf(nc)

            # Write each variable in turn
            for _var in self.dataset.variables:
                var = self.dataset[_var]
                if var.write:
                    self._write_var(nc, var)
                else:
                    continue

                self.dataset._backend.decache()

            # Create an index for the Time variable
            dates = pd.date_range(
                self.start_time, self.end_time, freq='s'
            )

            # Time will natively be nanoseconds from 1970-01-01, so just
            # convert this to seconds.
            _delta_secs = (
                self.dataset.date - datetime.date(1970, 1 ,1)
            ).total_seconds()

            nc['Time'][:] = [i / 1e9 - _delta_secs for i in
                            dates.values.astype(np.int64)]

            # Set the ID of the dataset before writing to file
            self.dataset.data_id = os.path.basename(
                filename.replace('.nc', '')
            )

            # Write global attributes.
            for _gkey, _gval in self.dataset.globals().items():
                self._write_global(nc, _gkey, _gval)

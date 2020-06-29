import abc
import csv
import datetime
import glob
import json
import os
import re
import sys
import tempfile
import zipfile
import warnings

from dateutil import relativedelta

import numpy as np
import pandas as pd
import yaml


from ppodd.decades import DecadesVariable
from ppodd.readers import register
from ..utils import pd_freq

C_BAD_TIME_DEV = 43200

class FileReader(abc.ABC):
    """
    An abstract class which should be subclassed to implement a decades
    file reader.
    """
    level = 0

    def __init__(self):
        self.files = []
        self.variables = []
        self.metadata = []

    @abc.abstractmethod
    def read(self, decades_file):
        """Override the read method to implement a FileReader."""

    def __eq__(self, other):
        return self.__class__ == other.__class__


class FlightConstantsReader(FileReader):
    """
    Read a flight constants file.
    """
    def _load(self, *args, **kwargs):
        raise NotImplementedError

    def read(self):
        for _file in self.files:
            print('Reading {}'.format(_file.filepath))
            consts = self._load(_file)

            for key, value in consts['Globals'].items():
                if type(value) is list:
                    value = '\n'.join(value)

                _file.dataset.add_global(key, value)

            for mod_name, mod_content in consts['Constants'].items():
                for key, value in mod_content.items():
                    _file.dataset.constants[key] = value

            for key, val in consts['Modifications']['Variables'].items():
                _file.dataset._variable_mods[key] = val

            for pp_mod in consts['Modifications']['Exclude']:
                _file.dataset._mod_exclusions.append(pp_mod)


@register(patterns=['.*\.json'])
class JsonConstantsReader(FlightConstantsReader):
    def _load(self, _file):
        with open(_file.filepath, 'r') as _consts:
            consts = json.loads(_consts.read())
            return consts


@register(patterns=['.*\.yaml'])
class YamlConstantsReader(FlightConstantsReader):
    def _load(self, _file):
        with open(_file.filepath, 'r') as _consts:
            consts = yaml.load(_consts, Loader=yaml.Loader)
            return consts


@register(patterns=['.*\.zip'])
class ZipFileReader(FileReader):
    """
    Read a Zip file. Zip files are assumed to comtain other files which all
    have associated FileReaders.
    """
    level = 0

    def read(self):
        """
        Reads a Zip file.
        """
        _tempdir = tempfile.mkdtemp()
        print('Working in {}'.format(_tempdir))

        for _file in self.files:
            _dataset = _file.dataset
            print('Reading {}...'.format(_file))
            _zipfile = zipfile.ZipFile(_file.filepath)
            _zipfile.extractall(_tempdir)

        for _file in glob.glob(os.path.join(_tempdir, '*')):
            print(_file)
            _dataset.add_file(_file)

@register(patterns=['(^SEAPROBE|.{8})_.+_\w\d{3}'])
class TcpFileReader(FileReader):
    level = 2
    time_variable = 'utc_time'

    def scan(self, dfile, definition):
        print('Scanning {}...'.format(dfile))

        filename = dfile.filepath

        with open(filename, 'rb') as f:
            rawdata = f.read()

        offsets = self._get_packet_offsets(definition, rawdata)
        packet_lens = np.diff(offsets)
        np.append(packet_lens, 1)

        good_indicies = packet_lens == definition.packet_length

        megaslice = [
            slice(i, i+definition.packet_length)
            for i, j in zip(offsets, good_indicies) if j
        ]

        index = np.zeros(len(rawdata), dtype=bool)

        for i, _slice in enumerate(megaslice):
            index[_slice] = True

        rawbytes = np.frombuffer(rawdata, 'b')
        _output = np.frombuffer(rawbytes[index], dtype=definition.dtypes)

        return _output

    def _get_packet_offsets(self, definition, rawdata):
        rex = re.compile(b'\$' + definition.identifier.encode())
        offsets = [i.start() for i in rex.finditer(rawdata)]
        return offsets

    def _scan_read_packet_len(self, definition, offset, rawdata):
        packet_dlen = definition.dtypes['packet_length'].itemsize
        id_length = definition.dtypes[0].itemsize

        packet_length = np.frombuffer(
            rawdata[offset+id_length:offset+id_length+packet_dlen],
            dtype=definition.dtypes['packet_length']
        )[0]

        return packet_length

    def _get_definition(self, _file):
        for _definition in _file.dataset.definitions:
            _crio_type = os.path.basename(_file.filepath).split('_')[0]
            if _definition.identifier == _crio_type:
                return _definition

    def _get_index_fast(self, time, frequency):
        """
        Returns an interpolated (subsecond) index using a fast,
        but potentially error-prone method.

        args:
            _data: a named np.ndarray, assumed to contain
                'utc_time'
            frequency: the frequency to interpolate to (s).

        returns:
            a pandas.DatetimeIndex
        """
        _time = np.append(
            time, [time[-1] + 1]
        )
        index = pd.to_datetime(_time, unit='s')
        return pd.date_range(_time[0], time[-1], freq=pd_freq[frequency])[:-1]

    def _get_index_slow(self, time, frequency):
        """
        Returns an interpolated (subsecond) index using a slightly,
        slower, but more robust method.

        args:
            _data: a named np.ndarray, assumed to contain
                'utc_time'
            frequency: the frequency to interpolate to (s).

        returns:
            a pandas.DatetimeIndex
        """
        _time = np.append(
            time, [time[-1] + 1]
        )
        _index = pd.to_datetime(_time, unit='s')
        _index = _index.unique()

        df = pd.DataFrame(index=_index)
        df['temp'] = 0

        i = df.asfreq(
            pd_freq[frequency]
        ).interpolate(
            method='time', limit=frequency-1
        ).dropna().index

        return i[:-1]

    def _get_index(self, var, name, time, definition):
        try:
            frequency = definition.dtypes[name].shape[0]
        except IndexError:
            frequency = 1

        if frequency != 1:
            if frequency not in self._index_dict:
                try:
                    self._index_dict[frequency] = self._get_index_fast(time, frequency)
                except ValueError:
                    # Hacky - why do we need this (TODO)
                    self._index_dict[frequency] = np.array([])
                if self._index_dict[frequency].shape != var.ravel().shape:
                    self._index_dict[frequency] = self._get_index_slow(time, frequency)
        else:
            if 1 not in self._index_dict:
                self._index_dict[1] = pd.to_datetime(time, unit='s')

        return frequency, self._index_dict[frequency]

    def _get_group_name(self, definition):
        return definition.identifier[:-2]

    def read(self):
        for _file in sorted(self.files, key=lambda x: os.path.basename(x.filepath)):
            self.dataset = _file.dataset
            self._index_dict = {}

            definition = self._get_definition(_file)

            if definition is None:
                warnings.warn(
                    'No CRIO definition found for {}'.format(_file.filepath),
                    RuntimeWarning
                )
                continue

            dtypes = definition.dtypes

            print('Reading {}...'.format(_file))
            _data = np.fromfile(_file.filepath, dtype=dtypes)

            _read_fail = False
            for d in _data:
                try:
                    data_id = d[0].decode('utf-8')
                except UnicodeDecodeError:
                    _read_fail = True
                    break

                data_id = data_id.replace('$', '')
                if data_id != definition.identifier:
                    _read_fail = True
                    break

            if _read_fail:
                del _data
                _data = self.scan(_file, definition)

            _time = _data[self.time_variable]

            # If there isn't any time info, then get out of here before we
            # raise an exception.
            if not len(_time):
                continue

            # Small amount of error tolerence. If there's a single dodgy
            # timestamp in between two otherwise OK timestamps, assume that
            # it's OK to interpolate across it
            _time = pd.Series(_time)
            _time.loc[(_time - _time.median()).abs() > C_BAD_TIME_DEV] = np.nan
            _time = _time.interpolate(limit=1).values

            _good_times = np.where(~np.isnan(_time))
            _time = _time[_good_times]

            for _name, _dtype in _data.dtype.fields.items():

                if _name[0] == '$':
                    continue
                if _name == self.time_variable:
                    continue

                # Pandas doesn't enjoy non-native endianess, so convert data
                # to system byteorder if required
                if definition.get_field(_name).byte_order != sys.byteorder:
                    _var = _data[_name].byteswap().newbyteorder()
                else:
                    _var = _data[_name]

                if len(_var.shape) == 1:
                    _var = _var[_good_times]
                else:
                    _var = _var[_good_times, :]

                frequency, index = self._get_index(
                    _var, _name, _time, definition
                )

                # Define the decades variable
                dtd = self._get_group_name(definition)

                variable_name = '{}_{}'.format(dtd,  _name)

                max_var_len = len(self._index_dict[frequency])
                if max_var_len != len(_var.ravel()):
                    print('WARN: index & variable len differ')
                    print('      ({})'.format(variable_name))

                _var = _var.ravel()[:max_var_len]

                variable = DecadesVariable(
                    {variable_name: _var.ravel()},
                    index=self._index_dict[frequency],
                    name=variable_name,
                    long_name=definition.get_field(_name).long_name,
                    units='RAW',
                    frequency=frequency,
                    write=False
                )

                if variable.index.size != variable.index.unique().size:
                    print('Non-unique index: {}'.format(variable.name))
                    print(f'{variable.name}: using last of non-unique entries')

                    variable._df = variable._df.groupby(variable._df.index).last()
                    variable.flag._df = variable.flag._df.groupby(
                        variable.flag._df.index
                    ).last()

                _file.dataset.add_input(variable)


class CrioFileReader(TcpFileReader):
    pass


@register(patterns=['GINDAT.+\.bin'])
class GinFileReader(TcpFileReader):
    time_variable = 'time1'
    frequency = 50

    def _get_definition(self, _file):
        for _definition in _file.dataset.definitions:
            if _definition.identifier == 'GRP':
                return _definition

    def _get_group_name(self, *args):
        return 'GINDAT'

    def _scan_read_packet_len(self, definition, offset, rawdata):
        packet_dlen = definition.dtypes['packet_length'].itemsize
        id_length = definition.dtypes[0].itemsize

        packet_length = np.frombuffer(
            rawdata[offset+id_length+2:offset+id_length+2+packet_dlen],
            dtype=definition.dtypes['packet_length']
        )[0]

        return packet_length

    def _time_last_saturday(self):
        return self.dataset.date - relativedelta.relativedelta(
            weekday=relativedelta.SU(-1)
        )

    def _get_index(self, var, name, time, definition):
        try:
            return self.frequency, self._index_dict[self.frequency]
        except KeyError:
            pass

        index = pd.DatetimeIndex(np.array(time) * 1e9)
        index += pd.DateOffset(
            seconds=(self._time_last_saturday() - datetime.datetime(
                1970, 1, 1)).total_seconds()
        )
        self._index_dict[self.frequency] = index
        return self.frequency, index


class DefinitionReader(FileReader):
    level = 1

    def _get_datatype(self, size, num_points, typestr):
        type_dict = {
            'unsigned_int': 'u', 'int': 'i', 'signed_int': 'i', 'double': 'f',
            'single_float': 'f', 'float': 'f', 'single': 'f',
            'double_float': 'f', 'boolean': 'u', 'f': 'f', 'i': 'i', 'u': 'u',
            'text': 'S'
        }

        endian = '>'
        if '<' in typestr:
            endian = '<'
        typestr = typestr.replace('>', '').replace('<', '')

        typestr = '{endian}{num_points}{kind}{size}'.format(
            endian=endian,
            kind=type_dict[typestr],
            size=size,
            num_points=num_points
        )

        return typestr

@register(patterns=['.+_TCP_?.*\.csv'])
class CrioDefinitionReader(DefinitionReader):

    def read(self):
        for _file in self.files:
            print('Reading {}'.format(_file))
            tcp_def = CrioTcpDefintion()
            is_header = True
            with open(_file.filepath, 'r') as _csv:
                reader = csv.reader(_csv)
                for row in reader:

                    # Ignore empty lines
                    if not row:
                        continue

                    # Ignore tyhe header
                    if row[0] == 'field':
                        continue

                    # Exctract the crio identifier
                    if row[0][0] == '$':
                        tcp_def.identifier = row[0].replace('$', '')

                    if is_header:
                        tcp_def.header_length += int(row[1])
                    else:
                        tcp_def.body_length += int(row[1])

                    if row[0] == 'packet_length':
                        is_header = False

                    _num_bytes = int(row[1])
                    _bytes_per_point = int(row[2])
                    _num_points = int(_num_bytes / _bytes_per_point)
                    _type = row[3]

                    tcp_def.fields.append(
                        DataField(
                            datatype=self._get_datatype(
                                _bytes_per_point, _num_points, _type
                            ),
                            short_name=row[0],
                            long_name=row[4]
                        )
                    )

            _file.dataset.add_definition(tcp_def)


class CrioTcpDefintion(object):
    """
    A CrioTcpDefinition contains the variable names and associated datatypes
    and metadata required to read a crio data file. This is essentially a
    parsed version of the crio .csv definition file.
    """

    def __init__(self):
        self.header_length = 0
        self.body_length = 0
        self.identifier = None
        self.fields = []

    @property
    def packet_length(self):
        return self.header_length + self.body_length

    @property
    def dtypes(self):
        return np.dtype([(i.short_name, i.datatype) for i in self.fields])

    def get_field(self, name):
        for f in self.fields:
            if f.short_name == name:
                return f


class DataField(object):
    def __init__(self, datatype=None, short_name=None, long_name=None):
        self.datatype = datatype
        self.short_name = short_name
        self.long_name = long_name

        if '>' in datatype:
            self.byte_order = 'big'
        elif '<' in datatype:
            self.byte_order = 'little'
        else:
            self.byte_order = None

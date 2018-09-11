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


from ppodd.decades import DecadesVariable


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

    def read(self):
        for _file in self.files:
            pass


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

        index = np.array([False] * len(rawdata))

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
        _ser_str = '{}N'.format(1 / frequency * 10**9)
        return pd.Series(index=index).asfreq(_ser_str).index[:-1]

    def _get_index_slow(self, time, frequency):
        """
        Returns an interpolated (subsecond) index using a slow,
        but (hopefully) robust method.

        args:
            _data: a named np.ndarray, assumed to contain
                'utc_time'
            frequency: the frequency to interpolate to (s).

        returns:
            a pandas.DatetimeIndex
        """

        _ser_str = '{}N'.format(1 / frequency * 10**9)
        index = pd.to_datetime(time, unit='s')
        dti = None
        for i in index:
            _dti = pd.DatetimeIndex(start=i, freq=_ser_str,
                                    periods=frequency)

            if dti is None:
                dti = _dti
            else:
                dti = dti.append(_dti)

        return dti

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
            print('Reading {}'.format(_file))
            self._index_dict = {}

            definition = self._get_definition(_file)

            if definition is None:
                warnings.warn(
                    'No CRIO definition found for {}'.format(_file.filepath),
                    RuntimeWarning
                )
                continue

            dtypes = definition.dtypes

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
                _data = self.scan(_file, definition)

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

                frequency, index = self._get_index(
                    _var, _name, _data[self.time_variable], definition
                )

                # Define the decades variable
                dtd = self._get_group_name(definition)

                variable_name = '{}_{}'.format(dtd,  _name)

                variable = DecadesVariable(
                    {variable_name: _var.ravel()},
                    index=self._index_dict[frequency],
                    name=variable_name,
                    long_name=definition.get_field(_name).long_name,
                    units='RAW',
                    frequency=frequency
                )

                _file.dataset.add_input(variable)


class CrioFileReader(TcpFileReader):
    pass


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
        index = pd.DatetimeIndex(
            [self._time_last_saturday() + datetime.timedelta(seconds=i) for i in time]
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


class JsonConstantsReader(FileReader):

    def read(self):
        for _file in self.files:
            with open(_file.filepath, 'r') as _consts:
                consts = json.loads(_consts.read())

            for _module in consts['MODULES']:
                for key, value in consts['MODULES'][_module].items():
                    _file.dataset.add_constant(key, value)


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

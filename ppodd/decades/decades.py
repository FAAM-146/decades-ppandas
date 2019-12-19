import collections
import datetime
import gc
import importlib
import sys
import re
import os

import numpy as np
import pandas as pd

from .backends import PandasInMemoryBackend
from .flags import DecadesClassicFlag


class DecadesFile(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def __repr__(self):
        return '{}({!r})'.format(
            self.__class__.__name__, self.filepath
        )


class DecadesVariable(object):

    NC_ATTRS = [
        'long_name', 'frequency', 'standard_name', 'units',
        '_FillValue', 'valid_min', 'valid_max', 'comment'
    ]

    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)
        _flag = kwargs.pop('flag', DecadesClassicFlag)

        self.attrs = {
            '_FillValue': -9999.,
        }

        write = kwargs.pop('write', True)

        for _attr in DecadesVariable.NC_ATTRS:
            _val = kwargs.pop(_attr, None)
            if _val is not None:
                self.attrs[_attr] = _val

        self._df = pd.DataFrame(*args, **kwargs)

        # We only want to be given at most two columns (variable, flag) and at
        # least one column (variable). Raise an error if this is not the case.
        _columns = [i for i in self._df]
        if len(_columns) != 1:
            raise ValueError('Too many columns in instance DataFrame')


        # Deal with variable/instance naming. If no 'name' keyword is
        # specified, then insist that the variable and flag are consistently
        # named. Otherwise, rename both to be consistent with the kwarg.
        _var = _columns[0]
        if name is None:
            self.name = _var
        else:
            self.name = name
            _rename = {_var: name}
            self._df.rename(columns=_rename, inplace=True)

        self._write = write

        self.attrs['ancillary_variables'] = '{}_FLAG'.format(self.name)
        self.flag = _flag(self)

    def __len__(self):
        return len(self._df)

    def __str__(self):
        return 'DecadesVariable: {}'.format(self.name)

    def __repr__(self):
        return '<DecadesVariable({!r})>'.format(self.name)

    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except KeyError:
            pass

        try:
            return self.attrs[attr]
        except KeyError:
            pass

        if attr == 'data':
            return self._df[self.name]

        if attr == 'flag':
            return self.flag

        return getattr(self._df, attr)

    def __setattr__(self, attr, value):
        if attr in DecadesVariable.NC_ATTRS:
            self.attrs[attr] = value
        else:
            super().__setattr__(attr, value)

    def time_bounds(self):
        return (self.index[0], self.index[-1])

    @property
    def write(self):
        return self._write

    @write.setter
    def write(self, write):
        self._write = write


class DecadesDataset(object):
    def __init__(self, date=None, backend=PandasInMemoryBackend):

        self._date = date
        self.readers = []
        self.definitions = []
        self.constants = {}
        self._variable_mods = {}
        self._mod_exclusions = []
        self._globals = {}
        self.inputs = []
        self.outputs = []
        self._dataframes = {}
        self._garbage_collect = False
        self._qa_dir = None
        self._takeoff_time = None
        self._backend = backend()

        import ppodd.pod
        import ppodd.qa

        self.qa_modules = [qa(self) for qa in ppodd.qa.qa_modules]
        self.pp_modules = collections.deque(
            [pp(self) for pp in ppodd.pod.pp_modules]
        )

    def __getitem__(self, item):

        try:
            return self._backend[item]
        except KeyError:
            pass

        for _var in self.outputs:
            if _var.name == item:
                return _var

        try:
            return self.constants[item]
        except KeyError:
            pass

        raise KeyError('Unknown variable: {}'.format(item))

    def time_bounds(self):
        """
        Return the time period covered by this dataset.

        Returns:
            a 2-tuple containing the smallest and largest times of variables in
            this dataset.
        """
        start_time = datetime.datetime.max
        end_time = datetime.datetime.min

        for var in self.outputs:

            if not var.write:
                continue

            if var.data.index[0] < start_time:
                start_time = var.data.index[0]

            if var.data.index[-1] > end_time:
                end_time = var.data.index[-1]

        start_time = start_time.replace(microsecond=0)

        return (start_time, end_time)


    def garbage_collect(self, collect):
        """
        Turn garbage collection on or off. If on, variables which are not
        required for processing are removed from the dataset.
        """
        if collect:
            self._garbage_collect = True
            return
        self._garbage_collect = False

    @property
    def date(self):
        if self._date:
            return datetime.datetime.combine(
                self._date, datetime.datetime.min.time()
            )

        try:
            return datetime.datetime.combine(
                self._globals['date'], datetime.datetime.min.time()
            )
        except KeyError:
            pass

        raise AttributeError('No date has been set')


    @property
    def _dynamic_globals(self):
        _globals = {}
        try:
            _globals['geospatial_lat_min'] = self['LAT_GIN'].data.min()
            _globals['geospatial_lat_max'] = self['LAT_GIN'].data.max()
        except KeyError:
            pass

        try:
            _globals['geospatial_lon_min'] = self['LON_GIN'].data.min()
            _globals['geospatial_lon_max'] = self['LON_GIN'].data.max()
        except KeyError:
            pass

        try:
            _globals['geospatial_vertical_min'] = self['ALT_GIN'].data.min()
            _globals['geospatial_vertical_max'] = self['ALT_GIN'].data.max()
            _globals['geospatial_vertical_positive'] = 'up'
        except KeyError:
            pass

        _time_bnds = self.time_bounds()
        _strf_pattern = '%Y-%m-%dT%H:%M:%SZ'

        _globals['time_coverage_start'] = _time_bnds[0].strftime(_strf_pattern)
        _globals['time_coverage_end'] = _time_bnds[-1].strftime(_strf_pattern)

        return _globals

    @property
    def globals(self):
        """
        Returns:
            A dict containing dataset globals. This is a combination of
            the self._globals dict and globals generated on the fly from the
            contents of the dateset, via self._dynamic_globals
        """

        _globals = {}
        _globals.update(self._globals)
        _globals.update(self._dynamic_globals)
        return _globals

    def add_global(self, key, value):
        """
        Add a global key/value pair to the dataset globals.

        Args:
            key: the name of the global attribute to add
            value: the value of the global attribute <key>
        """
        self._globals[key] = value

    @property
    def qa_dir(self):
        return self._qa_dir

    @qa_dir.setter
    def qa_dir(self, qa_dir):
        if not os.path.isdir(qa_dir) and os.path.exists(qa_dir):
            raise OSError('{} exists and is not a directory'.format(qa_dir))

        elif not os.path.exists(qa_dir):
            os.makedirs(qa_dir)

        self._qa_dir = qa_dir

    @staticmethod
    def infer_reader(dfile):
        from ppodd.readers import reader_patterns
        _filename = os.path.basename(dfile.filepath)

        for pattern in reader_patterns:
            if re.fullmatch(pattern, _filename):
                return reader_patterns[pattern]

        print('No reader found for {}'.format(dfile))

    def add_definition(self, definition):
        self.definitions.append(definition)

    def add_constant(self, name, data):
        self.constants[name] = data

    def add_input(self, variable):
        """
        Add a new DecadesVariable to this DecadesDataset. If the variable
        already exists, as identified by name, then append to that variable
        using pd.Series.combine_first().

        args:
            variable: the DecadesVariable to add to this DecadesDataset.
        """

        self._backend.add_input(variable)

    @property
    def variables(self):
        return self._backend.variables + [i.name for i in self.outputs]

    @property
    def files(self):
        """
        Return a list of all files currently accociated with this
        DecadesDataset.
        """
        _files = []
        for reader in self.readers:
            _files += reader.files
        return _files

    @files.setter
    def files(self, value):
        """
        Explicitally forbid <DecadesDataset>.files = Something type
        assignments.
        """
        raise ValueError(('Cannot add directly to files. Use add_file() '
                         'or add_decades_file()'))

    def get_reader(self, cls):
        """
        Get the FileReader of type cls associated with this Dataset. If no such
        reader exists, raise a ValueError.

        args:
            cls: The class (type) of the reader we wish to retrieve.

        returns:
            The reader of type cls, if such a reader exists.
        """
        for reader in self.readers:
            if reader.__class__ == cls:
                return reader
        raise ValueError

    def add_decades_file(self, dfile):
        """
        Add a DecadesFile to the DecadesDataset. Each file must be associated
        with a reader. If a suitable reader is already associated with the
        Dataset, then add the file to that reader. If not, create a new reader,
        add that to the dataset, and then add the file to the reader.

        args:
            dfile:
                The DecadesFile to add to the dataset.
        """

        dfile.dataset = self

        reader_class = DecadesDataset.infer_reader(dfile)

        reader = None

        try:
            reader = self.get_reader(reader_class)
        except ValueError:
            pass

        if reader is None:
            try:
                reader = reader_class()
            except TypeError:
                reader = None

        if reader is None:
            raise ValueError

        if reader not in self.readers:
            self.readers.append(reader)
            self.readers.sort(key=lambda r: r.level)

        reader.files.append(dfile)

    def add_file(self, filename):
        """
        Add a file to the dataset, first creating a DecadesFile.

        args:
            filename: the path to the file to add.

        kwargs:
            file_type: the file type to pass to DecadesFile
        """
        print('adding {}'.format(filename))
        try:
            self.add_decades_file(DecadesFile(filename))
        except ValueError:
            print('failed to add {}'.format(filename))

    def load(self):
        """
        Load all of the data from files associated with readers in this
        dataset.
        """

        for reader in self.readers:
            reader.read()
            self._collect_garbage()

        self.readers = None
        gc.collect()

    @property
    def takeoff_time(self):
        if self._takeoff_time is not None:
            return self._takeoff_time

        if 'PRTAFT' not in self._dataframes:
            return None

        series = self._dataframes['PRTAFT'][1]['PRTAFT_wow_flag']

        self._takeoff_time =  series.diff().where(
            series.diff()==-1
        ).dropna().tail(1).index

        return self._takeoff_time

    def _get_required_data(self):
        _required_inputs = []
        for mod in self.pp_modules:
            _required_inputs += mod.inputs

        for qa in self.qa_modules:
            _required_inputs += qa.inputs

        _required_inputs = list(set(_required_inputs))

        return _required_inputs

    def _collect_garbage(self):
        if not self._garbage_collect:
            return

        required_inputs = self._get_required_data()

        self._backend.collect_garbage(required_inputs)

        gc.collect()

    def run_qa(self):

        while self.qa_modules:
            _mod = self.qa_modules.pop()
            try:
                _mod.run()
            except Exception as e:
                print(' ** Error in {}: {}'.format(_mod, e))

    def _trim_data(self):
        if self.takeoff_time is not None:
            CUTOFF = self.takeoff_time[0] - datetime.timedelta(hours=4)
            for key, value in self._dataframes.items():
                dlu = self._dataframes[key]
                for key, value in dlu.items():
                    df = dlu[key]

                    print('dropping')
                    df.drop(df.index[df.index < CUTOFF], inplace=True)

    def process(self, modname=None):
        """
        Run processing modules.
        """
        import ppodd.pod
        import ppodd.qa

        self._trim_data()

        if modname is not None:
            mods = ppodd.pod.pp_modules
            for mod in mods:
                if mod.__name__ is modname:
                    _mod = mod(self)
                    _mod_ready, _missing = _mod.ready()
                    if _mod_ready:
                        _mod.process()
                        _mod.finalize()
                    else:
                        print(
                            ('Module {} not ready: '
                             'inputs not available {}').format(
                                 mod.__name__, ','.join(_missing))
                        )
            return

        self.qa_modules = [qa(self) for qa in ppodd.qa.qa_modules]

        self.outputs = []

        self.pp_modules = collections.deque(
            [pp(self) for pp in ppodd.pod.pp_modules]
        )
        self.completed_modules = []
        self.failed_modules = []

        temp_modules = []

        module_ran = True
        while self.pp_modules and module_ran:

            module_ran = False

            pp_module = self.pp_modules.popleft()

            _mod_ready, _missing = pp_module.ready()
            if not _mod_ready:
                print('{} not ready (missing {})'.format(
                    pp_module, ', '.join(_missing)
                ))
                temp_modules.append(pp_module)
                module_ran = True
                continue
            if str(pp_module) in self._mod_exclusions:
                print('Skipping {} (excluded)'.format(pp_module))
                module_ran = True
                continue
            try:
                print('Running {}'.format(pp_module))
                pp_module.process()
                pp_module.finalize()
            except Exception as e:
                import traceback
                print(' ** Error in {}: {}'.format(pp_module, e))
                traceback.print_exc()
                self.failed_modules.append(pp_module)
            else:
                self.completed_modules.append(pp_module)

            module_ran = True
            while temp_modules:
                self.pp_modules.append(temp_modules.pop())

            self._collect_garbage()

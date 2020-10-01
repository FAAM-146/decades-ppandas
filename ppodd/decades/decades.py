import collections
import datetime
import gc
import importlib
import sys
import re
import os

from pydoc import locate

import numpy as np
import pandas as pd

import ppodd

from .backends import PandasInMemoryBackend
from .globals import GlobalsCollection
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
        '_FillValue', 'valid_min', 'valid_max', 'comment', 'sensor_type',
        'sensor_serial', 'instrument_serial'
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
        self.pp_modules = []
        self.qa_modules = []
        self.globals = GlobalsCollection(dataset=self)
        self._dataframes = {}
        self.lon = None
        self.lat = None
        self._garbage_collect = False
        self._qa_dir = None
        self._takeoff_time = None
        self._landing_time = None
        self._decache = False
        self._trim = False
        self.allow_overwrite = False
        self._backend = backend()

    def __getitem__(self, item):

        try:
            return self._backend[item]
        except KeyError:
            pass

        try:
            return self.constants[item]
        except KeyError:
            pass

        if item == 'TIME_MIN_CALL':
            return lambda: self.time_bounds()[0].strftime('%Y-%m-%dT%H:%M:%SZ')

        if item == 'TIME_MAX_CALL':
            return lambda: self.time_bounds()[1].strftime('%Y-%m-%dT%H:%M:%SZ')

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

        for _var in self.variables:
            var = self[_var]

            if not var.write:
                continue

            if var.data.index[0] < start_time:
                start_time = var.data.index[0]

            if var.data.index[-1] > end_time:
                end_time = var.data.index[-1]

            self._backend.decache()

        start_time = start_time.replace(microsecond=0)

        return (start_time, end_time)

    def remove(self, name):
        self._backend.remove(name)

    def garbage_collect(self, collect):
        """
        Turn garbage collection on or off. If on, variables which are not
        required for processing are removed from the dataset.
        """
        if collect:
            self._garbage_collect = True
            return
        self._garbage_collect = False

    def decache(self, dc):
        if dc:
            self._decache = True
            return
        self._decache = False

    @property
    def outputs(self):
        return self._backend.outputs

    def clear_outputs(self):
        self._backend.outputs = []

    @property
    def trim(self):
        return self._trim

    @trim.setter
    def trim(self, trm):
        self._trim = trm

    @property
    def date(self):
        if self._date:
            return datetime.datetime.combine(
                self._date, datetime.datetime.min.time()
            )

        try:
            return datetime.datetime.combine(
                self.globals['date'], datetime.datetime.min.time()
            )
        except KeyError:
            pass

        raise AttributeError('No date has been set')

    def _interpolate_globals(self):
        """
        Interpolate across global attributes, allowing placeholders to be used
        in globals in the flight constants file.
        """
        for key, value in self.globals.static_items():
            if isinstance(value, str):
                self.globals[key] = self.globals[key].format(**self.globals())

    def add_global(self, key, value):
        """
        Add a global key/value pair to the dataset globals. The value can be
        either a literal or a directive. Directives are strings of the form
        <action args> where action is one of [call, data]. Call should have a
        single argument identifying an object in the python path, which should
        be either a serializable literal or a callable which resolves to one.
        Data can have any number of (whitespace separated) arguments. The first
        should be a variable name, and the following attributes to successively
        call on that data. The final attribute may be a serializable literal,
        or a callable which resolves to one.

        Dictionaries may be passed as values; these will be recursively
        flattened.

        Args:
            key: the name of the global attribute to add
            value: the value of the global attribute <key>
        """

        # Recursively flatten any dictionaries passed as values
        if type(value) is dict:
            for _key, _val in value.items():
                _key = '{}_{}'.format(key, _key)
                self.add_global(_key, _val)
            return

        # See if the value passed is a directive
        try:
            rex = re.compile('<(?P<action>[a-z]+) (?P<value>.+)>')
            result = rex.search(value)
        except TypeError:
            result = None

        if result:
            # The value is a directive; resolve it
            groups = result.groupdict()

            if groups['action'] == 'call':
                # Attempt to locate the value of the call directive. Abandon
                # this global if its not found
                value = locate(groups['value'])
                if value is None:
                    return

            if groups['action'] == 'data':
                # Data directive: parse the string and delegate to
                # add_data_global
                values = [v.strip() for v in groups['value'].split()]
                iter_vals = (values[0], values[1:])
                self.globals.add_data_global(key, iter_vals)
                return

        # Add the global
        self.globals[key] = value

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

        self[variable.name].flag._df = self[variable.name].flag._df.reindex(
            self[variable.name].index
        ).fillna(method='ffill').fillna(method='bfill')

    def add_output(self, variable):
        self._backend.add_output(variable)

    @property
    def variables(self):
        return self._backend.variables

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
        import ppodd.pod
        import ppodd.qa
        import ppodd.flags

        for reader in self.readers:
            try:
                reader.read()
            except Exception as e:
                print(f'Error in reading module {reader}')
                print(str(e))
            del reader

        self.readers = None
        gc.collect()

        self.qa_modules = [qa(self) for qa in ppodd.qa.qa_modules]

        # Initialise postprocessing modules
        _pp_modules = []
        for pp in ppodd.pod.pp_modules:
            try:
                _pp_modules.append(pp(self))
            except Exception as e:
                print('Couldn\'t init {}: {}'.format(pp, str(e)))
        self.pp_modules = collections.deque(_pp_modules)

        self.flag_modules = [flag(self) for flag in ppodd.flags.flag_modules]

        self._interpolate_globals()

        self._collect_garbage()

    @property
    def takeoff_time(self):
        """
        Return the latest takeoff time of the data set, as determined by the
        last time at which PRTAFT_wow_flag changing 1 -> 0
        """

        if self._takeoff_time is not None:
            return self._takeoff_time

        try:
            wow = self['PRTAFT_wow_flag']
        except KeyError:
            return None

        try:
            series = wow.data.astype(np.int8)
        except ValueError:
            return None

        try:
            self._takeoff_time = series.diff().where(
                series.diff() == -1
            ).dropna().tail(1).index[0]
        except IndexError:
            return None

        return self._takeoff_time

    @property
    def landing_time(self):
        """
        Return the latest landing time of the dataset, as determined by the
        last time at which PRTAFT_wow_flag changes from 0 -> 1
        """

        if self._landing_time is not None:
            return self._landing_time

        try:
            wow = self['PRTAFT_wow_flag']
        except KeyError:
            return None

        try:
            series = wow.data.astype(np.int8)
        except ValueError:
            return None

        try:
            self._landing_time = series.diff().where(
                series.diff() == 1
            ).dropna().tail(1).index[0]
        except IndexError:
            return None

        return self._landing_time

    def _get_required_data(self):
        _required_inputs = []
        for mod in self.pp_modules:
            _required_inputs += mod.inputs

        for qa in self.qa_modules:
            _required_inputs += qa.inputs

        for var in self.variables:
            if self[var].write:
                _required_inputs.append(var)

        for var in self._variable_mods:
            try:
                if self._variable_mods[var]['write']:
                    _required_inputs.append(var)
            except KeyError:
                # Likely an output modifier
                pass

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
                del _mod
            except Exception as e:
                print(' ** Error in {}: {}'.format(_mod, e))

    def run_flagging(self):

        while self.flag_modules:
            _flag = self.flag_modules.pop()
            try:
                print('running {}'.format(_flag))
                _flag.flag()
                del _flag
            except Exception as e:
                print(' ** Error in {}: {}'.format(_flag, e))

    def _trim_data(self):
        if self.takeoff_time and self.landing_time:
            start_cutoff = self.takeoff_time - datetime.timedelta(hours=2)
            end_cutoff = self.landing_time + datetime.timedelta(minutes=30)
            self._backend.trim(start_cutoff, end_cutoff)


    def process(self, modname=None):
        """
        Run processing modules.
        """
        import ppodd.pod
        import ppodd.qa
        import ppodd.flags

        if self.trim:
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

        # Initialise postprocessing modules
        _pp_modules = []
        for pp in ppodd.pod.pp_modules:
            try:
                _pp_modules.append(pp(self))
            except Exception as e:
                print('Couldn\'t init {}: {}'.format(pp, str(e)))
        self.pp_modules = collections.deque(_pp_modules)

        self.flag_modules = [flag(self) for flag in ppodd.flags.flag_modules]

        self._backend.clear_outputs()

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
                del pp_module
                continue
            if str(pp_module) in self._mod_exclusions:
                print('Skipping {} (excluded)'.format(pp_module))
                module_ran = True
                del pp_module
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
            self._backend.decache()

        self.run_flagging()

        # Modify any attributes on inputs, canonically specified in flight
        # constants file.
        for var in self._backend.inputs:
            name = var.name
            if name in self._variable_mods:
                for key, value in self._variable_mods[name].items():
                    setattr(var, key, value)

    def cleanup(self):
        self._backend.cleanup()

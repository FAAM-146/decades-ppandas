import collections
import datetime
import gc
import glob
import importlib
import sys
import re
import os

from pydoc import locate

import numpy as np
import pandas as pd
from scipy.stats import mode

import ppodd

from .backends import DefaultBackend
from .attributes import AttributesCollection, Attribute
from .flags import DecadesClassicFlag
from ..standard import faam_globals, faam_attrs
from ..utils import pd_freq, infer_freq


class DecadesFile(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def __repr__(self):
        return '{}({!r})'.format(
            self.__class__.__name__, self.filepath
        )


class DecadesVariable(object):

    def __init__(self, *args, **kwargs):
        _flag = kwargs.pop('flag', DecadesClassicFlag)

        self.attrs = AttributesCollection(
            dataset=self, definition=faam_attrs['VariableAttrs'], version=1.0
        )

        self.name = kwargs.pop('name', None)
        self.write = kwargs.pop('write', True)

        _attrs = self.attrs.REQUIRED_ATTRIBUTES + self.attrs.OPTIONAL_ATTRIBUTES
        for _attr in _attrs:
            try:
                _default = self.attrs._definition[_attr]['default']
            except KeyError:
                _default = None
            _val = kwargs.pop(_attr, _default)
            if _val is not None:
                self.attrs.add(Attribute(_attr, _val))

        _df = pd.DataFrame(*args, **kwargs)
        _freq = self._get_freq(df=_df)

        _index = pd.date_range(
            start=_df.index[0], end=_df.index[-1],
            freq=_freq
        )

        if self.name is None:
            self.name = _df.columns[0]

        if len(_df.index) != len(_df.index.unique()):
            _df = _df.groupby(_df.index).last()

        self.array = self._downcast(np.array(
            _df.reindex(
                _index, tolerance=_freq, method='nearest', limit=1
            ).values.flatten()
        ))

        self.t0 = _index[0]
        self.t1 = _index[-1]
        self.flag = _flag(self)
        self.attrs.add(Attribute('ancillary_variables', f'{self.name}_FLAG'))

    def __call__(self):
        i = pd.date_range(
            start=self.t0, end=self.t1,
            freq=self._get_freq()
        )
        return pd.Series(
            self.array, index=i, name=self.name
        )

    def __len__(self):
        return len(self.array)

    def __getattribute__(self, attr):
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            pass

        if attr == 'attrs':
            try:
                return self.__dict__['attrs']
            except KeyError:
                pass

        if attr == 'index':
            return self().index

        if attr == 'data':
            return self()

        if attr == 'flag':
            return self.flag

        if attr in self.attrs.keys:
            return self.attrs[attr]

        try:
            # Dirty check that we have t0 and t1, or self() will recurse with
            # __getattribute__()
            self.__dict__['t0']
            self.__dict__['t1']
            return getattr(self(), attr)
        except (KeyError, AttributeError):
            pass

        raise AttributeError(f'Not a variable attribute: {attr}')

    def __setattr__(self, attr, value):
        if attr == 'attrs':
            super().__setattr__(attr, value)

        if attr in (
            self.attrs.REQUIRED_ATTRIBUTES + self.attrs.OPTIONAL_ATTRIBUTES
        ):
            self.attrs[attr] = value

        super().__setattr__(attr, value)

    def __str__(self):
        return self.name

    def __repr__(self):
        return r'<DecadesVariable[{!r}]>'.format(self.name)

    def _get_freq(self, df=None):
        try:
            return pd_freq[self.attrs['frequency']]
        except (KeyError, AttributeError):
            _freq = pd.infer_freq(df.index)

        if _freq is None:
            _freq = infer_freq(df.index)

        if len(_freq) == 1:
            _freq = f'1{_freq}'

        self.frequency = int(1/pd.to_timedelta(_freq).total_seconds())
        return _freq

    def _downcast(self, array):
        """
        Downcast a numeric array to its smallest compatable type, via
        pd.to_numeric.

        Args:
            array: the numpy array, or pd.Series to downcast.

        Returns:
            a downcast copy of array, or array if it cannot be downcast.
        """
        dc = 'float'
        try:
            if np.all(array == array.astype(int)):
                dc = 'integer'
        except (ValueError, TypeError):
            pass

        try:
            return pd.to_numeric(array, downcast=dc)
        except (ValueError, TypeError):
            pass

        return array

    def _merge_fast(self, other):
        _data = np.concatenate([self.array, other.array])
        _index = self.index.union(other.index)

        _df = pd.DataFrame(_data, index=_index).reindex(
            pd.date_range(start=_index[0], end=_index[-1],
                          freq=pd_freq[self.frequency])
        )

        self.array = _df.values.flatten()
        self.t0 = _df.index[0]
        self.t1 = _df.index[-1]

    def trim(self, start, end):
        self.flag.trim(start, end)

        _df = self()
        loc = (_df.index >= start) & (_df.index <= end)
        trimmed = _df.loc[loc]
        self.array = trimmed.values.flatten()
        self.t0 = trimmed.index[0]
        self.t1 = trimmed.index[-1]

    def merge(self, other):

        if(other.t0 > self.t1):
            self._merge_fast(other)
            return

        other = other()
        current = self()

        merge_index = current.index.union(other.index).sort_values().unique()
        current = current.reindex(merge_index)
        current.loc[other.index] = other
        full_index = pd.date_range(
            start=merge_index[0], end=merge_index[-1],
            freq=pd_freq[self.frequency]
        )
        current = current.reindex(full_index)

        self.array = current.values.flatten()
        self.t0 = current.index[0]
        self.t1 = current.index[-1]

    def time_bounds(self):
        return (self.t0, self.t1)


class DecadesDataset(object):
    def __init__(self, date=None, version=1.0, backend=DefaultBackend):

        self._date = date
        self.readers = []
        self.definitions = []
        self.constants = {}
        self._variable_mods = {}
        self._mod_exclusions = []
        self.pp_modules = []
        self.qa_modules = []

        self.globals = AttributesCollection(
            dataset=self, definition=faam_globals['Globals'], version=version
        )

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

            if var.t0 < start_time:
                start_time = var.t0

            if var.t1 > end_time:
                end_time = var.t1

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
                self.globals.add_data_attribute(key, iter_vals)
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

    def add_glob(self, pattern):
        """
        Add files to the dataset which match a given globbing pattern.

        Args:
            pattern: the glob pattern to match.
        """
        for f in glob.glob(pattern):
            self.add_file(f)

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
            wow = self['PRTAFT_wow_flag']()
        except KeyError:
            return None

        try:
            self._takeoff_time = wow.diff().where(
                wow.diff() == -1
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
            wow = self['PRTAFT_wow_flag']()
        except KeyError:
            return None

        try:
            self._landing_time = wow.diff().where(
                wow.diff() == 1
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
            try:
                if self[var].write:
                    _required_inputs.append(var)
            except KeyError:
                continue

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
            print(f'trimming: {start_cutoff} -- {end_cutoff}')
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

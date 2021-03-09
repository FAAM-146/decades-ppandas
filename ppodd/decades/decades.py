"""
This module contains top-level classes for data processing jobs. A minimal
data processing workflow may look something like:

from ppodd.decades import DecadesDataset
d = DecadesDataset() # using defaults for everything
d.add_glob('*zip')   # assuming all data are in a zip file
d.add_glob('*yaml')  # adding a constants file
d.load()             # load all data
d.process()          # run the processing
d.write()            # write the data to a default filename
"""
#pylint: disable=invalid-name, useless-object-inheritance,
#pylint: disable=too-many-instance-attributes, too-many-lines
import collections
import datetime
import gc
import glob
import importlib
import re
import os
import traceback

from pydoc import locate

import numpy as np
import pandas as pd

import ppodd.flags

from ppodd.decades.backends import DefaultBackend
from ppodd.decades.attributes import AttributesCollection, Attribute
from ppodd.decades.flags import DecadesClassicFlag
from ppodd.utils import pd_freq, infer_freq
from ppodd.writers import NetCDFWriter


class Lazy(object):
    """
    A hacky deferral wrapper for assigning dataset constants to processing
    module outputs potentially before they've actually been set on the dataset.
    This works in this context as variable metadata are allowed to be either
    literal or callable.
    """

    def __init__(self, parent):
        """
        Initialize a class instance.

        Args:
            parent: the object whos state we want to defer.
        """
        self.parent = parent

    def __getitem__(self, item):
        """
        Implement [x], potentially deferred to a callable.

        Args:
            item: the item to get from the parent.

        Returns:
            either <item> got from parent, or a callable deferring this
            operation.
        """
        try:
            return self.parent[item]
        except KeyError:
            return lambda: self.parent[item]

    def __getattr__(self, attr):
        """
        Implement .x, potentially deferred to a callable.

        Args:
            attr: the attribute to get from the parent.

        Returns:
            either <attr> from parent, or a callable deferring this
            operation.
        """
        try:
            return getattr(self.parent, attr)
        except AttributeError:
            return lambda: getattr(self.parent, attr)


class DecadesFile(object):
    """
    A DecadesFile is just a wrapper around a filepath. Factored out for
    potential future use, but currently isn't really doing anything useful.
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, filepath):
        """
        Initialize an instance.

        Args:
            filepath: a string giving the absolute path to a file.
        """
        self.filepath = filepath

    def __repr__(self):
        return '{}({!r})'.format(
            self.__class__.__name__, self.filepath
        )


class DecadesVariable(object):
    """
    A DecadesVariable is a container for a timeseries, a corresponding data
    quality flag and associated Metadata, in the form of an AttributeCollection
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a class instance. Arbitrary args and kwargs are accepted.
        These will be passed to pandas during the initial creation of a
        DataFrame, other than keys which are defined in the <standard> and:

        Kwargs:
            name: the name of the variable
            write: whether or not this variable should be written to file as an
                   output. Default True.
            flag: a class to be used for holding data quality information.
                  Default is DecadesClassicFlag.
            standard: a metadata 'standard' which should be adhered to. Default
                      is 'ppodd.standard.core'.
            standard_version: the version of <standard> to apply. Default is
                              1.0
            strict: whether the <standard> should be strictly enforced. Default
                    is True.
            tolerance: tolerance to use when reindexing onto a regular index.
        """

        _flag = kwargs.pop('flag', DecadesClassicFlag)
        _standard = kwargs.pop('standard', 'ppodd.standard.core')
        _standard_version = kwargs.pop('standard_version', 1.0)
        _strict = kwargs.pop('strict', True)
        _tolerance = kwargs.pop('tolerance', '2L')

        self.attrs = AttributesCollection(
            dataset=self, definition='.'.join((_standard, 'variable_attrs')),
            version=_standard_version, strict=_strict
        )

        self.name = kwargs.pop('name', None)
        self.write = kwargs.pop('write', True)

        # Set attributes given as keyword arguments
        _attrs = self.attrs.REQUIRED_ATTRIBUTES + self.attrs.OPTIONAL_ATTRIBUTES
        for _attr in _attrs:
            # Get the default value of an attribute, if it exists
            try:
                _default = self.attrs._definition[_attr]['default']
            except KeyError:
                _default = None

            # Pop the attribute off the keyword stack, and set it if it has a
            # value
            _val = kwargs.pop(_attr, _default)
            if _val is not None:
                self.attrs.add(Attribute(_attr, _val))

        # Create an interim DataFrame, and infer its frequency
        _df = pd.DataFrame(*args, **kwargs)
        _freq = self._get_freq(df=_df)

        # Create an index spanning the variable value at the correct frequency
        _index = pd.date_range(
            start=_df.index[0], end=_df.index[-1],
            freq=_freq
        )

        # If no variable name is given, infer it from the first column of the
        # dataframe
        if self.name is None:
            self.name = _df.columns[0]

        # Deal with non-unique entries in the dataframe, by selecting the last
        # element
        if len(_df.index) != len(_df.index.unique()):
            _df = _df.groupby(_df.index).last()

        # Create the data array that we're going to keep. We're going to
        # reindex the dataframe onto the complete index, and downcast it to the
        # smallest reasonable datatype. This is a memory saving trick, as we
        # don't have to have a 64-bit index associated with every variable.
        self.array = self._downcast(np.array(
            _df.reindex(
                _index, tolerance=_tolerance, method='nearest', limit=1
            ).values.flatten()
        ))

        # t0 and t1 are the start and end times of the array, which we're going
        # to store as we dispose of the index
        self.t0 = _index[0]
        self.t1 = _index[-1]

        # Create the QC Flag array, and add the name of the flag variable to
        # the 'ancillary_variables' attribute. TODO: should we check if this
        # attribute already exists?
        self.flag = _flag(self)
        self.attrs.add(Attribute('ancillary_variables', f'{self.name}_FLAG'))

    def __call__(self):
        """
        Implement (). When a class instance is called, create and return a
        Pandas Series with the correct index.
        """
        i = pd.date_range(
            start=self.t0, end=self.t1,
            freq=self._get_freq()
        )

        return pd.Series(self.array, index=i, name=self.name)

    def __len__(self):
        """
        Impement len(). The length of a variable is the length of its array
        attribute.
        """
        return len(self.array)

    def __getattribute__(self, attr):
        """
        Implement .<attr>.
        """
        # pylint: disable=too-many-return-statements

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
            # pylint: disable=pointless-statement
            self.__dict__['t0']
            self.__dict__['t1']
            return getattr(self(), attr)
        except (KeyError, AttributeError):
            pass

        raise AttributeError(f'Not a variable attribute: {attr}')

    def __setattr__(self, attr, value):
        """
        Manage setting of attributes.

        Kwargs:
            attr: the name of the attribute to set
            value: the value of the attribute to set
        """
        if attr == 'attrs':
            super().__setattr__(attr, value)

        if attr in (
            self.attrs.REQUIRED_ATTRIBUTES + self.attrs.OPTIONAL_ATTRIBUTES
        ):
            self.attrs[attr] = value

        super().__setattr__(attr, value)

    def __str__(self):
        """
        Implement str()
        """
        return self.name

    def __repr__(self):
        """
        Implement repr()
        """
        return r'<DecadesVariable[{!r}]>'.format(self.name)

    def _get_freq(self, df=None):
        """
        Return the frequency of the variable.

        Kwargs:
            df: if given, infer the frequency of this dataframe.

        Returns:
            _freq: the frequency code of the variable.
        """
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

    @staticmethod
    def _downcast(array):
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
        """
        Merge this variable with another variable, assumed to be the same data
        stream over a different time interval. This fast merge assumes that the
        <other> variable occurs after this one in time, and that the
        intersection of indicies is the empty set.

        Args:
            other: the variable to merge with this one.
        """

        # Create a union of data and indicies
        _data = np.concatenate([self.array, other.array])
        _index = self.index.union(other.index)

        # Reindex to ensure no gaps in the data
        _df = pd.DataFrame(_data, index=_index).reindex(
            pd.date_range(start=_index[0], end=_index[-1],
                          freq=pd_freq[self.frequency])
        )

        # Store the merged data
        self.array = _df.values.flatten()
        self.t0 = _df.index[0]
        self.t1 = _df.index[-1]

    def trim(self, start, end):
        """
        'Trim' the variable to a subset of itself, via a top and tail. The
        interval is closed (i.e. start and end will remain in the variable).

        Args:
            start: a datetime like indicating the start of the period to keep
            end: a datetime like indicating the end of the period to keep.
        """

        # Trim the QC flag over the same interval.
        self.flag.trim(start, end)

        # Create a dataframe, index to the required interval, and extract the
        # required attributes to store.
        _df = self()
        loc = (_df.index >= start) & (_df.index <= end)
        trimmed = _df.loc[loc]
        self.array = trimmed.values.flatten()
        self.t0 = trimmed.index[0]
        self.t1 = trimmed.index[-1]

    def merge(self, other):
        """
        Merge another variable, assumed to represent the same data field over a
        different period into this one.

        Args:
            other: the variable to merge into this one.
        """

        # If the other variable is after this one, we can merge fast...
        if other.t0 > self.t1:
            self._merge_fast(other)
            return

        # ...otherwise we have to use a slower method
        other = other()
        current = self()

        # Create a union of the indexes of both variables
        merge_index = current.index.union(other.index).sort_values().unique()

        # Reindex this variable onto the merged index...
        current = current.reindex(merge_index)
        # ...and merge in the other variable
        current.loc[other.index] = other

        # Reindex to ensure there aren't any data gaps
        full_index = pd.date_range(
            start=merge_index[0], end=merge_index[-1],
            freq=pd_freq[self.frequency]
        )
        current = current.reindex(full_index)

        # Store the required attributes
        self.array = current.values.flatten()
        self.t0 = current.index[0]
        self.t1 = current.index[-1]

    def time_bounds(self):
        """
        Return the start and end times of this variable, as the tuple (start,
        end).
        """
        return (self.t0, self.t1)


class DecadesDataset(object):
    """
    A DecadesDataset is a top-level wrapper used to load input data, run
    processing, qa, and potentially write-out data via an injected dependency.
    """
    # pylint: disable=too-many-public-methods, too-many-arguments

    def __init__(self, date=None, standard_version=1.0, backend=DefaultBackend,
                 writer=NetCDFWriter, pp_plugins='ppodd.pod',
                 standard='ppodd.standard.core', strict=True):
        """
        Create a class instance.

        Kwargs:
            date [None]: a date representing the date of the flight.
            standard [ppodd.standard.core]: the metadata standard to adhere to.
            standard_version [1.0]: the version of <standard> to apply.
            strict [True]: indicates whether the <standard> should be strictly
                enforced.
            backend [DefaultBackend]: the backend to use for variable storage.
            pp_plugins [ppodd.pod]: a string pointing to the module containing
                the postprocessing modules to use.
            writer [NetCDFWriter]: the writer class to use by default.
        """

        self._date = date
        self.lazy = Lazy(self)
        self.readers = []
        self.definitions = []
        self.constants = {}
        self._variable_mods = {}
        self._flag_mods = {}
        self._mod_exclusions = []
        self.pp_modules = []
        self.qa_modules = []
        self.flag_modules = []
        self.completed_modules = []
        self.failed_modules = []

        self.globals = AttributesCollection(
            dataset=self, definition='.'.join((standard, 'dataset_globals')),
            version=standard_version, strict=strict
        )

        self.writer = writer
        self.pp_plugins = pp_plugins

        self._dataframes = {}
        self.lon = None
        self.lat = None
        self._garbage_collect = False
        self._qa_dir = None
        self._takeoff_time = None
        self._landing_time = None
        self._decache = False
        self._trim = False
        self._standard = standard
        self._strict_mode = strict
        self.allow_overwrite = False
        self._backend = backend()

    def __getitem__(self, item):
        """
        Implement getitem().

        Args:
            item: the item to get and return.
        """

        # Try to get the item from the backend - i.e. return a variable
        try:
            return self._backend[item]
        except KeyError:
            pass

        # Try to return a constant
        try:
            return self.constants[item]
        except KeyError:
            pass

        # TIME_(MIN/MAX)_CALL return callables when can be used to return the
        # minimum and maximum times of any variable in the dataset.
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
        """
        Remove a variable from the backend.

        Args:
            name: the name of the variable to remove from the backend.
        """
        self._backend.remove(name)

    def garbage_collect(self, collect):
        """
        Turn garbage collection on or off. If on, variables which are not
        required for processing are removed from the dataset. TODO: this should
        probably be done as a property.
        """
        if collect:
            self._garbage_collect = True
            return
        self._garbage_collect = False

    def decache(self, decache):
        """
        Turn decaching on or off. If on, the backend will regularly remove data
        from memory to whatever storage medium is used.

        Args:
            decache: a boolean indicating whether decaching should be enable of
                disabled.
        """
        if decache:
            self._decache = True
            return
        self._decache = False

    @property
    def outputs(self):
        """
        Return a list of output variable from the backend.
        """
        return self._backend.outputs

    def clear_outputs(self):
        """
        Clear all outputs from the backend.
        """
        self._backend.outputs = []

    @property
    def trim(self):
        """
        Return True if trim mode is on, False otherwise. If trim mode is on,
        data will be trimmed to a set period before takeoff and after landing.
        """
        return self._trim

    @trim.setter
    def trim(self, trm):
        """
        Set trim mode.

        Args:
            trm: if True, turn trim mode on, if False turn trim mode off.
        """
        self._trim = trm

    @property
    def date(self):
        """
        Return the date of this dataset.
        """
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
        if isinstance(value, dict):
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
        """
        Return the qa_dir property, which is the directory in which QC should
        be done.
        """
        return self._qa_dir

    @qa_dir.setter
    def qa_dir(self, qa_dir):
        """
        Set the qa_dir, creating it if it does not exist. If it exists and is
        not a directory, raise OSError.

        Args:
            qa_dir: a string pointing to the path to use as a qa directory.
        """
        if not os.path.isdir(qa_dir) and os.path.exists(qa_dir):
            raise OSError('{} exists and is not a directory'.format(qa_dir))

        if not os.path.exists(qa_dir):
            os.makedirs(qa_dir)

        self._qa_dir = qa_dir

    @staticmethod
    def infer_reader(dfile):
        """
        Infer the reader class which should be used to load a given file.

        Args:
            dfile: the file from which to infer the reader. Expected to be of
                type DecadesFile.
        """
        # pylint: disable=relative-beyond-top-level, import-outside-toplevel
        from ..readers import reader_patterns
        _filename = os.path.basename(dfile.filepath)

        for pattern in reader_patterns:
            if re.fullmatch(pattern, _filename):
                return reader_patterns[pattern]

        print('No reader found for {}'.format(dfile))
        return None

    def add_definition(self, definition):
        """
        Add a definition to the list of definitions stored in the instance.

        Args:
            definition: the definition to add.
        """
        self.definitions.append(definition)

    def add_constant(self, name, data):
        """
        Add a constant to the dataset, to be stored in the constants list
        attribute.

        Args:
            name: the name of the constant
            data: the value of the constant
        """
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
        """
        Add an output variable to the dataset, via the backend.

        Args:
            variable: the DecadesVariable to add as an output.
        """
        self._backend.add_output(variable)

    @property
    def variables(self):
        """
        Define the .variables property, which returns all of the variables held
        in the backend.
        """
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
        # pylint: disable=no-self-use
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
        # pylint: disable=redefined-outer-name, import-outside-toplevel
        import ppodd.qa
        for reader in self.readers:
            try:
                reader.read()
            except Exception as err: # pylint: disable=broad-except
                print(f'Error in reading module {reader}')
                print(str(err))
            del reader

        self.readers = None
        gc.collect()

        self.qa_modules = [qa(self) for qa in ppodd.qa.qa_modules]

        # Initialise postprocessing modules
        _pp_modules = []
        for pp in importlib.import_module(self.pp_plugins).pp_modules:
            try:
                _pp_modules.append(pp(self))
            except Exception as err: # pylint: disable=broad-except
                print('Couldn\'t init {}: {}'.format(pp, str(err)))
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
        """
        Get all of the data which may still be required for a processing job.
        Allows for garbage collection.
        """

        _required_inputs = []
        # Any inputs from pp modules which haven't been run may be required
        for mod in self.pp_modules:
            _required_inputs += mod.inputs

        # Any inputs from qa modules which haven't run may be required.
        for qa in self.qa_modules:
            _required_inputs += qa.inputs

        # Any input variable who's write attribute is True is still required
        for var in self.variables:
            try:
                if self[var].write:
                    _required_inputs.append(var)
            except KeyError:
                continue

        # Any variable who's write attribute may be set to True later is still
        # required
        for var in self._variable_mods:
            try:
                if self._variable_mods[var]['write']:
                    _required_inputs.append(var)
            except KeyError:
                # Likely an output modifier
                pass

        # Ensure the list of required inputs is unique
        _required_inputs = list(set(_required_inputs))

        return _required_inputs

    def _collect_garbage(self):
        """
        Run the garbage collector, via the backend.
        """

        # No-op if garbage collect mode is off.
        if not self._garbage_collect:
            return

        # Get data which are still required
        required_inputs = self._get_required_data()

        # Run the garbage collector
        self._backend.collect_garbage(required_inputs)

        # Run the interpreter garbage collector, 'cause overkill
        gc.collect()

    def run_qa(self):
        """
        Run QA (QC) modules. These are typically used to produce figures for
        QC, but may do anything QC-related.
        """
        while self.qa_modules:
            _mod = self.qa_modules.pop()
            try:
                _mod.run()
                del _mod
            except Exception as err: # pylint: disable=broad-except
                print(' ** Error in {}: {}'.format(_mod, err))

    def run_flagging(self):
        """
        Run flagging modules. Flagging modules provide data quality flags which
        cannot be defined at processing time, typically due to requiring data
        from some other processing module which can't be guaranteed to exist.
        """

        while self.flag_modules:
            _flag = self.flag_modules.pop()
            try:
                print('running {}'.format(_flag))
                _flag.flag()
                del _flag
            except Exception as err: # pylint: disable=broad-except
                print(' ** Error in {}: {}'.format(_flag, err))

    def _trim_data(self):
        """
        Trim all of the data in this dataset based on takeoff and landing
        times.
        """
        if self.takeoff_time and self.landing_time:
            start_cutoff = self.takeoff_time - datetime.timedelta(hours=2)
            end_cutoff = self.landing_time + datetime.timedelta(minutes=30)
            print(f'trimming: {start_cutoff} -- {end_cutoff}')
            self._backend.trim(start_cutoff, end_cutoff)


    def process(self, modname=None):
        """
        Run processing modules.
        """
        # The linter directives below probably mean this could do with
        # some refactoring...
        # pylint: disable=too-many-locals, too-many-branches
        # pylint: disable=too-many-statements

        if self.trim:
            self._trim_data()

        # If a module name is given, then we're only going to try to run that
        # module.
        if modname is not None:
            # Import all of the processing modules
            mods = importlib.import_module(self.pp_plugins).pp_modules

            # Find an run the correct module
            for mod in mods:
                if mod.__name__ is modname:
                    _mod = mod(self)
                    _mod_ready, _missing = _mod.ready()
                    if _mod_ready:
                        # We have all the inputs to run the module
                        _mod.process()
                        _mod.finalize()
                    else:
                        # The module  could not be run, due to lacking inputs.
                        print(
                            ('Module {} not ready: '
                             'inputs not available {}').format(
                                 mod.__name__, ','.join(_missing))
                        )

            # Don't run further modules
            return

        # Initialize qa modules
        self.qa_modules = [qa(self) for qa in ppodd.qa.qa_modules]

        # Initialize postprocessing modules
        _pp_modules = []
        for pp in importlib.import_module(self.pp_plugins).pp_modules:
            try:
                _pp_modules.append(pp(self))
            except Exception as err: # pylint: disable=broad-except
                print('Couldn\'t init {}: {}'.format(pp, str(err)))
        self.pp_modules = collections.deque(_pp_modules)

        # Initialize flagging modules
        self.flag_modules = [flag(self) for flag in ppodd.flags.flag_modules]

        # Clear all outputs whice have been defined previously
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
            except Exception as err: # pylint: disable=broad-except
                print(' ** Error in {}: {}'.format(pp_module, err))
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

    def write(self, *args, **kwargs):
        """
        Write data to file, by instantiating the self.writer injected class.
        All args and kwargs are passed to this classes __init__.
        """
        self.writer(self).write(*args, **kwargs)

    def cleanup(self):
        """
        Do any cleanup required by the backend.
        """
        self._backend.cleanup()

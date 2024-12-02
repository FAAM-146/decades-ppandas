import collections
import datetime
import gc
import glob
import logging
import os
import re
import traceback

from pydoc import locate
import warnings

import numpy as np

from vocal.types import OptionalDerivedString


from ppodd.decades.process import DecadesProcessor
from ppodd.decades.backends import DefaultBackend
from ppodd.decades.utils import DatasetNormalizer, Lazy
from ppodd.decades.file import DecadesFile
from ppodd.decades.attributes import (
    AttributesCollection, Attribute, Context, ATTR_USE_EXAMPLE
)
from ppodd.decades.attrutils import attribute_helpers
from ppodd.writers import NetCDFWriter


logger = logging.getLogger(__name__)


class DecadesDataset(object):
    """
    A DecadesDataset is a top-level wrapper used to load input data, run
    processing, qa, and potentially write-out data via an injected dependency.
    """
    # pylint: disable=too-many-public-methods, too-many-arguments

    def __init__(self, date=None, processor=DecadesProcessor,
                 backend=DefaultBackend,
                 writer=NetCDFWriter, pp_group='core',
                 standard='faam_data', strict=True, logfile=None):
        """
        Create a class instance.

        Args:
            date (datetime.datetime, optional): a date representing the date of
                the flight. Default is None, in which case a date is expected
                via a constants file.
            standard (str, optional): the metadata standard to adhere to.
                Default is `ppodd.standard.core`.
            strict (bool, optional): indicates whether the <standard> should be
                strictly enforced. Default is True.
            backend (ppodd.decades.backends.DecadesBackend): the backend to use
                for variable storage. Default is
                ppodd.decades.backends.DefaultBackend.
            pp_group (str, optional): a string pointing indicating which group
                of postprocessing modules should be run. Default is `core`.
            writer (ppodd.writers.writers.DecadesWriter): the writer class to
                use by default. Default is NetCDFWriter.
        """

        self._date = date
        self.lazy = Lazy(self)
        self.readers = []
        self.definitions = []
        self.constants = {}
        self._variable_mods = {}
        self._flag_mods = {}
        self._mod_exclusions = []
        self.qa_modules = []

        self.load_hooks = []
        self.process_hooks = []
        self._attribute_helpers = []
        self.writer = writer
        self.processor = processor(self)
        self.pp_group = pp_group

        self._dataframes = {}
        self.lon = None
        self.lat = None
        self._garbage_collect = False
        self._qa_dir = None
   
        self._decache = False
        self._trim = False
        self._standard = standard
        self._strict_mode = strict
        self.allow_overwrite = False
        self._backend = backend()

        self.globals = AttributesCollection(
            dataset=self, definition='.'.join((standard, 'GlobalAttributes')),
            strict=strict
        )

        for helper in attribute_helpers:
            self._attribute_helpers.append(helper(self))

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
            return lambda: self.time_bounds()[0]#.strftime('%Y-%m-%dT%H:%M:%SZ')

        if item == 'TIME_MAX_CALL':
            return lambda: self.time_bounds()[1]#.strftime('%Y-%m-%dT%H:%M:%SZ')

        raise KeyError('Unknown variable: {}'.format(item))

    def __getattr__(self, attr):
        """
        Allow the use of helpers to provide additional attributes to the dataset.
        """
        try:
            for helper in self._attribute_helpers:
                if attr in helper.attributes:
                    return getattr(helper, attr)
        except Exception:
            pass

        raise AttributeError(f'{attr!r} is not an attribute of {self!r}')

    def __setattr__(self, name, value):
        """
        Intercept attributes that are set, and pass them onto attribute helpers,
        if required. This would probably be better done with on-the-fly mixins,
        but never mind...
        """
       
        try:
            for helper in self.__dict__['_attribute_helpers']:
                if name in helper.attributes:
                    setattr(helper, name, value)
                    return
        except KeyError:
            pass
            
        super().__setattr__(name, value)

    def normalize(self, frequency: int) -> DatasetNormalizer:
        """
        Return a context manager which will normalize all variables in the
        dataset to a given frequency.

        Args:
            frequency (int): the frequency to normalize to.
        """
        return DatasetNormalizer(self, frequency)

    @property
    def coords(self):
        """
        str: the string to use as a variable coordinates attribute. None if
        this is not set.
        """
        if self.lat and self.lon:
            return f'{self.lon} {self.lat}'
        return None

    def time_bounds(self):
        """
        Return the time period covered by this dataset.

        Returns:
            tuple: a 2-tuple containing the smallest and largest times of
            variables in this dataset.
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
            name (str): the name of the variable to remove from the backend.
        """
        self._backend.remove(name)

    def garbage_collect(self, collect):
        """
        Turn garbage collection on or off. If on, variables which are not
        required for processing are removed from the dataset. TODO: this should
        probably be done as a property.

        Args:
            collect (bool): A bool indicating if garbage collection should be
                turned on (True) or off (False).
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
            decache (bool): a boolean indicating whether decaching should be
                enabled or disabled.
        """
        if decache:
            self._decache = True
            return
        self._decache = False

    @property
    def outputs(self):
        """
        list: Return a list of output variables from the backend.
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
        bool: True if trim mode is on, False otherwise. If trim mode is on,
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
        datetime.datetime: Return the date of this dataset.
        """
        if self._date:
            return datetime.datetime.combine(
                self._date, datetime.datetime.min.time()
            ).date()

        try:
            return datetime.datetime.combine(
                self.globals['date'], datetime.datetime.min.time()
            ).date()
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
            key (str): the name of the global attribute to add
            value (Object): the value of the global attribute <key>
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

        _context = None
        _context_type = Context.ATTR

        if result:
            # The value is a directive; resolve it
            groups = result.groupdict()

            if groups['action'] == 'call':
                # Attempt to locate the value of the call directive. Abandon
                # this global if its not found
                value = locate(groups['value'])
                if value is None:
                    return

            if groups['action'] == 'example':
                value = ATTR_USE_EXAMPLE

            if groups['action'] == 'attribute':
                _context = self
                _context_type = Context.ATTR
                value = groups['value']

                if value is None:
                    return

            if groups['action'] == 'data':
                values = [v.strip() for v in groups['value'].split()]
                item, *callables = values
                _context = self
                _context_type = Context.DATA
                value = tuple([item] + callables)

        # Add the global 
        attr = Attribute(key, value, context=_context, context_type=_context_type)
        self.globals.add(attr)

    @property
    def qa_dir(self):
        """
        str: The directory in which QC should be done. When set, will create
        the directory if it does not exist.

        Raises:
            OSError: if the path exists but is not a directory.
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
            name (str): the name of the constant
            data (Object): the value of the constant
        """
        self.constants[name] = data

    def add_input(self, variable):
        """
        Add a new DecadesVariable to this DecadesDataset. If the variable
        already exists, as identified by name, then append to that variable
        using pd.Series.combine_first().

        Args:
            variable (DecadesVariable): the variable to add to this
                DecadesDataset.
        """

        self._backend.add_input(variable)

    def add_output(self, variable):
        """
        Add an output variable to the dataset, via the backend.

        Args:
            variable: the DecadesVariable to add as an output.
        """
        coord_var = (
            OptionalDerivedString if variable.doc_mode else lambda: self.coords
        )
        variable.attrs.add(Attribute('coordinates', coord_var))
        self._backend.add_output(variable)

    @property
    def variables(self):
        """
        :obj:`list` of :obj:`str`: Returns all of the variables held
        in the backend.
        """
        return self._backend.variables

    @property
    def files(self):
        """
        :obj:`list` of :obj:`str`: Return a list of all files currently
        accociated with this :obj:`DecadesDataset`. Cannot be set
        explicitally.
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

        Args:
            cls: The class (type) of the reader we wish to retrieve.

        Returns:
            The reader of type cls, if such a reader exists.
        """

        for reader in self.readers:
            if reader.__class__ == cls:
                return reader
        raise ValueError

    @staticmethod
    def infer_reader(dfile):
        """
        Infer the reader class which should be used to load a given file.

        Args:
            dfile (DecadesFile): the file from which to infer the
                reader.
        """
        # pylint: disable=relative-beyond-top-level, import-outside-toplevel
        from ppodd.readers import reader_patterns
        _filename = os.path.basename(dfile.filepath)

        for pattern in reader_patterns:
            if re.fullmatch(pattern, _filename):
                return reader_patterns[pattern]

        logger.warning('No reader found for {}'.format(dfile))
        return None

    def add_decades_file(self, dfile):
        """
        Add a DecadesFile to the DecadesDataset. Each file must be associated
        with a reader. If a suitable reader is already associated with the
        Dataset, then add the file to that reader. If not, create a new reader,
        add that to the dataset, and then add the file to the reader.

        Args:
            dfile (DecadesFile): The file to add to the dataset.
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

        Args:
            filename (str): the path to the file to add.
        """
        logger.info('Adding {}'.format(filename))
        try:
            self.add_decades_file(DecadesFile(filename))
        except ValueError:
            logger.error('failed to add {}'.format(filename))

    def run_load_hooks(self):
        """
        Run all of the post-load hooks associated with this dataset.
        """
        for hook in self.load_hooks:
            try:
                hook(self)
            except Exception:
                logger.error('Error running post-load hook')

    def run_process_hooks(self, module_name=None):
        """
        Run some or all of the post-process hooks associated with this dataset.

        Args:
            module_name (str, optional): the name of the module to run the
                post-process hooks for. If None, run all post-process hooks.
        """
        for hook in self.process_hooks:

            if module_name is not None:
                if getattr(hook, 'module_name', None) != module_name:
                    continue

            if module_name is None and getattr(hook, 'module_name', None) is not None:
                continue

            name = getattr(hook, 'hook_name', 'Unnamed hook')
            logger.info(f'Running post-process hook "{name}"')
            
            try:
                hook(self)
            except Exception:
                logger.error('Error running post-process hook', exc_info=True)

    def load(self):
        """
        Load all of the data from files associated with readers in this
        dataset.
        """
        # pylint: disable=redefined-outer-name, import-outside-toplevel
        import ppodd.qa
        # from ppodd.pod.base import pp_register
        for reader in self.readers:
            try:
                reader.read()
            except Exception as err: # pylint: disable=broad-except
                logger.error(f'Error in reading module {reader}')
                logger.error(str(err))
                raise
            del reader

        self.readers = None
        # gc.collect()

        self.qa_modules = [qa(self) for qa in ppodd.qa.qa_modules]

        # # Initialise postprocessing modules
        # _pp_modules = []
        # for pp in pp_register.modules(self.pp_group, date=self.date):
        #     try:
        #         _pp_modules.append(pp(self))
        #     except Exception as err: # pylint: disable=broad-except
        #         logger.warning('Couldn\'t init {}: {}'.format(pp, str(err)))

        self._interpolate_globals()

        self.run_load_hooks()

   
    def _get_required_data(self):
        """
        Get all of the data which may still be required for a processing job.
        Allows for garbage collection.

        Returns:
            :obj:`list` of :obj:`str`.
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
        warnings.warn('run_qa is deprecated, use run_qc instead', DeprecationWarning)
        self.run_qc()

    def run_qc(self):
        """
        Run QC modules. These are typically used to produce figures for
        QC, but may do anything QC-related.
        """
        while self.qa_modules:
            _mod = self.qa_modules.pop()
            if not _mod.ready():
                continue
            try:
                _mod.run()
                del _mod
            except Exception as err: # pylint: disable=broad-except
                logger.error('Error in {}: {}'.format(_mod, err))
                traceback.print_exc()

    def _trim_data(self, start_cutoff=None, end_cutoff=None):
        """
        Trim all of the data in this dataset based on takeoff and landing
        times.
        """
        if start_cutoff is None:
            if self.takeoff_time and self.landing_time:
                start_cutoff = self.takeoff_time - datetime.timedelta(hours=2)

        if end_cutoff is None:
            if self.takeoff_time and self.landing_time:
                end_cutoff = self.landing_time + datetime.timedelta(minutes=30)

        if end_cutoff is None or start_cutoff is None:
            return

        logger.debug(f'trimming: {start_cutoff} -- {end_cutoff}')
        self._backend.trim(start_cutoff, end_cutoff)


    def process(self, modname=None):
        self.processor.process(modname=modname)

    def _finalize(self):
        """
        Finalization tasks
        """
        for output in [i.name for i in self._backend.outputs if i.name.endswith('_CU')]:
            var_name = output.replace('_CU', '')

            if self[var_name].flag is not None:
                try:
                    flag = self[var_name].flag()
                    self[output].array[flag > 0] = np.nan
                except Exception:
                    logger.error('Failed to nan flag for output',
                                 exc_info=True)

    def write(self, *args, **kwargs):
        """
        Write data to file, by instantiating the self.writer injected class.
        All args and kwargs are passed to this classes `write` method.
        """
        self.writer(self).write(*args, **kwargs)

    def cleanup(self):
        """
        Do any cleanup required by the backend.
        """
        self._backend.cleanup()

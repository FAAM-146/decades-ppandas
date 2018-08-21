import collections
import importlib
import sys

import pandas as pd

CRIO_FILE_IDENTIFIERS = ['AERACK01', 'CORCON01', 'CPC378001', 'PRTAFT01',
                         'TWCDAT01', 'WVSS2A01', 'WVSS2B01', 'CPC37801',
                         'UPPBBR01', 'LOWBBR01']

GIN_FILE_IDENTIFIERS = ['GINDAT01']


class DecadesFile(object):
    def __init__(self, filepath, file_type=None):
        self.filepath = filepath

        if file_type is not None:
            self.file_type = file_type
        else:
            self.file_type = DecadesFile.infer_type(self.filepath)

    @staticmethod
    def infer_type(filepath):
        if filepath.endswith('.json'):
            return 'constants'

        if filepath.endswith('.zip'):
            return 'zip'

        for _id in CRIO_FILE_IDENTIFIERS:
            if _id in filepath:
                if filepath.endswith('.csv'):
                    return 'criodef'
                return 'crio'

        for _id in GIN_FILE_IDENTIFIERS:
            if _id in filepath:
                if filepath.endswith('.csv'):
                    return 'gindef'
                if filepath.endswith('.bin'):
                    return 'gin'

    def __repr__(self):
        return '{}({!r}, file_type={!r})'.format(
            self.__class__.__name__, self.filepath, self.file_type
        )


class DecadesVariable(pd.DataFrame):
    _metadata = ['long_name', 'name', 'standard_name', 'frequency', 'number',
                 'units']

    def __init__(self, *args, **kwargs):
        long_name = kwargs.pop('long_name', None)
        name = kwargs.pop('name', None)
        standard_name = kwargs.pop('standard_name', None)
        frequency = kwargs.pop('frequency', None)
        number = kwargs.pop('number', None)
        units = kwargs.pop('units', None)

        super(DecadesVariable, self).__init__(*args, **kwargs)

        self.long_name = long_name
        self.name = name
        self.standard_name = standard_name
        self.frequency = frequency
        self.number = number
        self.units = units

    def _set_metadata(self, frame):
        for meta in DecadesVariable._metadata:
            setattr(frame, meta, getattr(self, meta))

    def combine_first(self, other):
        """
        We use combine_first to merge variables from different crio files into
        a single pd.Series. However we want to ensure that a) variables from
        different files can only be merged if they have the same name, and
        b) that the resulting Series object retains this name.
        """
        if self.name != other.name:
            return ValueError('Can only combine variables of same type')
        df = super(DecadesVariable, self).combine_first(other)
        self._set_metadata(df)
        return df

    def asfreq(self, freq, *args, **kwargs):
        """
        As a convenience, we may wish to as asfreq() with a simple Hz value.
        Thus we override to accept an integer, which is converted to a pandas
        period string (Nanosecond) before use.
        """
        if type(freq) is int:
            _period = int((1 / freq) * 10**9)
            freq = '{}N'.format(_period)

        return super(DecadesVariable, self).asfreq(freq, *args, **kwargs)

    def round(self, *args, **kwargs):
        return self._set_metadata(
            super(DecadesVariable, self).round(*args, **kwargs)
        )

    @property
    def _constructor(self):
        return DecadesVariable


class DecadesDataset(object):
    def __init__(self, date=None):

        if date is None:
            raise ValueError('Flight date must be given')

        self.date = date
        self.readers = []
        self.definitions = []
        self.constants = {}
        self.inputs = []
        self.outputs = []

    def __getitem__(self, item):
        for _var in self.inputs + self.outputs:
            if _var.name == item:
                return _var

        try:
            return self.constants[item]
        except KeyError:
            pass

        raise KeyError('Unknown variable: {}'.format(item))

    @staticmethod
    def infer_reader(file_type):
        if file_type == 'crio':
            return 'ppodd.readers.CrioFileReader'
        if file_type == 'criodef':
            return 'ppodd.readers.CrioDefinitionReader'
        if file_type == 'gindef':
            return 'ppodd.readers.CrioDefinitionReader'
        if file_type == 'gin':
            return 'ppodd.readers.GinFileReader'
        if file_type == 'zip':
            return 'ppodd.readers.ZipFileReader'
        if file_type == 'constants':
            return 'ppodd.readers.JsonConstantsReader'

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

        _var = None
        for i, _variable in enumerate(self.inputs):
            if _variable.name == variable.name:
                _var = self.inputs.pop(i)
                break

        if _var is not None:
            _new_var = _var.combine_first(variable)
            self.inputs.append(_new_var)
            return

        self.inputs.append(variable)

    @property
    def variables(self):
        return [i.name for i in self.inputs + self.outputs]

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

        if dfile.file_type is None:
            raise ValueError('Cannot infer filetype of {}'.format(dfile))

        print(dfile.file_type)

        reader_path = DecadesDataset.infer_reader(dfile.file_type)
        reader_module, reader_class = reader_path.rsplit('.', maxsplit=1)

        # Ensure we only import once, as using importlib twice can screw up
        # calls to super(). Computers, eh?
        if reader_module in sys.modules:
            reader_module = sys.modules[reader_module]
        else:
            reader_module = importlib.import_module(reader_module)

        reader_class = getattr(
            reader_module, reader_class
        )

        try:
            reader = self.get_reader(reader_class)
        except ValueError:
            reader = reader_class()
            self.readers.append(reader)
            self.readers.sort(key=lambda r: r.level)

        reader.files.append(dfile)

    def add_file(self, filename, file_type=None):
        """
        Add a file to the dataset, first creating a DecadesFile.

        args:
            filename: the path to the file to add.

        kwargs:
            file_type: the file type to pass to DecadesFile
        """
        print('adding {}'.format(filename))
        try:
            self.add_decades_file(DecadesFile(filename, file_type=file_type))
        except ValueError:
            print('failed to add {}'.format(filename))

    def load(self):
        """
        Load all of the data from files associated with readers in this
        dataset.
        """

        for reader in self.readers:
            reader.read()

    def process(self):
        """
        Run processing modules.
        """
        import ppodd.pod

        self.outputs = []

        self.pp_modules = collections.deque(
            [pp(self) for pp in ppodd.pod.pp_modules]
        )
        self.completed_modules = []
        self.failed_modules = []

        module_ran = True
        while self.pp_modules and module_ran:
            module_ran = False

            pp_module = self.pp_modules.popleft()

            if not pp_module.ready():
                self.pp_modules.append(pp_module)
                continue
            try:
                print('Running {}'.format(pp_module))
                pp_module.process()
            except Exception as e:
                print(str(e))
                raise
                self.failed_modules.append(pp_module)
            else:
                self.completed_modules.append(pp_module)

            module_ran = True


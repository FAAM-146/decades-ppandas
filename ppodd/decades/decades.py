import collections
import importlib
import sys
import re
import os

import numpy as np
import pandas as pd


class DecadesFile(object):
    def __init__(self, filepath):
        self.filepath = filepath

    def __repr__(self):
        return '{}({!r})'.format(
            self.__class__.__name__, self.filepath
        )


class DecadesVariable(object):

    def __init__(self, *args, **kwargs):
        long_name = kwargs.pop('long_name', None)
        name = kwargs.pop('name', None)
        standard_name = kwargs.pop('standard_name', None)
        frequency = kwargs.pop('frequency', None)
        number = kwargs.pop('number', None)
        units = kwargs.pop('units', None)

        self._df = pd.DataFrame(*args, **kwargs)

        # We only want to be given at most two columns (variable, flag) and at
        # least one column (variable). Raise an error if this is not the case.
        _columns = [i for i in self._df]
        if len(_columns) > 2:
            raise ValueError('Too many columns in instance DataFrame')
        if len(_columns) < 1:
            raise ValueError('Too few columns in instance DataFrame')

        # If there's a column with FLAG in the name, then this is our flag
        # variable.
        try:
            _flag_var = [i for i in _columns if 'FLAG' in i][0]
            _columns.remove(_flag_var)
            self.is_flagged = True
        except IndexError as e:
            self.is_flagged = False

        # Deal with variable/instance naming. If no 'name' keyword is
        # specified, then insist that the variable and flag are consistently
        # named. Otherwise, rename both to be consistent with the kwarg.
        _var = _columns[0]
        if name is None:
            if self.is_flagged and _var != _flag_var.replace('_FLAG', ''):
                raise ValueError(
                    ('Inconsistent variable/flag names. Expected '
                     'VAR / VAR_FLAG')
                )
            self.name = _var
        else:
            self.name = name
            _rename = {_var: name}
            if self.is_flagged:
                _rename[_flag_var] = '{}_FLAG'.format(name)
            self._df.rename(columns=_rename, inplace=True)

        self.long_name = long_name
        self.standard_name = standard_name
        self.frequency = frequency
        self.number = number
        self.units = units
        self.attrs = {}

    def __str__(self):
        return 'DecadesVariable: {}'.format(self.name)

    def __repr__(self):
        return '<DecadesVariable({!r})>'.format(self.name)

    def __getattr__(self, attr):
        try:
            return self.__dict__[attr]
        except KeyError:
            pass

        if attr == 'data':
            return self._df[self.name]

        if attr == 'flag':
            if not self.is_flagged:
                raise ValueError('Variable is not flagged.')
            return self._df['{}_FLAG'.format(self.name)]

        return getattr(self._df, attr)

    def add_flag(self, flag_data=None, method='merge'):
        """
        Add a flag to the DecadesVariable. If no flag_data is given, the flag
        array is initialized to zero everywhere.

        If the instance already has a flag variable, this can be replaced with
        method='clobber' or merged with method='merge'. A merge is simply the
        element-wise maximum of the existing flag and new flag data.

        Kwargs:
            flag_data: an iterable of flag data, which must be the same length
                       as the variable data.
            method: either 'merge' (default) or 'clobber'. Merge is an
                    element-wise max with the current flag, clobber replaces
                    the current flag if it exists.
        """

        _flag_name = '{}_FLAG'.format(self.name)

        if flag_data is not None:
            if len(flag_data) != len(self.data):
                raise ValueError(
                    'Flag data must be the same length as variable'
                )

        if _flag_name in self._df:
            self.is_flagged = True
            if flag_data is not None:
                if method == 'merge':
                    self._df[_flag_name] = np.fmax(flag_data, self.flag)
                elif method == 'clobber':
                    self._df[_flag_name] = flag_data
                else:
                    raise ValueError('Unknown method: {}'.format(method))

            return _flag_name

        if flag_data is None:
            self._df[_flag_name] = 0
        else:
            self._df[_flag_name] = flag_data
        self.is_flagged = True
        return _flag_name


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
        self.attrs = {}

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

        _var = None
        for i, _variable in enumerate(self.inputs):
            if _variable.name == variable.name:
                _var = self.inputs.pop(i)
                break

        if _var is not None:
            _var._df = _var._df.combine_first(variable._df)
            self.inputs.append(_var)
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

        reader_class = DecadesDataset.infer_reader(dfile)

        try:
            reader = self.get_reader(reader_class)
        except ValueError:
            pass

        try:
            reader = reader_class()
        except TypeError:
            reader = None

        if reader is None:
            raise ValueError

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

        temp_modules = []

        module_ran = True
        while self.pp_modules and module_ran:
            module_ran = False

            pp_module = self.pp_modules.popleft()

            if not pp_module.ready():
                # self.pp_modules.append(pp_module)
                temp_modules.append(pp_module)
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


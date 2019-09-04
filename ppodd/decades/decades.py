import collections
import datetime
import gc
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

    NC_ATTRS = [
        'long_name', 'frequency', 'standard_name', 'units', 'number',
        '_FillValue', 'valid_min', 'valid_max'
    ]

    def __init__(self, *args, **kwargs):
        name = kwargs.pop('name', None)

        self.attrs = {
            '_FillValue': -9999.
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
        except (IndexError, TypeError) as e:
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

        self.write = write

        self.attrs['ancillary_variables'] = '{}_FLAG'.format(self.name)

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
            if not self.is_flagged:
                raise ValueError('Variable is not flagged.')
            return self._df['{}_FLAG'.format(self.name)]

        return getattr(self._df, attr)

    def __setattr__(self, attr, value):
        if attr in DecadesVariable.NC_ATTRS:
            self.attrs[attr] = value
        else:
            super(DecadesVariable, self).__setattr__(attr, value)

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
                    self._df[_flag_name] = flag_data.astype(np.int8)
                else:
                    raise ValueError('Unknown method: {}'.format(method))

            return _flag_name

        if flag_data is None:
            self._df[_flag_name] = 0
        else:
            flag_data[~np.isfinite(flag_data)] = 3
            self._df[_flag_name] = flag_data.astype(np.int8)
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
        self._dataframes = {}
        self._garbage_collect = False
        self._qa_dir = None
        self._takeoff_time = None

        import ppodd.pod
        import ppodd.qa

        self.qa_modules = [qa(self) for qa in ppodd.qa.qa_modules]
        self.pp_modules = collections.deque(
            [pp(self) for pp in ppodd.pod.pp_modules]
        )

    def __getitem__(self, item):

        # When getting a variable, assign its dataframe (_df) attribute from
        # it's parent dataset
        for _var in self.inputs:
            if _var.name == item:

                if _var._df is None:
                    _var._df = self._dataframes[
                        _var.name.split('_')[0]
                    ][_var.frequency][[_var.name]]

                return _var

        for _var in self.outputs:
            if _var.name == item:
                return _var

        try:
            return self.constants[item]
        except KeyError:
            pass

        raise KeyError('Unknown variable: {}'.format(item))

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

        _var = None

        _dlu = variable.name.split('_')[0]
        _freq = variable.frequency

        # If the variable index is not unique, select only the last entry.
        if len(variable._df.index) != len(variable._df.index.unique()):
            variable._df = variable._df.loc[~variable._df.duplicated(keep='last')]

        if _dlu not in self._dataframes:
            self._dataframes[_dlu] = {}

        if variable.frequency not in self._dataframes[_dlu]:
            # The dataframe for the current DLU at the current frequency does
            # not yet exist, so we need to create it, using the variable
            # dataframe
            self._dataframes[_dlu][_freq] = variable._df

        else:
            # The dataframe does exist, so attempt to merge into it, assuming
            # that the index in the variable is covered by that in the
            # dataframe 
            try:
                self._dataframes[_dlu][_freq].loc[
                    variable._df.index, variable.name
                ] = variable._df[variable.name]

            except KeyError:
                # The dataframe does not include all of the indicies present in
                # this variable, therefore we need to reindex

                # Create the new index as the unique union between the
                # dataframe and the variable. 
                _index = self._dataframes[_dlu][_freq].index.union(
                    variable.index).sort_values().unique()

                # Reindex the dataframe
                _df = self._dataframes[_dlu][_freq].reindex(_index)
                self._dataframes[_dlu][_freq] = _df

                # And merge in the variable
                self._dataframes[_dlu][_freq].loc[
                    variable._df.index, variable.name
                ] = variable._df[variable.name]

        # Once the variable has been merged into a dataset dataframe, it no
        # longer needs to maintain its own internal dataframe
        variable._df = None

        # Append the variable to the list of dataset input variables
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

        _copied_inputs = self.inputs.copy()

        for var in _copied_inputs:
            if var.name not in required_inputs:

                try:
                    _dlu = var.name.split('_')[0]
                except Exception:
                    continue

                _freq = var.frequency

                self._dataframes[_dlu][_freq].drop(
                    var.name, axis=1, inplace=True, errors='ignore'
                )

                print('Garbage collect: {}'.format(var.name))
                self.inputs.remove(var)
                del var

        del _copied_inputs
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
                    if _mod.ready():
                        _mod.process()
                        _mod.finalize()
                    else:
                        print('Module {} not ready'.format(mod.__name__))
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

            if not pp_module.ready():
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

            self._collect_garbage()


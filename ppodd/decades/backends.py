import gc
import glob
import os
import pickle
import sqlite3 as sql
import numpy as np
import pandas as pd


class DecadesBackend(object):

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def _dlu_from_variable(self, variable):
        return variable.name.split('_')[0]

    def decache(self):
        pass

    def collect_garbage(self, required_inputs):
        pass

    def cleanup(self):
        pass

    def trim(self, start, end):
        pass

    def add_input(self, variable):
        raise NotImplementedError

    def add_output(self, variable):
        self.outputs.append(variable)

    def clear_outputs(self):
        self.outputs = []

    @property
    def variables(self):
        return [i.name for i in self.inputs + self.outputs]


class Sqlite3Backend(DecadesBackend):

    def __init__(self):
        self.db_file = '_decades_pp.sql'
        if os.path.exists(self.db_file):
            print('warning: removing previous sql file')
            os.remove(self.db_file)
        self.conn = sql.connect(self.db_file)
        super(Sqlite3Backend, self).__init__()

    def _check_table(self, table):
        sql_query = (
            'SELECT name FROM sqlite_master WHERE type="table" '
            'AND name="{table_name}";'.format(table_name=table)
        )

        c = self.conn.cursor()
        c.execute(sql_query)
        return c.fetchone()

    def _create_table(self, dlu, name, _type):
        if _type is bytes:
            _dtype = 'TEXT'
        else:
            _dtype = 'REAL'

        sql_query = (
            'CREATE table {dlu} (time INTEGER primary key, {name} {dtype})'
        ).format(dlu=name, name=name, dtype=_dtype)

        c = self.conn.cursor()
        c.execute(sql_query)
        self.conn.commit()

    def _add_column(self, variable):
        dlu = self._dlu_from_variable(variable)
        name = variable.name

        _type = type(variable.data[0])

        if not self._check_table(name):
            self._create_table(dlu, name, _type=_type)

    def _conv_type(self, variable):
        if type(variable.data[0]) is bytes:
            return [i.decode() for i in variable.data]
        return variable.data.astype(float)

    def _add_data(self, variable):
        _index = variable.data.index.astype(int)
        _data = self._conv_type(variable)

        _table = variable.name
        sql_query = (
            'INSERT into {table} (time, {var}) VALUES (?, ?) '
        )
        sql_query = sql_query.format(table=_table, var=variable.name)
        c = self.conn.cursor()
        c.executemany(sql_query, zip(_index, _data))
        self.conn.commit()

    def add_input(self, variable):
        # If a timestamp is duplicated, only keep the last one
        variable._df = variable._df.groupby(variable._df.index).last()

        self._add_column(variable)
        self._add_data(variable)
        variable._df = None
        self.inputs.append(variable)

    def collect_garbage(self, required_inputs):
        _copied_inputs = self.inputs.copy()

        for var in _copied_inputs:
            if var.name not in required_inputs:
                sql_query = 'DROP table {}'.format(var.name)
                print('GC: {}'.format(sql_query))
                c = self.conn.cursor()
                try:
                    c.execute(sql_query)
                    self.inputs.remove(var)
                except sql.OperationalError:
                    continue
            del var
        del _copied_inputs
        self.conn.commit()

    def cleanup(self):
        os.remove('_decades_pp.sql')

    def __getitem__(self, item):
        for _var in self.inputs:
            if _var.name == item:

                if _var._df is not None:
                    return _var

                _table = item
                sql_query = 'SELECT time, {item} FROM {table} ORDER BY time ASC'.format(
                    table=_table, item=item
                )
                c = self.conn.cursor()
                try:
                    c.execute(sql_query)
                except sql.OperationalError:
                    break
                results = c.fetchall()
                _time, _data = zip(*results)
                _time = np.array(_time, dtype='datetime64[ns]')
                _var._df = pd.DataFrame({item: _data}, index=_time)
                return _var

        for _var in self.outputs:
            if _var.name == item:
                return _var

        raise KeyError('No input: {}'.format(item))


class PandasPickleBackend(DecadesBackend):

    def __getitem__(self, item):
        for _var in self.inputs:
            if _var.name == item:
                if _var._df is not None:
                    return _var

                try:
                    print(f'loading {item}')
                    with open('{}.pkl'.format(item), 'rb') as pkl:
                        _df = pickle.load(pkl)
                except Exception:
                    break

                _var._df = _df
                return _var

        for _var in self.outputs:
            if _var.name == item:
                return _var

        raise KeyError('No input: {}'.format(item))

    def decache(self):
        for var in self.inputs:
            if var._df is None:
                continue

            print(f'decaching {var.name}')
            with open('{}.pkl'.format(var.name), 'wb') as pkl:
                pickle.dump(var._df, pkl)

            var._df = None
        gc.collect()

    def add_input(self, var):
        _df = var._df
        _name = var.name

        try:
            _cdf = self[var.name]._df
        except KeyError:
            with open('{}.pkl'.format(_name), 'wb') as pkl:
                pickle.dump(_df, pkl)
            self.inputs.append(var)
            return

        try:
            self[var.name]._df.loc[
                var._df.index, var.name
            ] = var._df[var.name]

        except KeyError:
            # The dataframe does not include all of the indicies present in
            # this variable, therefore we need to reindex

            # Create the new index as the unique union between the
            # dataframe and the variable. 
            _index = self[var.name].index.union(
                var.index
            ).sort_values().unique()

            # Reindex the dataframe
            _df = self[var.name]._df.reindex(_index)
            self[var.name]._df = _df

            # And merge in the variable
            self[var.name]._df.loc[
                var._df.index, var.name
            ] = var._df[var.name]

        with open('{}.pkl'.format(_name), 'wb') as pkl:
            pickle.dump(var._df, pkl)
        var._df = None

    def trim(self, start, end):
        for var in self.inputs:
            print(f'Trimming: {var.name}')
            df = var._df
            df.drop(df.index[df.index < start], inplace=True)
            df.drop(df.index[df.index > end], inplace=True)

            with open('{}.pkl'.format(var.name), 'wb') as pkl:
                pickle.dump(df, pkl)

            var._df = None

    def collect_garbage(self, required_inputs):
        _copied_inputs = self.inputs.copy()

        for var in _copied_inputs:
            if var.name not in required_inputs:
                print('GC: {}'.format(var))
                self.inputs.remove(var)
                try:
                    os.remove(f'{var.name}.pkl')
                except (FileNotFoundError, OSError):
                    pass

        gc.collect()

    def cleanup(self):
        for _file in glob.glob('*.pkl'):
            os.remove(_file)


class PandasInMemoryBackend(DecadesBackend):
    def __getitem__(self, item):
        for _var in self.inputs + self.outputs:
            if _var.name == item:
                return _var

        raise KeyError('No input: {}'.format(item))

    def trim(self, start, end):
        for _var in self.inputs:
            _var._df.drop(_var._df.index[_var._df.index < start], inplace=True)
            _var._df.drop(_var._df.index[_var._df.index > end], inplace=True)
            _var.flag.trim(start, end)

    def add_input(self, var):
        if var.name not in [i.name for i in self.inputs]:
            self.inputs.append(var)
            return

        try:
            self[var.name]._df.loc[
                var._df.index, var.name
            ] = var._df[var.name]

        except KeyError:
            # The dataframe does not include all of the indicies present in
            # this variable, therefore we need to reindex

            # Create the new index as the unique union between the
            # dataframe and the variable. 
            _index = self[var.name].index.union(
                var.index
            ).sort_values().unique()

            # Reindex the dataframe
            _df = self[var.name]._df.reindex(_index)
            self[var.name]._df = _df

            # And merge in the variable
            self[var.name]._df.loc[
                var._df.index, var.name
            ] = var._df[var.name]


    def collect_garbage(self, required_inputs):
        _copied_inputs = self.inputs.copy()

        for var in self.inputs:
            if var.name not in required_inputs:
                self.inputs.remove(var)
                print('GC: {}'.format(var))
                del var

        gc.collect()


class PandasConsolidatedInMemoryBackend(DecadesBackend):

    def __init__(self):
        self._dataframes = {}
        super().__init__()

    def __getitem__(self, item):
        for _var in self.inputs:
            if _var.name == item:
                if _var._df is None:
                     _var._df = self._dataframes[
                         self._dlu_from_variable(_var)
                     ][_var.frequency][[_var.name]].copy()

                return _var

        for _var in self.outputs:
            if _var.name == item:
                return _var

        raise KeyError('No input: {}'.format(item))

    def trim(self, start, end):
        for key, value in self._dataframes.items():
            dlu = self._dataframes[key]
            for key, value in dlu.items():
                df = dlu[key]

                df.drop(df.index[df.index < start], inplace=True)
                df.drop(df.index[df.index > end], inplace=True)

        self.decache(hard=True)

    def add_input(self, variable):
        _var = None

        _dlu = self._dlu_from_variable(variable)
        _freq = variable.frequency

        # If there are entries with duplicate timestamps, only accept the final
        # one
        variable._df = variable._df.groupby(variable._df.index).last()

        if _dlu not in self._dataframes:
            self._dataframes[_dlu] = {}

        if variable.frequency not in self._dataframes[_dlu]:
            # The dataframe for the current DLU at the current frequency does
            # not yet exist, so we need to create it, using the variable
            # dataframe
            self._dataframes[_dlu][_freq] = variable._df
            self.inputs.append(variable)

        else:
            # The dataframe does exist, so attempt to merge into it, assuming
            # that the index in the variable is covered by that in the
            # dataframe 

            if variable.name not in self._dataframes[_dlu][_freq].columns:
                self.inputs.append(variable)

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

                self.inputs.append(variable)

        # Once the variable has been merged into a dataset dataframe, it no
        # longer needs to maintain its own internal dataframe
        self.decache()

    def decache(self, variables=None, hard=False):
        for var in (variables or self.inputs):
            if var._df is None:
                continue

            if not hard:
                _dlu = self._dlu_from_variable(var)
                _freq = var.frequency
                _df = var._df

                self._dataframes[_dlu][_freq].loc[
                    _df.index, var.name
                ] = var._df.loc[_df.index, var.name]

            var._df = None
        gc.collect()

    def collect_garbage(self, required_inputs):
        _copied_inputs = self.inputs.copy()

        for var in _copied_inputs:
            if var.name not in required_inputs:
                try:
                    _dlu = self._dlu_from_variable(var)
                except Exception:
                    continue

                _freq = var.frequency

                self._dataframes[_dlu][_freq].drop(
                    var.name, axis=1, inplace=True, errors='ignore'
                )

                self.inputs.remove(var)
                print('GC: {}'.format(var))
                del var

        del _copied_inputs
        self.decache()

class DecadesBackend(object):

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def _dlu_from_variable(self, variable):
        return variable.name.split('_')[0]

    @property
    def variables(self):
        return [i.name for i in self.inputs + self.outputs]


class PandasInMemoryBackend(DecadesBackend):

    def __init__(self):
        self._dataframes = {}
        super(PandasInMemoryBackend, self).__init__()

    def __getitem__(self, item):
        for _var in self.inputs:
            if _var.name == item:
                if _var._df is None:
                     _var._df = self._dataframes[
                         self._dlu_from_variable(_var)
                     ][_var.frequency][[_var.name]]

                return _var

        raise KeyError('No input: {}'.format(item))

    def add_input(self, variable):
        _var = None

        _dlu = self._dlu_from_variable(variable)
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

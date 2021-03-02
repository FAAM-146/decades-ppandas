"""
This module provides backends for use by DecadesDataset. Initially this was to
allow for different memory optimization strategies, however a refactoring of
the way variables are stored internally has negated the need for this. The only
backend now provided is DefaultBackend.
"""
# pylint: disable=useless-object-inheritance
import gc


class DecadesBackend(object):
    """
    DecadesBackend is essentially an abstract class for a backend to a
    DecadesDataset. The requirement for a Backend is depreciated now, really,
    but kept for prosperity.
    """

    def __init__(self):
        """
        Initialize the backend.
        """
        self.inputs = []
        self.outputs = []

    @staticmethod
    def _dlu_from_variable(variable):
        """
        Get the name of a DLU from a variable name. This is essentially the
        first token of a string, when split with an underscore delimiter.

        Args:
            variable: A string containing the name of a variable from which to
                      extract the name of the DLU.

        Returns:
            The name of the DLU responsible for <variable>.
        """
        return variable.name.split('_')[0]

    def decache(self):
        """
        Offload variables from instance state to some other storage solution.
        """

    def collect_garbage(self, required_inputs):
        """
        Clear up any variables which are no longer required for the processing
        job.

        Args:
            required_inputs: an array of variable names which are still
            required for the processing job.
        """

    def cleanup(self):
        """
        Perform any cleanup required by the backend.
        """

    def trim(self, start, end):
        """
        Trim any variables, removing any data before start or after end.
        """

    def add_input(self, var):
        """
        Add in input - a variable created from reading input data - to the
        backend.

        Args:
            var: the input variable to add.
        """
        raise NotImplementedError

    def add_output(self, variable):
        """
        Add an output - a variable created during processing.

        Args:
            variable: the output variable to add.
        """
        self.outputs.append(variable)

    def remove(self, name):
        """
        Remove a variable from the backend.

        Args:
            name: the name of the variable to remove.
        """
        raise NotImplementedError

    def clear_outputs(self):
        """
        Clear all output variables from the backend.
        """
        self.outputs = []

    @property
    def variables(self):
        """
        Return a list of the names of all of the variables (both inputs and
        outputs) contained on the backend.
        """
        return [i.name for i in self.inputs + self.outputs]


class DefaultBackend(DecadesBackend):
    """
    The DefaultBackend, subclassing DecadesBackend, is the default backend to
    use for processing if no other backend is specified.
    """

    def __getitem__(self, item):
        """
        Implement []. Given the name of a variable, return that variable if it
        exists as and input or output in the backend.

        Args:
            item: the name of the variable to retreive.
        """

        for _var in self.inputs + self.outputs:
            if _var.name == item:
                return _var

        raise KeyError('No input: {}'.format(item))

    def trim(self, start, end):
        """
        Trim all of the input variables. That is, remove all data from before
        start or after end.

        Args:
            start: a datetime like which indicates the start time of the data
                   to keep.
            end: a datetime like which indicates the end time of the data to
                 keep.
        """
        for _var in self.inputs:
            _var.trim(start, end)

    def remove(self, name):
        """
        Remove a variable from the backend.

        Args:
            name: the name of the variable to remove.
        """
        for var in self.inputs:
            if var.name == name:
                self.inputs.remove(var)
                return
        for var in self.outputs:
            if var.name == name:
                self.outputs.remove(var)
                return

    def add_input(self, var):
        """
        Add an input to the backend, attempting to merge if an input with the
        name name is already present.

        Args:
            var: the variable, nominally a DecadesVariable, to add to the
                 backend.
        """

        if var.name not in [i.name for i in self.inputs]:
            self.inputs.append(var)
            return

        self[var.name].merge(var)

    def collect_garbage(self, required_inputs):
        """
        Remove any variables which are no longer required by the processing.
        Forces interpreter garbage collection, which is probably overkill.

        Args:
            required_inputs: an iterable of names of variables which are still
                             required for the processing, and so should not be
                             removed during garbage collection.
        """
        _copied_inputs = self.inputs.copy()

        for var in self.inputs:
            if var.name not in required_inputs:
                self.inputs.remove(var)
                print('GC: {}'.format(var))
                del var

        gc.collect()

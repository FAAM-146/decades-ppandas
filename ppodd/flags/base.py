"""
Provides an abstract base class for flagging modules. Flagging modules run
after the main processing modules allowing modification to data quality flags
using data which may not be available at the time processing modules are run.
"""

import abc

class FlaggingBase(abc.ABC):
    """
    An abstract base class from which flagging modules should inherit.
    """

    # Names of required input variables
    inputs = []
    prerequisites = []

    def __init__(self, dataset):
        """
        Initialize an instance.

        Args:
            dataset: a DecadesDataset from which the processing is being run.
        """
        self.dataset = dataset

    def ready(self, run_modules):
        """
        Decide if the module is ready to run. It can be run if all of its
        inputs are available in the parent dataset.

        Returns:
            True if the module is ready to run, False otherwise.
        """
        for _input in self.inputs:
            if _input not in self.dataset.variables:
                print(f'{_input} not in {self.dataset.variables}')
                return False
        for prereq in self.prerequisites:
            if prereq not in run_modules:
                print(prereq, ' not in ', run_modules)
                return False
        return True

    @abc.abstractmethod
    def flag(self):
        """Add extra flag info to derived variables."""

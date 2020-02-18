import abc

class FlaggingBase(abc.ABC):
    inputs = []

    def __init__(self, dataset):
        self.dataset = dataset

    def ready(self):
        for _input in self.inputs:
            if _input not in self.dataset.inputs + self.dataset.outputs:
                return False
        return True

    @abc.abstractmethod
    def flag(self):
        """Add extra flag info to derived variables."""

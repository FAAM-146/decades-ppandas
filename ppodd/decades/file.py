from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from ppodd.decades.dataset import DecadesDataset


class DecadesFile(object):
    """
    A DecadesFile is just a wrapper around a filepath. Factored out for
    potential future use, but currently isn't really doing anything useful.
    """

    def __init__(self, filepath: str) -> None:
        """
        Initialize an instance.

        Args:
            filepath (str): a string giving the absolute path to a file.
        """
        self.filepath = filepath
        self.dataset: DecadesDataset | None = None

    def __repr__(self) -> str:
        """
        Return a string representation of the instance.
        """
        return '{}({!r})'.format(
            self.__class__.__name__, self.filepath
        )
    
    def set_dataset(self, dataset: DecadesDataset) -> None:
        """
        Set the dataset for this file.

        Args:
            dataset: the dataset to set.
        """
        self.dataset = dataset
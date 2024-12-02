from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ppodd.decades import DecadesDataset


class Processor(Protocol):
    """
    Protocol for a Processor class.
    """
    def __init__(self, dataset: 'DecadesDataset') -> None:
        ...

    def process(self, modname: str | None) -> None:
        ...
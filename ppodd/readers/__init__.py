from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .readers import FileReader

reader_patterns: dict[str, type['FileReader']] = {}


def register(patterns: list[str] | None = None) -> Callable:
    # TODO: This should be a generic decorator
    def _register(f: type['FileReader']) -> type['FileReader']:
        nonlocal patterns
        if patterns is None:
            patterns = []
        for pattern in patterns:
            reader_patterns[pattern] = f
        return f

    return _register


from ppodd.readers.readers import *
from ppodd.readers.fgga import FGGAReader
from ppodd.readers.sea.reader import WcmFileReader
from ppodd.readers.ccn import CCNReader

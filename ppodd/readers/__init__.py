reader_patterns = {}


def register(patterns=None):
    def _register(f):
        for pattern in patterns:
            reader_patterns[pattern] = f
        return f
    return _register

from ppodd.readers.readers import *
from ppodd.readers.sea.reader import WcmFileReader

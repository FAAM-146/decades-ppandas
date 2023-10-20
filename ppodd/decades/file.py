class DecadesFile(object):
    """
    A DecadesFile is just a wrapper around a filepath. Factored out for
    potential future use, but currently isn't really doing anything useful.
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, filepath):
        """
        Initialize an instance.

        Args:
            filepath (str): a string giving the absolute path to a file.
        """
        self.filepath = filepath

    def __repr__(self):
        return '{}({!r})'.format(
            self.__class__.__name__, self.filepath
        )
import logging

logger = logging.getLogger(__name__)

class Lazy(object):
    """
    A hacky deferral wrapper for assigning dataset constants to processing
    module outputs potentially before they've actually been set on the dataset.
    This works in this context as variable metadata are allowed to be either
    literal or callable.
    """

    def __init__(self, parent):
        """
        Initialize a class instance.

        Args:
            parent: the object whos state we want to defer.
        """
        self.parent = parent

    def __getitem__(self, item):
        """
        Implement [x], potentially deferred to a callable.

        Args:
            item: the item to get from the parent.

        Returns:
            either <item> got from parent, or a callable deferring this
            operation.
        """
        def _callable():
            try:
                return self.parent[item]
            except KeyError:
                return None

        try:
            return self.parent[item]
        except KeyError:
            return _callable

    def __getattr__(self, attr):
        """
        Implement .x, potentially deferred to a callable.

        Args:
            attr: the attribute to get from the parent.

        Returns:
            either <attr> from parent, or a callable deferring this
            operation.
        """
        try:
            return getattr(self.parent, attr)
        except AttributeError:
            return lambda: getattr(self.parent, attr)

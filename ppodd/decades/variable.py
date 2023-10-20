import re

import numpy as np
import pandas as pd

from vocal.types import DerivedFloat32

from ppodd.decades.flags import DecadesClassicFlag
from ppodd.decades.attributes import (
    AttributesCollection, Attribute, ATTR_USE_EXAMPLE, Context
)
from ppodd.utils import pd_freq, infer_freq


class DecadesVariable(object):
    """
    A DecadesVariable is a container for a timeseries, a corresponding data
    quality flag and associated Metadata, in the form of an AttributeCollection
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize a class instance. Arbitrary args and kwargs are accepted.
        These will be passed to pandas during the initial creation of a
        DataFrame, other than keys which are defined in the `standard` and:

        Args:
            name (str, optional): the name of the variable
            write (bool, optional): whether or not this variable should be
                written to file as an output. Default `True`.
            flag (Object, optional): a class to be used for holding data quality
                information. Default is `DecadesClassicFlag`.
            standard (str, optional): a metadata 'standard' which should
                be adhered to. Default is `ppodd.standard.core`.
            strict (bool, optional): whether the `standard` should be strictly
                enforced. Default is `True`.
            tolerance (int, optional): tolerance to use when reindexing onto a
                regular index.
            flag_postfix (str, optional): the postfix to add to a
                variable when output to indicate it is a quality flag. Default
                is `FLAG`.
            doc_mode (bool, optional): If true, put the variable in documentation
                mode, which may affect the value of attributes
        """

        _flag = kwargs.pop('flag', DecadesClassicFlag)
        _standard = kwargs.pop('standard', 'faam_data')
        _strict = kwargs.pop('strict', False)
        _tolerance = kwargs.pop('tolerance', 0)
        _flag_postfix = kwargs.pop('flag_postfix', 'FLAG')

        self.attrs = AttributesCollection(
            dataset=self, definition='.'.join((_standard, 'VariableAttributes')),
            strict=_strict
        )
        self.dtype = kwargs.pop('dtype', None)
        self.name = kwargs.pop('name', None)
        self.write = kwargs.pop('write', True)
        self.doc_mode = kwargs.pop('doc_mode', False)

        # Set attributes given as keyword arguments
        _attrs = self.attrs.REQUIRED_ATTRIBUTES + self.attrs.OPTIONAL_ATTRIBUTES
        for _attr in _attrs:
            # Get the default value of an attribute, if it exists
            try:
                _default = self.attrs._definition.schema()['properties'][_attr]['ppodd_default']
            except KeyError:
                _default = None

            # Pop the attribute off the keyword stack, and set it if it has a
            # value

            _context = None
            _context_type = None

            if _default:
                try:
                    rex = re.compile('^<call (.+)>$')
                    hook = rex.findall(_default)[0]
                    _default = [i.strip() for i in hook.strip().split()][0]
                    _context = self
                    _context_type = Context.ATTR
                except (TypeError, IndexError):
                    pass

            _val = kwargs.pop(_attr, _default)
            if _val is not None:
                self.attrs.add(
                    Attribute(_attr, _val, context=_context, context_type=_context_type)
                )

        # Create an interim DataFrame, and infer its frequency
        _df = pd.DataFrame(*args, **kwargs)
        _freq = self._get_freq(df=_df)

        # Create an index spanning the variable value at the correct frequency
        _index = pd.date_range(
            start=_df.index[0], end=_df.index[-1],
            freq=_freq
        )

        # If no variable name is given, infer it from the first column of the
        # dataframe
        if self.name is None:
            self.name = _df.columns[0]

        # Deal with non-unique entries in the dataframe, by selecting the last
        # element
        if len(_df.index) != len(_df.index.unique()):
            _df = _df.groupby(_df.index).last()

        # Ensure input is monotonic
        _df = _df.sort_index()

        # Create the data array that we're going to keep. We're going to
        # reindex the dataframe onto the complete index, and downcast it to the
        # smallest reasonable datatype. This is a memory saving trick, as we
        # don't have to have a 64-bit index associated with every variable.
        array = self._downcast(np.array(
            _df.reindex(
                _index, tolerance=_tolerance, method='nearest', limit=1
            ).values.flatten()
        ))
        if self.dtype:
            array = array.astype(self.dtype)
        self.array = array

        # t0 and t1 are the start and end times of the array, which we're going
        # to store as we dispose of the index
        self.t0 = _index[0]
        self.t1 = _index[-1]

        # Create the QC Flag array, and add the name of the flag variable to
        # the 'ancillary_variables' attribute. TODO: should we check if this
        # attribute already exists?
        if _flag is not None:
            self.flag = _flag(self, postfix=_flag_postfix)
            self.attrs.add(
                Attribute(
                    'ancillary_variables', f'{self.name}_{_flag_postfix}'
                )
            )
        else:
            if self.name.endswith('_CU'):
                self.attrs.add(
                    Attribute(
                        'ancillary_variables',
                        f'{self.name[:-3]}_{_flag_postfix}'
                    )
                )
            self.flag = _flag

    def __call__(self):
        """
        Implement (). When a class instance is called, create and return a
        Pandas Series with the correct index.
        """
        i = pd.date_range(
            start=self.t0, end=self.t1,
            freq=self._get_freq()
        )

        kwargs = {}
        if self.dtype:
            kwargs['dtype'] = self.dtype

        return pd.Series(self.array, index=i, name=self.name, **kwargs)

    def __len__(self):
        """
        Impement len(). The length of a variable is the length of its array
        attribute.
        """
        return len(self.array)

    def __getattr__(self, attr):
        """
        Implement .<attr>.
        """
        # pylint: disable=too-many-return-statements

        if attr == 'index':
            return self().index

        if attr == 'data':
            return self()

        if attr == 'flag':
            return self.flag

        if attr in self.attrs.keys:
            return self.attrs[attr]

        try:
            return getattr(self(), attr)
        except (KeyError, AttributeError):
            pass

        raise AttributeError(f'Not a variable attribute: {attr}')

    def __setattr__(self, attr, value):
        """
        Manage setting of attributes.

        Kwargs:
            attr: the name of the attribute to set
            value: the value of the attribute to set
        """
        if attr == 'attrs':
            super().__setattr__(attr, value)

        if attr in (
            self.attrs.REQUIRED_ATTRIBUTES + self.attrs.OPTIONAL_ATTRIBUTES
        ):
            self.attrs[attr] = value

        super().__setattr__(attr, value)

    def __str__(self):
        """
        Implement str()
        """
        return self.name

    def __repr__(self):
        """
        Implement repr()
        """
        return r'<DecadesVariable[{!r}]>'.format(self.name)

    def _get_freq(self, df=None):
        """
        Return the frequency of the variable.

        Args:
            df (pd.DataFrame): if given, infer the frequency of this dataframe.

        Returns:
            int: the frequency code of the variable.
        """
        try:
            return pd_freq[self.attrs['frequency']]
        except (KeyError, AttributeError):
            _freq = pd.infer_freq(df.index)

        if _freq is None:
            _freq = infer_freq(df.index)

        if len(_freq) == 1:
            _freq = f'1{_freq}'

        self.frequency = int(1/pd.to_timedelta(_freq).total_seconds())
        return _freq

    @staticmethod
    def _downcast(array):
        """
        Downcast a numeric array to its smallest compatable type, via
        pd.to_numeric.

        Args:
            array (np.array): the numpy array, or pd.Series to downcast.

        Returns:
            np.array: a downcast copy of array, or array if it cannot be
            downcast.
        """
        dc = 'float'
        try:
            if np.all(array[np.isfinite(array)] == array[np.isfinite(array)].astype(int)):
                dc = 'integer'
        except (ValueError, TypeError):
            pass

        try:
            return pd.to_numeric(array, downcast=dc)
        except (ValueError, TypeError):
            pass

        return array

    def _merge_fast(self, other):
        """
        Merge this variable with another variable, assumed to be the same data
        stream over a different time interval. This fast merge assumes that the
        <other> variable occurs after this one in time, and that the
        intersection of indicies is the empty set.

        Args:
            other (:obj:`DecadesVariable`): the variable to merge with this one.
        """

        # Create a union of data and indicies
        _data = np.concatenate([self.array, other.array])
        _index = self.index.union(other.index)

        # Reindex to ensure no gaps in the data
        _df = pd.DataFrame(_data, index=_index).reindex(
            pd.date_range(start=_index[0], end=_index[-1],
                          freq=pd_freq[self.frequency]),
            tolerance='{}N'.format((.5/self.frequency)*1e9),
            method='nearest'
        )

        # Store the merged data
        array = _df.values.flatten()
        if self.dtype:
            array = array.astype(self.dtype)

        self.array = array
        self.t0 = _df.index[0]
        self.t1 = _df.index[-1]

    @property
    def isnumeric(self):
        """
        bool: True if the datatype of the variable array is numeric, False
        otherwise
        """
        return np.issubdtype(self.array.dtype, np.number)

    def trim(self, start, end):
        """
        'Trim' the variable to a subset of itself, via a top and tail. The
        interval is closed (i.e. start and end will remain in the variable).

        Args:
            start: a `datetime` like indicating the start of the period to keep
            end: a `datetime` like indicating the end of the period to keep.
        """

        # Create a dataframe, index to the required interval, and extract the
        # required attributes to store.
        _df = self()
        loc = (_df.index >= start) & (_df.index <= end)
        trimmed = _df.loc[loc]

        try:
            self.array = trimmed.values.flatten()
            self.t0 = trimmed.index[0]
            self.t1 = trimmed.index[-1]
        except Exception:
            return

        # Trim the QC flag over the same interval.
        if self.flag is not None:
            self.flag.trim(start, end)

    def merge(self, other):
        """
        Merge another variable, assumed to represent the same data field over a
        different period into this one.

        Args:
            other (DecadesVariable): the variable to merge into this one.
        """

        # If the other variable is after this one, we can merge fast...
        if other.t0 > self.t1:
            self._merge_fast(other)
            return

        # ...otherwise we have to use a slower method
        other = other()
        current = self()

        # Create a union of the indexes of both variables
        merge_index = current.index.union(other.index).sort_values().unique()

        # Reindex this variable onto the merged index...
        current = current.reindex(merge_index)
        # ...and merge in the other variable
        current.loc[other.dropna().index] = other

        # Reindex to ensure there aren't any data gaps
        full_index = pd.date_range(
            start=merge_index[0], end=merge_index[-1],
            freq=pd_freq[self.frequency]
        )

        current = current.reindex(
            full_index,
            tolerance='{}N'.format((.5/self.frequency)*1e9),
            method='nearest'
        )

        # Store the required attributes
        array = current.values.flatten()
        if self.dtype:
            array = array.astype(self.dtype)
        self.array = array
        self.t0 = current.index[0]
        self.t1 = current.index[-1]

    def range(self):
        if self.doc_mode:
            return [DerivedFloat32, DerivedFloat32]

        try:
            if not self.isnumeric:
                return None
            if not np.any(np.isfinite(self.array)):
                return None
            return [
                np.nanmin(self.array).astype(np.float32),
                np.nanmax(self.array).astype(np.float32)
            ]
        except Exception:
            return None

    def time_bounds(self):
        """
        Return the start and end times of this variable, as the tuple (start,
        end).

        Returns:
            tuple: A 2-tuple, (start, end), containing the start and end times
            of this variable.
        """
        return (self.t0, self.t1)
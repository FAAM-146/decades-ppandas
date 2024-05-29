"""
This module provides classes which are used to implement different strategies
for QC flagging of data variables.
"""
# pylint: disable=useless-object-inheritance, invalid-name
import numpy as np
import pandas as pd

from ppodd.decades.utils import resample_variable
from ppodd.utils import pd_freq

__all__ = ('DecadesClassicFlag', 'DecadesBitmaskFlag')

REPLACE = 'replace'
MAXIMUM = 'maximum'

CALIBRATION = 'in_calibration'
DATA_GOOD = 'data_good'
DATA_MISSING = 'data_missing'
OUT_RANGE = 'data_out_of_range'
WOW = 'aircraft_on_ground'
DEPENDENCY = 'dependency_is_flagged'


class DecadesFlagABC(object):
    """
    Almost Abstract Base Class for Decades Flagging.
    """

    def __init__(self, var, postfix='FLAG'):
        """
        Initialisation

        Args:
            var: the DecadesVariable that this flag is attached to.
        """

        self._df = pd.DataFrame(index=range(len(var.data)),
                                dtype=np.dtype('int8'))
        self._var = var
        self.t0 = var.t0
        self.t1 = var.t1
        self.frequency = var.frequency
        self.postfix = postfix
        self._long_name = f'Data quality flag for {var.name}'
        self.descriptions = {}

    @property
    def index(self):
        """
        Return the index associated with the flag data.
        """
        return pd.date_range(
            start=self.t0, end=self.t1, freq=pd_freq[self.frequency]
        )

    @property
    def df(self):
        """
        Return a copy of a dataframe representing the internal state of the
        flag.
        """
        _df = self._df.copy()
        _df.index = self.index
        return _df

    def description(self, flag_name_or_val):
        """
        Add a more verbose, human readable description to each flag name or
        value.

        Args:
            flag_name_or_val: the flag name for e.g. Bitmask type flags, or
                              value for classic type flags.
        """
        try:
            return self.descriptions[flag_name_or_val]
        except KeyError:
            return None

    def trim(self, start, end):
        """
        Drop any flagging data outside specified time bounds.

        Args:
            start: the minumum valid time
            end: the maximum valid time
        """
        _index = self.index
        loc = (_index >= start) & (_index <= end)

        _df = self._df.copy(deep=True)
        _df.index = _index
        _df = _df.loc[loc]
        _df.index = range(len(_df.index))
        self._df = _df
        self.t0 = start
        self.t1 = end

    @property
    def cfattrs(self):
        """
        Return a dict of flag attributes for cf compliant netCDF files.
        """


class DecadesClassicFlag(DecadesFlagABC):
    """
    DecadesClassicFlag: a flag for the traditional DECADES flagging strategy.
    That is, integer flag values with increasingly large values generally
    associated with lower quality data.
    """

    def __init__(self, var, *args, **kwargs):
        """
        Initialisation overide.

        Args:
            var: the DecadesVariable that this flag is associated with.
        """
        super().__init__(var, *args, **kwargs)

        # Initialize the flag to -128, a fill_value
        self._df['FLAG'] = np.int8(-128)
        self.descriptions[-128] = ('A fill value. No flagging information '
                                   'has been provided')
        self.descriptions[0] = ('Data are assumed to be valid and '
                                'representative of the physical quantity '
                                'measured')

        # The meanings of each flag value. If no meanings are defined, no
        # flagging is assumed to have taken place.
        self.meanings = {}

    def __call__(self):
        """
        Return flag values when the instance is called.
        """
        if not self._var._forced_frequency:
            return self.to_series()
        
        return resample_variable(self, self._var.frequency, apply='max')
    
    def to_series(self):
        """
        Return the flag as a pandas Series.
        """
        s = pd.Series(self._df['FLAG'])
        s.index = self.index
        return s

    @property
    def cfattrs(self):
        """
        Implement the cfattrs getter. Returns a dict of attributes which should
        be added to the netCDF flag variable for cf compliance.
        """
        _cfattrs = {
            'coverage_content_type': 'qualityInformation'
        }

        if self.meanings:
            if 0 in self.meanings:
                _meanings = self.meanings
            elif np.any(self() != -128):
                _meanings = {0: DATA_GOOD}
                _meanings.update(self.meanings)
            else:
                _meanings = self.meanings
        else:
            _meanings = {-128: 'data_not_flagged'}

        # If the variable we're flagging has a standard name, then we use that
        # along with status_flag. Otherwise just use status_flag
        if getattr(self._var.attrs, 'standard_name', None):
            _cfattrs['standard_name'] = '{} status_flag'.format(
                self._var.standard_name
            )
        else:
            _cfattrs['standard_name'] = 'status_flag'

        _cfattrs['long_name'] = self._long_name
        _cfattrs['_FillValue'] = int(-128)
        _cfattrs['flag_values'] = [
            np.int8(i) for i in sorted(_meanings.keys())
        ]
        _cfattrs['flag_meanings'] = ' '.join(
            _meanings[i] for i in _cfattrs['flag_values']
        )

        _cfattrs['frequency'] = self.frequency
        _cfattrs['units'] = '1'

        return _cfattrs

    def add_meaning(self, value, meaning, description=None):
        """
        Add a flag meaning.

        Args:
            value: the value of a flag to assign a meaning
            meaning: a string describing the cause of flag value value
        """

        if value == 0 and meaning.replace(' ', '_').lower() != 'data_good':
            raise ValueError('Flag of zero must mean data_good')

        self.meanings[value] = meaning.replace(' ', '_').lower()
        self.descriptions[value] = description

    def add_flag(self, flag, method=MAXIMUM):
        """
        Add an array to the flag. Can either be merged with the current flag
        (through a elementwise max) or can replace the current flag values
        entirely.

        Args:
            flag: an iterable of the correct length containing flagging values.

        Kwargs:
            method: one of ppodd.decades.flags.MAXIMUM,
                    ppodd.decades.flags.REPLACE, defining the strategy for
                    adding the values to the flag.
        """
        flag = np.atleast_1d(flag)

        if len(flag) != len(self._df.index):
            print(f'{len(flag)} != {len(self._df.index)}')
            raise ValueError('Flag length is incorrect')

        if np.any(flag > np.atleast_1d(np.max(list(self.meanings.keys())))):
            raise ValueError('Flag value given has not been defined')

        if method == MAXIMUM:
            self._df.FLAG = np.maximum(
                np.array(self._df.FLAG.values), np.atleast_1d(flag)
            )
        else:
            self._df.FLAG = np.atleast_1d(flag)

        self._df.FLAG.loc[self._df.FLAG < 0] = -128

    @classmethod
    def from_nc_variable(cls, var, decadesvar):
        """
        Generate a flag variable from a netcdf variable.

        Args:
            var: the netCDF variable
            decadesvar: the correcponding decades variable.
        """
        flag = cls(decadesvar)

        _standard_meanings = (
                'data_good', 'possible_minor_issues', 'possible_major_issues',
                'data_bad_or_missing'
            )
        try:
            for meaning, value in zip(
                var.flag_meanings.split(), np.atleast_1d(var.flag_values)
            ):
                flag.add_meaning(value, meaning)
        except AttributeError:
            for meaning, value in zip(_standard_meanings, (0, 1, 2, 3)):
                flag.add_meaning(value, meaning)

        return flag


class DecadesBitmaskFlag(DecadesFlagABC):
    """
    DecadesBitmaskFlag. Defines a strategy that allows multiple mask (boolean)
    flags to be used in a single flag variable.
    """

    def __call__(self):
        """
        When an instance is called, build the flag from the mask values and
        return it.
        """
        if not self._var._forced_frequency:
            return self.to_series()
        
        return resample_variable(self, self._var.frequency, apply='max')
    
    def to_series(self):
        _meanings = self.meanings

        _masks = self.masks

        _flag_vals = np.zeros((len(self._df.index),))

        for i, meaning in enumerate(_meanings):
            try:
                _flag_vals += _masks[i] * np.array(self._df[meaning]).astype(int)
            except Exception:
                print(_flag_vals)
                print(_masks[i])
                raise

        _flag_vals = _flag_vals.astype(np.int8)

        return pd.Series(_flag_vals, index=self.index)
        
    @property
    def df(self):
        _df = self._df.copy()
        _df.index = self.index
        return _df

    @property
    def meanings(self):
        return list(self._df.columns)

    @property
    def masks(self):
        """
        Return an array containing flag_mask values. Canonically, this is an
        array of 2**n for integer n in 0 .. #masks.
        """
        return [int(2**i) for i in range(len(self._df.columns))]

    @property
    def cfattrs(self):
        """
        Implement the cfattrs getter. Returns a dict of attributes which should
        be added to the netCDF flag variable for cf compliance.
        """
        _valid_range_max = 2 * int(self.masks[-1]) - 1 if self.masks else 0
        _valid_range_min = 1 if self.masks else 0

        return {
            'long_name': self._long_name,
            '_FillValue': 0,
            'valid_range': [_valid_range_min, _valid_range_max],
            'flag_masks': [np.int8(i) for i in self.masks],
            'flag_meanings': ' '.join(self._df.columns),
            'coverage_content_type': 'qualityInformation',
            'frequency': self.frequency,
            'units': '1'
        }

    def add_mask(self, data, meaning, description=None):
        """
        Add a mask array to the flag.

        Args:
            data: the flag data, assumed to be a mask, and will be cast to a
                  boolean
            meaning: the meaning/description associated with the mask.
        """

        col_name = meaning.replace(' ', '_').lower()

        if col_name in self._df:
            return self._add_to_mask(data, col_name)

        if len(data) != len(self._df.index):
            print(f'{len(data)} != {len(self._df.index)}')
            raise ValueError('Flag length is incorrect')

        self._df[col_name] = np.atleast_1d(data).astype(bool)
        self.descriptions[col_name] = description

    def _add_to_mask(self, data, col_name):
        data = data.reindex(self.index).fillna(0)
        data.index = range(len(data.index))
        self._df.loc[data > 0, col_name] = data[data > 0]

    @classmethod
    def from_nc_variable(cls, ncvar, decadesvar):
        """
        Generate a flag variable from a netcdf variable.

        Args:
            var: the netCDF variable
            decadesvar: the correcponding decades variable.
        """

        flag = cls(decadesvar)
        masks = np.atleast_1d(ncvar.flag_masks)
        meanings = np.atleast_1d(ncvar.flag_meanings.split())

        _data = ncvar[:].ravel().data
        _flags = []

        for mask, meaning in zip(masks[::-1], meanings[::-1]):
            _flag_data = (_data >= mask)
            _flags.insert(0, _flag_data)
            _data[_flag_data] -= mask

        for _flag, meaning in zip(_flags, meanings):
            flag.add_mask(_flag, meaning)

        return flag

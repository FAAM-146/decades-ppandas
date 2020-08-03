import numpy as np
import pandas as pd


__all__ = ('DecadesClassicFlag', 'DecadesBitmaskFlag')

REPLACE = 'replace'
MAXIMUM = 'maximum'

CALIBRATION = 'in_calibration'
DATA_GOOD = 'data_good'
DATA_MISSING = 'data_missing'
OUT_RANGE = 'data_out_of_range'
WOW = 'aircraft_on_ground'


class DecadesFlagABC(object):
    """
    Almost Abstract Base Class for Decades Flagging.
    """

    def __init__(self, var):
        """
        Initialisation

        Args:
            var: the DecadesVariable that this flag is attached to.
        """

        _index = var._df.index
        self._df = pd.DataFrame(index=_index)
        self._var = var
        self._long_name = f'Flag for {var.name}'
        self.descriptions = {}

    def description(self, flag_name_or_val):
        """
        Add a more verbose, human readable description to each flag name or
        value.

        Args:
            flag_name_or_val: the flag name for e.g. Bitmask type flags, or
                              value for classic type flags.
        """
        try:
            return self.descriptions[flag_or_val]
        except KeyError:
            return None

    def trim(self, start, end):
        """
        Drop any flagging data outside specified time bounds.

        Args:
            start: the minumum valid time
            end: the maximum valid time
        """

        loc = (self._df.index >= start) & (self._df.index <= end)
        self._df = self._df.loc[loc]

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

    def __init__(self, var):
        """
        Initialisation overide.

        Args:
            var: the DecadesVariable that this flag is associated with.
        """
        super(DecadesClassicFlag, self).__init__(var)

        # Initialize the flag to -128, a fill_value
        self._df['FLAG'] = -128
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
        return self._df.FLAG

    @property
    def cfattrs(self):
        """
        Implement the cfattrs getter. Returns a dict of attributes which should
        be added to the netCDF flag variable for cf compliance.
        """
        _cfattrs = {}

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
        if getattr(self._var, 'standard_name', None):
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

        _meanings = self._df.columns

        _masks = self.masks

        _flag_vals = np.zeros((len(self._df.index),))

        for i, meaning in enumerate(_meanings):
            _flag_vals += _masks[i] * self._df[meaning]

        return pd.Series(_flag_vals.astype(np.int8), index=self._df.index)

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
        return {
            'long_name': self._long_name,
            '_FillValue': 0,
            'valid_range': [1, 2 * int(self.masks[-1]) - 1],
            'flag_masks': [np.int8(i) for i in self.masks],
            'flag_meanings': ' '.join(self._df.columns)
        }

    def add_mask(self, data, meaning, description=None):
        """
        Add a mask array to the flag.

        Args:
            data: the flag data, assumed to be a mask, and will be cast to a
                  boolean
            meaning: the meaning/description associated with the mask.
        """

        if len(data) != len(self._df.index):
            raise ValueError('Flag length is incorrect')

        col_name = meaning.replace(' ', '_').lower()

        self._df[col_name] = np.atleast_1d(data).astype(bool)
        self.descriptions[col_name] = description

    @classmethod
    def from_nc_variable(cls, ncvar, decadesvar):
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

import numpy as np
import pandas as pd

pd_freq = {
    1: '1S',
    2: '500L',
    4: '250L',
    8: '125L',
    10: '100L',
    16: '62500U',
    20: '50L',
    32: '31250000N',
    64: '15625000N',
    128: '7812500N',
    256: '3906250N'
}


def get_range_flag(var, limits, flag_val=2):
    """
    Get a flag variable which flags when a variable is outside a specified
    range.

      Args:
          var: the variable on which to base the flagging, np.array-like.
          limits: a 2-element array specifying valid-min and valid-max.

      Kwargs:
          flag_val: the value of the flag when var is outside limits. Default
                    2.

      Returns:
          flag: a flag [np.array] of same dimensions as var, with value 0
                wherever var in inside limits, and flag_val where are is
                outside limits.
    """
    flag = np.zeros_like(var)

    flag[var < limits[0]] = flag_val
    flag[var > limits[1]] = flag_val

    return flag


def flagged_avg(df, flag_col, data_col, fill_nan=None, flag_value=1,
                skip_start=0, skip_end=0, out_name=None, interp=False):
    """
    Average a variable where a corresponding flag has a given value

    Args:
        df: the DataFrame to operate on
        flag_col: the name of the column of df which contains the flag variable
        data_col: the name of the column of df which contains the data variable

    Kwargs:
        fill_nan: the value with which to fill any NaNs in the flag (default 0)
        flag_value: the value of the flag array used to identify averaging
                    windows (default 1)
        skip_start: the number of data points to skip in the averaging after
                    the flag changes to flag_value (default 0)
        skip_end: the number of data points to skip in the averaging before the
                  flag changes from flag_value to something else (default 0)
        out_name: the name of the resultant averaged data (a column in df). If
                  not given, the default is <data_col>_<flag_col>
        interp:   interpolate the averaged data across the full time span
                  (default False). If False, the output will be NaN everywhere
                  except the index closest to the middle of each averaging
                  period.

    Returns:
        None, df is modified in-place.
    """
    if out_name is None:
        out_name = '{}_{}'.format(data_col, flag_col)

    # Replace nans in the flag, either by backfilling or setting to a given
    # value 
    if fill_nan is not None:
        df[flag_col].fillna(fill_nan, inplace=True)
    else:
        df[flag_col].fillna(method='bfill', inplace=True)

    # Identify coherent groups of a single flag value, and drop those not
    # corresponding to the falg value that we're interested in
    groups = (df[flag_col] != df[flag_col].shift()).cumsum()
    groups[df[flag_col] != flag_value] = np.nan
    groups.dropna(inplace=True)

    _groups = df.groupby(groups)

    # Create a series containing mean data values over each flag group,
    # potentially skipping data at the start and end of each group
    # TODO: this is double-plus ugly.
    means = pd.Series(
       _groups.apply(lambda x: x[data_col].iloc[skip_start:len(x)-skip_end].mean()).values,
        index=_groups.apply(lambda x: x.index[int(len(x) / 2)])
    )

    # Either interpolate back across the full index, or create a series which
    # is NaN everywhere that the means are not defined.
    if interp:
        df[out_name] = (
            means.reindex(
                df.index.union(means.index).sort_values()
            ).interpolate().loc[df.index]
        )

    else:
        df[out_name] = np.nan
        df.loc[means.index, out_name] = means.values


class Either(object):

    def __init__(self, *args, name=None):
        self.options = args
        if name is None:
            raise ValueError('name must be given')

    def __eq__(self, other):
        if other in self.options:
            return True

        try:
            if other.options == self.options and type(other) == type(self):
                return True
        except AttributeError:
            pass

        return False

    def __str__(self):
        return '<{}>'.format(', '.join(self.options))

    def __repr__(self):
        _repr = 'Either('
        for i, option in enumerate(self.options):
            _repr += '{!r}'.format(option)
            if i != len(self.options) - 1:
                _repr += ', '
        _repr += ')'
        return _repr

    def __contains__(self, item):
        return item in self.options

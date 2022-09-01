import datetime
import importlib
from typing import Sequence
import numpy as np
import pandas as pd

from scipy.stats import mode

pd_freq = {
    1: '1S',
    2: '500L',
    4: '250L',
    8: '125L',
    10: '100L',
    16: '62500U',
    20: '50L',
    50: '20L',
    32: '31250000N',
    64: '15625000N',
    128: '7812500N',
    256: '3906250N'
}


def infer_freq(index, mode_conf=0.99):
    diff = index[1:] - index[:-1]
    a = mode(diff)
    if (a.count[0] / len(index)) >= mode_conf:
        return f'{a.mode[0]}N'

    raise ValueError('Cannot infer frequency')




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


def unwrap_array(ang):
    """
    Removes dicontinuities in directional (angular) data, similar to np.unwrap,
    but less good.

    Args:
        ang: the angular array to unwrap, expects a pd.Series.
    """
    temp = ang.diff().abs() > 180
    temp = ang[temp].dropna()
    indicies = temp.index

    diffed = ang.diff()
    for i in indicies:
        if diffed.loc[i] > 0:
            ang.loc[ang.index >= i] -= 360
        else:
            ang.loc[ang.index >= i] += 360

    return ang


def flagged_avg(df, flag_col, data_col, fill_nan=None, flag_value=1,
                skip_start=0, skip_end=0, out_name=None, interp=False,
                interp_method='linear'):
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
            ).interpolate(method=interp_method).loc[df.index]
        )

    else:
        df[out_name] = np.nan
        df.loc[means.index, out_name] = means.values


def try_to_call(call, compliance=False):
    try:
        if compliance:
            if callable(call):
                return '<derived from file>'
        return call()
    except TypeError:
        return call


def slrs(wow, ps, roll, min_length=120, max_length=None, roll_lim=3,
         ps_lim=2, roll_mean=5, freq=1):

    def _add_slrs(slrs, group):
        _gdf = group[1]
        if max_length is None:
            slrs.append(_gdf)
            return
        num_groups = len(_gdf) // max_length
        last_index = -(len(_gdf) % max_length)
        _gdfs = np.split(_gdf.iloc[:last_index], num_groups)
        slrs += _gdfs

        rem = _gdf.iloc[last_index:]
        if len(rem) >= min_length:
            slrs.append(rem)

    window_size = min_length
    slrs = []

    if max_length is not None and max_length < min_length:
        raise ValueError('max_length must be >= min_length')

    _df = pd.DataFrame()
    _df['WOW_IND'] = wow
    _df['PS_RVSM'] = ps
    _df['ROLL_GIN'] = roll
    _df = _df.asfreq('1S')

    # Drop an data on the ground
    _df.loc[_df.WOW_IND == 1] = np.nan
    _df.dropna(inplace=True)

    # Check the variance of the static pressure is sufficiently small
    _df['PS_C'] = _df.PS_RVSM.rolling(
        window_size, center=True
    ).std() < ps_lim

    # Check that the range of the GIN roll is inside acceptable limits
    _df['ROLL_C'] = _df.ROLL_GIN.rolling(roll_mean).mean().rolling(
        window_size, center=True
    ).apply(np.ptp, raw=True) < roll_lim

    # Identify discontiguous regions which pass the selection criteria
    # and group them
    _df['_SLR'] = (_df['PS_C'] & _df['ROLL_C']).astype(int)
    _df['_SLRCNT'] = (_df._SLR.diff(1) != 0).astype('int').cumsum()
    groups = _df.groupby(_df._SLRCNT)

    slrs = []
    for group in groups:
        _df = group[1]
        if _df._SLR.mean() == 0:
            continue
        if len(_df) < window_size:
            continue

        # Add slrs to list, splitting if required
        _add_slrs(slrs, group)

    return [i.asfreq('{0:0.0f}N'.format(1e9/freq)).index for i in slrs]


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


def stringify_if_datetime(dt):
        """
        Stringify a datetime like object. An object is assumed to be datetime
        like if it has a strftime method.

        If dt has an hour attribute, the full time is returned, otherwise just
        the date.

        Args:
            dt - on object

        Returns:
            an iso formatted date/time string if dt supports strftime, otherwise
            dt
        """
        if not (hasattr(dt, 'strftime') and callable(dt.strftime)):
            return dt

        if hasattr(dt, 'hour'):
            return dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        return dt.strftime('%Y-%m-%d')


def make_definition(pp_group, standard):
    
    from ppodd.pod.base import pp_register
    from vocal import schema_types
    from ruamel.yaml import YAML
    
    def make_variable(var, flag=False):
        _meta = {
            'name': var.name if not flag else f'{var.name}_FLAG',
            'datatype': schema_types.Float32 if not flag else schema_types.Integer8,
            'required': False
        }

        if not flag:
            
            _attributes = var.attrs()
            #_attributes['coordinates'] = schema_types.DerivedString
        else:
            _attributes = dict(var.flag.cfattrs)
            for item, value in _attributes.items():
                if isinstance(value, Sequence):
                    if isinstance(value, str):
                        continue
                    value = list(value)
                    try:
                        value = [i.item() for i in value]
                    except Exception:
                        pass
                try:
                    value = value.item()
                except Exception:
                    pass
                _attributes[item] = value

            if 'flag_masks' in _attributes or 'flag_values' in _attributes:
                _attributes['flag_masks'] = schema_types.DerivedByteArray

            if 'valid_range' in _attributes:
                _attributes['valid_range'] = [schema_types.DerivedByte, schema_types.DerivedByte]
                

        #if 'actual_range' in _attributes:
        #    _attributes['actual_range'] = [
        #        schema_types.DerivedFloat32, schema_types.DerivedFloat32
        #    ]

        _attributes['comment'] = schema_types.OptionalDerivedString

        # These are variable attributes which we expect to be derived from file
        # on a case-by-case basis
        var_vars = (
            'sensor_serial_number', 'instrument_serial_number', 'flag_meanings',
            'sensor_type', 'sensor_manufacturer', 'sensor_model'
        )
        for _var in var_vars:
            if _var in _attributes:
                _attributes[_var] = schema_types.DerivedString

        _dimensions = ['Time']
        if var.frequency > 1:
            _dimensions.append(f'sps{var.frequency:02d}')

        return {
            'meta': _meta,
            'attributes': _attributes,
            'dimensions': _dimensions
        }
    
    standard = importlib.import_module(standard)

    dimensions_to_add = {}
    
    _dataset = {
        'meta': {
            'file_pattern': 'core'
        }, 
        'attributes': {},
        'variables': [
            {
                'meta': {
                    'name': 'Time',
                    'datatype': '<int32>'
                },
                'dimensions': ['Time'],
                'attributes': {
                    'long_name': 'Time of measurement',
                    'standard_name': 'time',
                    'calendar': 'gregorian',
                    'coverage_content_type': 'coordinate',
                    'frequency': 1,
                    'units': schema_types.DerivedString
                }
            }
        ],
        'dimensions': [{
            'name': 'Time',
            'size': None
        }]
    }

    for module in pp_register.modules(pp_group, date=datetime.date.today()):
        instance = module.test_instance()
        instance.process()
        instance.finalize()
        for var in instance.dataset.outputs:
            dim_name = f'sps{var.frequency:02d}'
            if dim_name not in dimensions_to_add:
                dimensions_to_add[dim_name] = {
                    'name': dim_name,
                    'size': var.frequency
                }
            if var.write:
                exists = len([i for i in _dataset['variables'] if i['meta']['name']==var.name])
                if exists:
                    continue
                _dataset['variables'].append(make_variable(var))

                if var.flag is not None:
                    _dataset['variables'].append(make_variable(var, flag=True))

    for dim_to_add in dimensions_to_add.values():
        _dataset['dimensions'].append(dim_to_add)

    _dataset['dimensions'].sort(key=lambda x: x['size'] if x['size'] is not None else -9e99)
    
    

    import pprint
    pprint.pprint(_dataset)
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    with open('decades_definition.yaml', 'w') as f:
        yaml.dump( _dataset, f)

    
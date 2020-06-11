from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from .parser import parser_f


def timestamp_func(d3):
    """
    Create function to calculate interpolated time stamps

    The d3 sentence provides date and time stamps for that data
    line. Use linear interpolation based on the row number to
    produce a function to calculate datetime stamps for all data
    lines.

    :params d3:
    :type d3: dict

    .. TODO::
        Change so can input arrays into returned func

    """
    dt = np.array([
        (datetime.combine(d, t)).replace(tzinfo=None)
        for (d, t) in list(zip(d3['data']['date'], d3['data']['time']))
    ])

    # Convert to milliseconds to do fitting
    delta_dt = np.array([
        timedelta.total_seconds(dt_ - dt[0]) * 1e6
        for dt_ in dt
    ]).astype('timedelta64[us]')

    row_nums = d3['row']

    try:
        delta_fit = interp1d(
            row_nums, delta_dt, kind='linear', fill_value='extrapolate',
            assume_sorted=True
        )
    except ValueError:
        # Did not converge, probably not enough data points
        return None
    else:
        def _f(r_):
            return dt[0] + timedelta(
                microseconds=int(np.around(delta_fit(r_)))
            )
        return _f


def get_frequency(timestamp):
    """
    The frequency of the different data sentences of the SEADAS probe can
    be adjusted in the software. This small function estimates the most likely
    frequency setting using the timedelta of consecutive timestamps.

    :param timestamp: array of timestamps for every data line
    :type timestamp: numpy.array with numpy.datetime64 elements
    :returns: frequency
    :rtype: int
    """

    possible_freq_setting = [20, 10, 5, 2, 1]
    tolerance = 20  # in percent

    if timestamp.size < 3:
        return None

    _f = 1. / np.median((timestamp[1:]-timestamp[:-1])).item().total_seconds()

    for pfs in possible_freq_setting:

        if np.isclose(_f, pfs, rtol=tolerance/100.):
            return pfs

    return None


def to_dataframe(ifile, rtn_all=False):
    """
    returns a dictionary where each item holds the data for a data sentence.

    :param ifile: input file
    :key rtn_all: set to `True` if all data sentences should be parsed
    :return: dictionary of pandas.DataFrame

    :Example:

      In [1]: ifile = 'seaprobe_20171214_090141_C072.wcm'

      In [2]: d = to_dataframe(ifile, rtn_all=True)

      In [3]: print(d.keys())
      ['c0', 'd2', 'd3', 'd0', 'd1']

      In [4]: d['d0'].head()

###   TODO: If rtn_all is False then merge 'normal' data files and return
      single dataframe instead of dict of dataframes. If rtn_all is True
      then return dictionary of dataframes (?)

    """
    default_sentences = ['d0', 'd3', 'c0']

    # Read the wcm txt file into raw_data as a 1D-numpy.array of strings
    with open(ifile) as f:
        raw_data = np.genfromtxt(f, dtype='S')

        if len(raw_data) == 0:
            # Empty file so return
            return None

    raw_data = np.array([i.decode() for i in raw_data])

    # Create a list of sentence ids
    # Lines that do not conform to type are ignored
    mtype = np.array(
        [l.split(',')[0] for l in raw_data if l.split(',')!=[]]
    )

    if rtn_all is True:
        sentence_id = np.unique(mtype)
    else:
        # Only return d0, d3 and c0 by default
        # Those sentences are all that is
        # required for calculation of water content
        sentence_id = np.asarray(default_sentences)

    # Dictionary of raw and parsed data sentences
    # 'row' key is array of row numbers in file
    wcm = {'raw': {k: raw_data[(mtype == k)] for k in sentence_id},
           'parsed': {k: {'description': parser_f[k]['descr'],
                          'row': np.where(mtype == k)[0]} for k in sentence_id}}

    # Parse raw data sentences
    for k in sentence_id:
        # skip invalid lines instead of raising error
        wcm['parsed'][k]['data'] = np.genfromtxt(
            wcm['raw'][k], delimiter=',',
            dtype=list(zip(parser_f[k]['names'], parser_f[k]['dtypes'])),
            converters=parser_f[k]['converters'],
            invalid_raise=False
        )

    # Determine interpolated time stamp function
    dt_func = timestamp_func(wcm['parsed']['d3'])

    if dt_func is None:
        msg = 'SEA file too short for timestamp interpolation.'
        if ppodd:
            ppodd.logger.info(msg)
        else:
            sys.stdout.write(msg+'\n')
        return None

    df_dic = {}
    for k in list(wcm['parsed'].keys()):
        wcm['parsed'][k]['dt'] = np.array(
            [np.datetime64(dt_func(t_).replace(tzinfo=None))
             for t_ in wcm['parsed'][k]['row']]
        )

        freq = get_frequency(wcm['parsed'][k]['dt'])
        wcm['parsed'][k]['f'] = freq
        ts_start = wcm['parsed'][k]['dt'][0]
        ts_end = wcm['parsed'][k]['dt'][-1]

        # Extract data from parsed dict as a list of arrays for each column
        data_list = []
        for name in parser_f[k]['names']:
            data_list.append(wcm['parsed'][k]['data'][name])

        # Create 2D array of data for conversion to dataframe
        # Attempt to do some error catching to cope with mis-formed data
        try:
            data_array = np.column_stack(data_list)
        except:
            import pdb
            pdb.set_trace()

        # Create dataframe with datetime index. Note that index has been
        # (possibly) truncated to cope with partial lines at the end of
        # the file. Normally, len(data_array) == len(wcm['parsed'][k]['dt'])
        df = pd.DataFrame(data_array,
                          columns=parser_f[k]['names'],
                          index=wcm['parsed'][k]['dt'][:len(data_array)])

        dd = dict(zip(parser_f[k]['names'], parser_f[k]['dtypes']))
        for name in parser_f[k]['names']:
            if 'float' in dd[name]:
                df[name] = df[name].astype(float)

        if freq:
            # create a new index for resampling the irregular data
            # This new Index starts on a full second and is the correct frequency
            newIndex = pd.date_range(start=np.datetime64(ts_start, 's'),
                                     end=np.datetime64(ts_end, 's'),
                                     freq='%ims' % (1000/freq,), closed='left')

            df = df.reindex(index=newIndex, method='nearest')

        df_dic[k] = df.copy()

    _meta = {}
    # Add probe metadata as dictionary accessor
    for el in ['TWC','083','021','CMP']:
#
#        # Create list of valid element variables
        k = [s_ for s_ in ['l','w','f','s','o'] if 'el{}_{}'.format(el,s_) in df_dic['c0']]
#
#        # Add these parameters to the sea metadata. All rows are the same
#        df_dic['d0'].ppodd.set_sea_meta('el'+el,
#                                        {_k:df_dic['c0']['el{}_{}'.format(el,_k)][0] for _k in k})
        _meta['el' + el] = {
            _k:df_dic['c0']['el{}_{}'.format(el,_k)][0] for _k in k
        }
#
#    # Add serial number
#    df_dic['d0'].ppodd.set_sea_meta('sea',{'sn': df_dic['c0']['sn'][0]})
    _meta['sn'] = df_dic['c0']['sn'][0]

    return df_dic, _meta

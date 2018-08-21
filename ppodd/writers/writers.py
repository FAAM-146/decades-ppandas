import abc

import pandas as pd

from netCDF4 import Dataset

from ..utils import pd_freq


class DecadesWriter(abc.ABC):

    def __init__(self, dataset):
        self.dataset = dataset

    @abc.abstractmethod
    def write(self, filename):
        """Write some output"""


class NetCDFWriter(DecadesWriter):

    def _create_file(self, filename, indicies):
        self.nc = Dataset(filename, 'w')

        self.nc.createDimension('Time', None)
        sps_vars = {}
        for sps in indicies:
            if sps == 1:
                continue
            sps_name = 'sps{0:02d}'.format(sps)
            sps_vars[sps] = self.nc.createDimension(sps_name, sps)

        for output in self.dataset.outputs:
            print('writing {}'.format(output.long_name))
            for _var in output:
                if 'FLAG' in _var:
                    is_flag = True
                    _datatype = 'b'
                    _fill_value = -1
                else:
                    is_flag = False
                    _datatype = 'f'
                    _fill_value = -9999

                if output.frequency == 1:
                    _dims = ('Time',)
                else:
                    _dims = ('Time', 'sps{0:02d}'.format(output.frequency))

                ncvar = self.nc.createVariable(
                    _var, _datatype, _dims, zlib=True, fill_value=_fill_value,
                    complevel=9
                )

                if not is_flag:
                    setattr(ncvar, 'long_name', output.long_name)
                    if output.standard_name:
                        setattr(ncvar, 'standard_name', output.standard_name)
                    setattr(ncvar, 'units', output.units)
                    setattr(ncvar, 'frequency', int(output.frequency))
                else:
                    setattr(ncvar, 'long_name', 'Flag for {}'.format(
                            output.long_name))
                    setattr(ncvar, 'units', '1')
                    setattr(ncvar, 'frequency', int(output.frequency))

                data = output[_var].reindex(
                    indicies[output.frequency]
                ).interpolate(limit=1)

                if 'FLAG' in _var:
                    data.fillna(3, inplace=True)
                else:
                    data.fillna(-9999, inplace=True)

                ncvar[:] = data.values.reshape(
                    (len(indicies[1]), output.frequency)
                )

        time_var = self.nc.createVariable('Time', int, ('Time',))
        secs = (indicies[1] - pd.to_datetime(indicies[1].date)).total_seconds()
        time_var[:] = secs

        self.nc.close()

    def write(self, filename):

        start_index = None
        end_index = None
        freqs = []

        for output in self.dataset.outputs:
            if start_index is None or output.index[0] < start_index:
                start_index = output.index[0]

            if end_index is None or output.index[-1] > end_index:
                end_index = output.index[-1]

            if output.frequency not in freqs:
                freqs.append(output.frequency)

        print('start_index: {}'.format(start_index))
        print('end_index: {}'.format(end_index))
        print('freqs: {}'.format(freqs))

        output_index = {}

        for freq in freqs:
            output_index[freq] = pd.DatetimeIndex(
                start=start_index,
                end=end_index,
                freq=pd_freq[freq]
            )

            _mod = len(output_index[freq]) % freq
            if _mod:
                output_index[freq] = output_index[freq][:-_mod]
            if freq == 1:
                output_index[1] = output_index[1][:-1]

        try:
            self._create_file(filename, output_index)
        except Exception:
            self.nc.close()
            raise

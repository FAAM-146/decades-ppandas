from collections import OrderedDict

import numpy as np
import pandas as pd

from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from .base import PPBase
from .shortcuts import _z

parameters = OrderedDict()
parameters['VMR'] = {
    'units': 'ppmv',
    'long_name': 'Water Vapour Measurement from {ins} {interp}'
}
parameters['PRESS'] = {
    'units': 'hPa',
    'long_name': 'Pressure inside {ins} sample cell {interp}'
}
# parameters['TEMP'] = {
#     'units': 'deg C',
#     'long_name': 'Temperature of {ins} sample cell {interp}'
# }


VMR_VALID_MAX = 40000
VMR_VALID_MIN = 50


class WVSS2(object):

    SERIAL_STR_LEN = 11

    def __init__(self, *args, **kwargs):
        self.inputs = [
            '{}_ident'.format(self.unit),
            '{}_utc_time_msec'.format(self.unit),
            '{}_serial_data'.format(self.unit)
        ]

        super().__init__(*args, **kwargs)

    def declare_outputs(self):
        """
        Declare module outputs. These are defined in the module-level
        parameters OrderedDict.
        """
        try:
            self._ident = self.dataset[
                '{}_ident'.format(self.unit)
            ].data[0]

            if type(self._ident) is bytes:
                self._ident = self._ident.decode()

        except KeyError:
            if not self.test_mode:
                return

            self._ident = self.__class__.test()[
                '{}_ident'.format(self.unit)
            ][1][0].decode()

        for name, attrs in parameters.items():
            self.declare(
                'WVSS2{}_{}'.format(self._ident, name),
                frequency=1,
                comment=('WVSS-II measurements rely on manufacturer '
                         'calibrations, and are not tracable to national '
                         'standards'),
                **attrs
            )

    def process(self):
        serial_data_key = '{}_serial_data'.format(self.unit)
        millisecond_key = '{}_utc_time_msec'.format(self.unit)

        self.get_dataframe()

        # Split the serial strings
        serial_data = np.chararray.split(
            [i for i in self.d[serial_data_key]]
        )

        # Get the indicies with the correct number of elements.
        good_indicies = (
            np.array([len(i) for i in serial_data]) == self.SERIAL_STR_LEN
        )

        # Any serial strings of incorrect length are NaN'd and dropped
        self.d.loc[~good_indicies] = np.nan
        self.d.dropna(inplace=True)
        serial_data = serial_data[good_indicies]

        # Stack the data into a 2-d numpy array
        serial_data = np.stack(serial_data).astype(float)

        # Insert serial data defined in parameters into the dataframe
        for i, name in enumerate(parameters.keys()):
            column = 'WVSS2{}_{}'.format(self._ident, name)
            self.d[column] = serial_data[:, i]

        # Add the millisecond data onto the index
        delta = pd.to_timedelta(self.d[millisecond_key], 'ms')
        self.d.index += delta

        # Interpolate onto a 1 Hz time base. Do not allow gaps of more than 3
        # datapoints at this frequency
        start_index = self.d.index[0].round('1S')
        end_index = self.d.index[-1].round('1S')
        new_index = pd.date_range(start_index, end_index, freq='1S')

        interp_df = self.d.reindex(
            self.d.index.union(new_index).sort_values()
        ).interpolate('time', limit=3).loc[new_index]

        # Output data
        for name in parameters.keys():
            _key = 'WVSS2{}_{}'.format(self._ident, name)
            _vmr_key = 'WVSS2{}_VMR'.format(self._ident)

            interp_df['VMR_RANGE_FLAG'] = 0
            interp_df['DATA_MISSING_FLAG'] = 0

            # Flag VMR outside the design specifications of the instrument
            if _key is _vmr_key:

                interp_df.loc[
                    interp_df[_key] > VMR_VALID_MAX, 'VMR_RANGE_FLAG'
                ] = 1

                interp_df.loc[
                    interp_df[_key] < VMR_VALID_MIN, 'VMR_RANGE_FLAG'
                ] = 1

            interp_df.loc[
                ~np.isfinite(interp_df[_key]), 'DATA_MISSING_FLAG'
            ] = 3

            var = DecadesVariable(interp_df[_key], name=_key,
                                  flag=DecadesBitmaskFlag)

            # Fill in the gaps of the long_name attribute
            long_name = self.declarations[var.name]['long_name']
            long_name = long_name.format(
                ins='WVSS2{}'.format(self._ident),
                interp='linearly interpolated to 1 Hz'
            )
            self.declarations[var.name]['long_name'] = long_name

            if _key == _vmr_key:
                var.flag.add_mask(interp_df.VMR_RANGE_FLAG, flags.OUT_RANGE)
            var.flag.add_mask(interp_df.DATA_MISSING_FLAG, flags.DATA_MISSING)

            self.add_output(var)


class WVSS2A(WVSS2, PPBase):
    unit = 'WVSS2A'

    @staticmethod
    def test():
        return {
            'WVSS2A_ident': ('data', [b'F'] * 100),
            'WVSS2A_utc_time_msec': ('data', _z(100)),
            'WVSS2A_serial_data':  (
                'data', b'14500  990  35  12000 20000 260 72 50 50 1 1\n'
            )
        }


class WVSS2B(WVSS2, PPBase):
    unit = 'WVSS2B'

    @staticmethod
    def test():
        return {
            'WVSS2B_ident': ('data', [b'R'] * 100),
            'WVSS2B_utc_time_msec': ('data', _z(100)),
            'WVSS2B_serial_data':  (
                'data', b'14500  990  35  12000 20000 260 72 50 50 1 1\n'
            )
        }

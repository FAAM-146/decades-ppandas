from .base import PPBase
from .shortcuts import _o, _z
from ..decades import DecadesVariable


class BBRSols(PPBase):

    inputs = [
        'CALUP1S', 'CALUP2S', 'CALUIRS', 'CALLP1S', 'CALLP2S', 'CALLIRS',
        'UPPBBR_radiometer_1_sig',
        'UPPBBR_radiometer_2_sig',
        'UPPBBR_radiometer_3_sig',
        'UPPBBR_radiometer_1_zero',
        'UPPBBR_radiometer_2_zero',
        'UPPBBR_radiometer_3_zero',
        'UPPBBR_radiometer_1_temp',
        'UPPBBR_radiometer_2_temp',
        'UPPBBR_radiometer_3_temp',
        'LOWBBR_radiometer_1_sig',
        'LOWBBR_radiometer_2_sig',
        'LOWBBR_radiometer_3_sig',
        'LOWBBR_radiometer_1_zero',
        'LOWBBR_radiometer_2_zero',
        'LOWBBR_radiometer_3_zero',
        'LOWBBR_radiometer_1_temp',
        'LOWBBR_radiometer_2_temp',
        'LOWBBR_radiometer_3_temp'
    ]

    @staticmethod
    def test():
        return {
            'CALUP1S': ('const', [6, .05, 128., -6e-3]),
            'CALUP2S': ('const', [1, .03, 128., -6e-3]),
            'CALUIRS': ('const', [-1e3, 0.1, 134, -6e-3]),
            'CALLP1S': ('const', [.6, .04, 128, -6e-3]),
            'CALLP2S': ('const', [.25, .025, 128, -6e-3]),
            'CALLIRS': ('const', [-500, -.03, 134, -6e-3]),
            'UPPBBR_radiometer_1_sig': ('data', 15000 * _o(100)),
            'UPPBBR_radiometer_2_sig': ('data', 7000 * _o(100)),
            'UPPBBR_radiometer_3_sig': ('data', -54 * _o(100)),
            'UPPBBR_radiometer_1_zero': ('data', _z(100)),
            'UPPBBR_radiometer_2_zero': ('data', _z(100)),
            'UPPBBR_radiometer_3_zero': ('data', _z(100)),
            'UPPBBR_radiometer_1_temp': ('data', 20000 * _o(100)),
            'UPPBBR_radiometer_2_temp': ('data', 20000 * _o(100)),
            'UPPBBR_radiometer_3_temp': ('data', _z(100)),
            'LOWBBR_radiometer_1_sig': ('data', 8000 * _o(100)),
            'LOWBBR_radiometer_2_sig': ('data', 8000 * _o(100)),
            'LOWBBR_radiometer_3_sig': ('data', -38 * _o(100)),
            'LOWBBR_radiometer_1_zero': ('data', _z(100)),
            'LOWBBR_radiometer_2_zero': ('data', _z(100)),
            'LOWBBR_radiometer_3_zero': ('data', _z(100)),
            'LOWBBR_radiometer_1_temp': ('data', 20000 * _o(100)),
            'LOWBBR_radiometer_2_temp': ('data', 20000 * _o(100)),
            'LOWBBR_radiometer_3_temp': ('data', -10000 * _o(100))
        }

    def declare_outputs(self):
        """
        Declare the module outputs. These are interim products used downstream
        in the processing chain, and as such should not normally be written to
        file.
        """

        self.declare('UP1S', write=False, units='W m-2', frequency=1,
                     long_name='UPP VIS CLR SIG')

        self.declare('UP2S', write=False, units='W m-2', frequency=1,
                     long_name='UPP VIS RED SIG')

        self.declare('UIRS', write=False, units='W m-2', frequency=1,
                     long_name='UPP I/R SIGNAL')

        self.declare('UP1Z', write=False, units='W m-2', frequency=1,
                     long_name='UPP VIS CLR ZERO')

        self.declare('UP2Z', write=False, units='W m-2', frequency=1,
                     long_name='UPP VIS RED ZERO')

        self.declare('UIRZ', write=False, units='W m-2', frequency=1,
                     long_name='UPP I/R ZERO')

        self.declare('UP1T', write=False, units='degree C', frequency=1,
                     long_name='UPP VIS CLR TEMP')

        self.declare('UP2T', write=False, units='degree C', frequency=1,
                     long_name='UPP VIS RED TEMP')

        self.declare('UIRT', write=False, units='degree C', frequency=1,
                     long_name='UPP I/R TEMP')

        self.declare('LP1S', write=False, units='W m-2', frequency=1,
                     long_name='LWR VIS CLR SIG')

        self.declare('LP2S', write=False, units='W m-2', frequency=1,
                     long_name='LWR VIS RED SIG')

        self.declare('LIRS', write=False, units='W m-2', frequency=1,
                     long_name='LWR I/R SIGNAL')

        self.declare('LP1Z', write=False, units='W m-2', frequency=1,
                     long_name='LWR VIS CLR ZERO')

        self.declare('LP2Z', write=False, units='W m-2', frequency=1,
                     long_name='LWR VIS RED ZERO')

        self.declare('LIRZ', write=False, units='W m-2', frequency=1,
                     long_name='LWR I/R ZERO')

        self.declare('LP1T', write=False, units='degree C', frequency=1,
                     long_name='LWR VIS CLR TEMP')

        self.declare('LP2T', write=False, units='degree C', frequency=1,
                     long_name='LWR VIS RED TEMP')

        self.declare('LIRT', write=False, units='degree C', frequency=1,
                     long_name='LWR I/R TEMP')

    def process(self):
        """
        Processing entry point.
        """
        self.get_dataframe()
        d = self.d

        # Upper and lower BBRs
        for bbr in ('UPP', 'LOW'):

            # Each dome provides signal, zero, and thermopile temperature
            for measurement in ('sig', 'zero', 'temp'):

                # Three possible position for domes. Nominally clear, red and
                # IR
                for dome, dome_n in zip(('P1', 'P2', 'IR'), (1, 2, 3)):

                    # Define the input name for the current instrument
                    _var = '{bbr}BBR_radiometer_{dome_n}_{measurement}'.format(
                        bbr=bbr, dome_n=dome_n, measurement=measurement
                    )

                    # Define the output name for the current instrument
                    _varout_name = '{pos}{dome}{measurement}'.format(
                        pos=bbr[0], dome=dome,
                        measurement=measurement[0].upper()
                    )

                    # Get the correct calibration constants for the current
                    # signal
                    _cals = 'CAL{pos}{dome}S'.format(
                        pos=bbr[0], dome=dome
                    )

                    if measurement in ('sig', 'zero'):
                        _cal = self.dataset[_cals][:2]
                        _offs = 0
                    else:
                        _cal = self.dataset[_cals][2:]
                        _offs = 273.15

                    # Calibrate the signal with gradient / intercept and add an
                    # offset (K -> C) for the thermistor signal
                    d[_varout_name] = d[_var] * _cal[1] + _cal[0] + _offs

                    # Add the current output to the parent dataset
                    self.add_output(
                        DecadesVariable(d[_varout_name], name=_varout_name)
                    )

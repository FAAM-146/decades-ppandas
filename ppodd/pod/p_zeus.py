from ..decades import DecadesVariable, DecadesBitmaskFlag
from ..decades import flags
from .base import PPBase, register_pp
from .shortcuts import _o, _c, _z

@register_pp('core')
class ElectricFieldZeus(PPBase):
    r"""
    Provides the electric field measurement from the Zeus instrument. No processing
    is actually performed, the data reported by the instrument is simply passed
    through to the output, with the addition of a flag.

    ZEUS is a window mounted static mill created by the MetOffice OBR group. It uses a 
    sensor designed by Dr John Chubb (JCI 140) made by Hearn Morley/Fraser Anti-Static 
    and an Arduino Uno to digitize the analogue signal. It is fitted in SW1.
    """

    inputs = [
        'ZEUS00_zeus_eField',  # ZEUS electric field
        'WOW_IND'
    ]

    @staticmethod
    def test():
        """
        Provide dummy input data for testing.
        """ 
        return {
            'ZEUS00_zeus_eField': ('data', _o(100), 1),
            'WOW_IND': ('data', _c([_o(10), _z(80), _o(10)]), 1)
        }

    def declare_outputs(self):
        
        self.declare(
            'EXX_ZEUS',
            units='kV m-1',
            frequency=1,
            long_name=('Electric field measured by the Zeus instrument'),
            instrument_manufacturer='Met Office OBR, Hearn Morley, Fraser Anti-Static',
            instrument_serial_number='12230002'
        )


    def process(self):
        """
        Processing entry hook.
        """
        self.get_dataframe()

        exx = DecadesVariable(
            self.d['ZEUS00_zeus_eField'], name='EXX_ZEUS',
            flag=DecadesBitmaskFlag
        )

        exx.flag.add_mask(
            self.d['WOW_IND'],
            flags.WOW,
            'Aircraft is on the ground, as indicated by weight-on-wheels'
        )

        self.add_output(exx)